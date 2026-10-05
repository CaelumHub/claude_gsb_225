"""
processing.py — Single-file audio operations shared by the one-off API routes
and by the batch queue (:mod:`backend.batch`).

Every operation follows the same *render-then-commit* contract so that neither
a failure nor a cancellation can ever leave a half-finished file in the
library:

1. Render to a caller-provided destination path outside the library registry
   (batch jobs use their scratch directory).
2. On success register the finished file atomically in the library.
3. On failure / cancellation the partial output is removed and nothing is
   registered.

All processing is streaming with bounded memory and reports progress through
``on_progress(fraction)``; cancellation is cooperative via ``should_cancel()``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable, Dict, List, Optional, Tuple

from . import audio_io, effects, storage

ProgressFn = Optional[Callable[[float], None]]
CancelFn = Optional[Callable[[], bool]]


# --------------------------------------------------------------------------- #
# Source normalisation
# --------------------------------------------------------------------------- #

def ensure_wav(path: str, on_progress: ProgressFn = None,
               should_cancel: CancelFn = None) -> Tuple[str, Optional[str]]:
    """Return ``(wav_path, temp_to_clean)``.

    Library files are normally WAV; a file exported to a compressed container
    is decoded to a temporary WAV first.
    """
    if path.lower().endswith(".wav"):
        return path, None
    tmp = audio_io.decode_with_ffmpeg(path, on_progress=on_progress,
                                      should_cancel=should_cancel)
    return tmp, tmp


# --------------------------------------------------------------------------- #
# Streaming edit (trim / silence / fade / gain / normalize / reverse)
# --------------------------------------------------------------------------- #

def apply_edit(src_path: str, dst_path: str, op: str, params: Dict,
               on_progress: ProgressFn = None,
               should_cancel: CancelFn = None) -> None:
    """Streaming waveform edit; same semantics as the waveform-editor route."""
    def _check():
        if should_cancel and should_cancel():
            raise audio_io.CancelledError("processing cancelled")

    with audio_io.WavReader(src_path) as r:
        sr = r.sr
        ch = r.channels
        total = r.nframes

        if op == "reverse":
            bufs = [[] for _ in range(ch)]
            done = 0
            for chunk in r.iter_chunks():
                _check()
                for c, ch_data in enumerate(chunk):
                    bufs[c].extend(ch_data)
                done += len(chunk[0])
                if on_progress:
                    on_progress(min(0.95, done / (total or 1)))
            _check()
            with audio_io.WavWriter(dst_path, sr, ch, 2) as w:
                w.write_chunk([b[::-1] for b in bufs])
            if on_progress:
                on_progress(1.0)
            return

        start_f = int(float(params.get("start", 0)) * sr)
        end_f = int(float(params.get("end", total / sr)) * sr)
        start_f = max(0, min(start_f, total))
        end_f = max(start_f, min(end_f, total))

        gain = 10.0 ** (float(params.get("gain_db", 0)) / 20.0)
        fade_in = float(params.get("fade_in", 0)) * sr
        fade_out = float(params.get("fade_out", 0)) * sr

        # Normalise needs a peak pre-scan; it occupies the first 10 % of progress.
        prog_base = 0.0
        if op == "normalize":
            peak = 0.0
            with audio_io.WavReader(src_path) as rr:
                done = 0
                for chunk in rr.iter_chunks():
                    _check()
                    for c in chunk:
                        for v in c:
                            a = abs(v)
                            if a > peak:
                                peak = a
                    done += len(chunk[0])
                    if on_progress:
                        on_progress(min(0.1, 0.1 * done / (total or 1)))
            gain = (float(params.get("peak", 0.99)) / peak) if peak > 1e-9 else 1.0
            op = "gain"
            prog_base = 0.1

        with audio_io.WavWriter(dst_path, sr, ch, 2) as w:
            pos = 0
            done = 0
            while True:
                _check()
                chunk = r.read_chunk(1 << 16)
                if chunk is None:
                    break
                n = len(chunk[0])
                out = []
                for c in chunk:
                    o = []
                    for i, v in enumerate(c):
                        gpos = pos + i
                        x = v
                        if op == "gain":
                            x = x * gain
                        if op == "trim":
                            if gpos < start_f or gpos >= end_f:
                                x = 0.0
                        if op == "silence":
                            if start_f <= gpos < end_f:
                                x = 0.0
                        if op == "fade":
                            if fade_in > 0 and gpos < fade_in:
                                x *= gpos / fade_in
                            if fade_out > 0 and gpos >= total - fade_out:
                                x *= (total - gpos) / fade_out
                        o.append(x)
                    out.append(o)
                w.write_chunk(out)
                pos += n
                done += n
                if on_progress:
                    frac = prog_base + (1.0 - prog_base) * min(1.0, done / (total or 1))
                    on_progress(frac)
        if on_progress:
            on_progress(1.0)


# --------------------------------------------------------------------------- #
# Effect chain
# --------------------------------------------------------------------------- #

def apply_effects(src_path: str, dst_path: str, chain: List[Dict],
                  on_progress: ProgressFn = None,
                  should_cancel: CancelFn = None) -> None:
    wav_path, tmp = ensure_wav(src_path)
    try:
        effects.apply_chain_to_file(wav_path, dst_path, chain,
                                    on_progress=on_progress,
                                    should_cancel=should_cancel)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------- #
# Export / conversion
# --------------------------------------------------------------------------- #

def export_file(src_path: str, dst_path: str, fmt: str,
                dst_sr: Optional[int] = None, sample_width: str = "pcm16",
                channels: Optional[int] = None, bitrate: str = "192k",
                on_progress: ProgressFn = None,
                should_cancel: CancelFn = None) -> None:
    audio_io.convert(src_path, dst_path, fmt, dst_sr=dst_sr,
                     sample_width=sample_width, channels=channels,
                     bitrate=bitrate, on_progress=on_progress,
                     should_cancel=should_cancel)


# --------------------------------------------------------------------------- #
# Commit a finished file into the library registry
# --------------------------------------------------------------------------- #

def register_wav(store: "storage.Storage", source_id: Optional[str], name: str,
                 wav_path: str, extra: Optional[Dict] = None) -> Dict:
    """Move a finished WAV into the library and register it (atomic commit)."""
    file_id = storage.new_id()
    dst = os.path.join(store.audio_dir, file_id + ".wav")
    if wav_path != dst:
        shutil.move(wav_path, dst)
    with audio_io.WavReader(dst) as r:
        sr, ch, frames = r.sr, r.channels, r.nframes
    entry = {
        "id": file_id,
        "name": name,
        "path": f"audio/{file_id}.wav",
        "sr": sr,
        "channels": ch,
        "frames": frames,
        "duration": frames / sr if sr else 0.0,
        "size_bytes": os.path.getsize(dst),
        "derived_from": source_id,
    }
    entry.update(extra or {})
    return store.add_file(entry)


def register_exported(store: "storage.Storage", source_id: str, name_base: str,
                      out_path: str, fmt: str) -> Dict:
    """Register a finished exported (possibly compressed) file."""
    file_id_new = storage.new_id()
    dst = os.path.join(store.audio_dir, file_id_new + "." + fmt)
    if out_path != dst:
        shutil.move(out_path, dst)
    sr, ch, frames, duration, size = probe_file(dst, fmt)
    return store.add_file({
        "id": file_id_new,
        "name": f"{name_base}.{fmt}",
        "original_format": fmt,
        "path": f"audio/{file_id_new}.{fmt}",
        "sr": sr,
        "channels": ch,
        "frames": frames,
        "duration": duration,
        "size_bytes": size,
        "derived_from": source_id,
        "exported": True,
    })


def probe_file(path: str, fmt: str):
    """Best-effort metadata probe for a finished file (ffprobe for compressed)."""
    if fmt == "wav":
        try:
            with audio_io.WavReader(path) as r:
                return r.sr, r.channels, r.nframes, r.duration, os.path.getsize(path)
        except Exception:
            pass
    size = os.path.getsize(path)
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=sample_rate,channels:format=duration",
             "-of", "json", path], capture_output=True, timeout=60)
        import json
        info = json.loads(out.stdout)
        st = info.get("streams", [{}])[0]
        dur = float(info.get("format", {}).get("duration", 0) or 0)
        return int(st.get("sample_rate", 0) or 0), int(st.get("channels", 0) or 0), 0, dur, size
    except Exception:
        return 0, 0, 0, 0.0, size
