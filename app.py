"""
app.py — Flask application: the web front-end plus a REST API that exposes the
self-implemented DSP/MIR backend.

Run with:

    python3 app.py            # starts on http://127.0.0.1:8000

Pages (10) are rendered from the ``templates/`` directory; the API lives under
``/api/`` and the raw audio files under ``/api/audio/<id>``.
"""

from __future__ import annotations

import base64
import io
import math
import os
import shutil
import tempfile
from typing import Dict, Optional

from flask import Flask, jsonify, render_template, request, send_file

from backend import analysis, audio_io, batch, chords, effects, mixer, realtime, separation, storage

try:
    from flask_cors import CORS

    _CORS = True
except ImportError:  # pragma: no cover
    _CORS = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB upload cap
if _CORS:
    CORS(app)

store = storage.Storage(DATA_DIR)
rt = realtime.new_analyzer()

# Batch-processing queue: sequential background worker, outputs staged in a
# private temp dir and only moved into the library on success.
BATCH_TMP = os.path.join(tempfile.gettempdir(), "audio-mir-batch")
batches = batch.BatchManager(BATCH_TMP)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _abs_path(entry: Dict) -> str:
    return os.path.join(DATA_DIR, entry["path"])


def _entry(file_id: str) -> Optional[Dict]:
    return store.get_file(file_id)


def _register_derived(source_id: str, name: str, wav_path: str,
                      extra: Optional[Dict] = None) -> Dict:
    """Register a newly-created audio file that derives from ``source_id``."""
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


# --------------------------------------------------------------------------- #
# Page routes
# --------------------------------------------------------------------------- #

PAGES = {
    "library": "音频库",
    "waveform": "波形编辑",
    "spectrogram": "频谱分析",
    "pitch_beat": "音高与节拍",
    "chords": "和弦识别",
    "separation": "音源分离",
    "effects": "效果链",
    "mixer": "混音台",
    "export": "导出与转换",
    "history": "项目历史",
}


@app.route("/")
def index():
    return render_template("library.html", active="library", pages=PAGES)


@app.route("/<page>")
def page(page: str):
    if page not in PAGES:
        return render_template("library.html", active="library", pages=PAGES), 404
    return render_template(f"{page}.html", active=page, pages=PAGES)


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #

@app.get("/api/library")
def api_library():
    return jsonify(store.list_files())


@app.get("/api/library/<file_id>")
def api_file(file_id: str):
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    return jsonify(entry)


@app.patch("/api/library/<file_id>")
def api_update_file(file_id: str):
    entry = store.update_file(file_id, request.get_json(force=True) or {})
    if not entry:
        return jsonify(error="file not found"), 404
    return jsonify(entry)


@app.delete("/api/library/<file_id>")
def api_delete_file(file_id: str):
    if not store.delete_file(file_id):
        return jsonify(error="file not found"), 404
    return jsonify(ok=True)


@app.post("/api/library/upload")
def api_upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="no file provided"), 400
    name = f.filename
    ext = (name.rsplit(".", 1)[-1].lower() if "." in name else "bin")

    fd, tmp = tempfile.mkstemp(suffix="." + ext)
    os.close(fd)
    f.save(tmp)

    file_id = storage.new_id()
    dst = os.path.join(store.audio_dir, file_id + ".wav")
    try:
        if ext == "wav":
            audio_io._convert_wav(tmp, dst, None, "pcm16", None)
        else:
            decoded = audio_io.decode_with_ffmpeg(tmp)
            shutil.move(decoded, dst)
    except Exception as e:
        if os.path.exists(dst):
            os.unlink(dst)
        raise
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    with audio_io.WavReader(dst) as r:
        entry = store.add_file({
            "id": file_id,
            "name": name,
            "original_format": ext,
            "path": f"audio/{file_id}.wav",
            "sr": r.sr,
            "channels": r.channels,
            "frames": r.nframes,
            "duration": r.duration,
            "size_bytes": os.path.getsize(dst),
        })
    return jsonify(entry)


def _synth(kind: str, sr: int, duration: float, freq: float) -> list:
    """Generate a simple test signal (used by the 'generate' endpoint)."""
    n = int(sr * duration)
    t = [i / sr for i in range(n)]
    if kind == "sine":
        return [0.5 * math.sin(2 * math.pi * freq * x) for x in t]
    if kind == "sweep":
        f0, f1 = 100.0, 2000.0
        phase = 0.0
        out = []
        k = (f1 - f0) / duration
        for x in t:
            f = f0 + k * x
            phase += 2 * math.pi * f / sr
            out.append(0.5 * math.sin(phase))
        return out
    if kind == "noise":
        import random
        return [random.uniform(-0.5, 0.5) for _ in range(n)]
    if kind == "chord":
        freqs = [freq, freq * 5 / 4, freq * 3 / 2]  # major triad
        return [0.25 * sum(math.sin(2 * math.pi * f * x) for f in freqs) for x in t]
    if kind == "pluck":
        # Karplus-Strong plucked string.
        delay = max(2, int(sr / max(freq, 20)))
        buf = [0.0] * delay
        import random
        for i in range(delay):
            buf[i] = random.uniform(-1, 1)
        out = []
        idx = 0
        for _ in range(n):
            cur = buf[idx]
            nxt = buf[(idx + 1) % delay]
            avg = 0.5 * (cur + nxt) * 0.996
            buf[idx] = avg
            out.append(avg)
            idx = (idx + 1) % delay
        return out
    raise ValueError(f"unknown synth kind {kind}")


@app.post("/api/library/generate")
def api_generate():
    data = request.get_json(force=True) or {}
    kind = data.get("kind", "sine")
    sr = int(data.get("sr", 44100))
    duration = float(data.get("duration", 5.0))
    freq = float(data.get("freq", 440.0))
    duration = max(0.1, min(duration, 60.0))
    samples = _synth(kind, sr, duration, freq)

    file_id = storage.new_id()
    dst = os.path.join(store.audio_dir, file_id + ".wav")
    audio_io.save(dst, audio_io.AudioData([samples], sr))
    entry = store.add_file({
        "id": file_id,
        "name": f"{kind}-{int(freq)}Hz.wav",
        "original_format": "generated",
        "path": f"audio/{file_id}.wav",
        "sr": sr,
        "channels": 1,
        "frames": len(samples),
        "duration": duration,
        "size_bytes": os.path.getsize(dst),
    })
    return jsonify(entry)


# --------------------------------------------------------------------------- #
# Audio serving & waveform
# --------------------------------------------------------------------------- #

@app.get("/api/audio/<file_id>")
def api_audio(file_id: str):
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    return send_file(_abs_path(entry), mimetype="audio/wav")


@app.get("/api/download/<file_id>")
def api_download(file_id: str):
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    return send_file(_abs_path(entry), as_attachment=True, download_name=entry.get("name", "audio.wav"))


@app.get("/api/audio/<file_id>/waveform")
def api_waveform(file_id: str):
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    points = int(request.args.get("points", 2000))
    channel = int(request.args.get("channel", 0))
    return jsonify(analysis.waveform_envelope(_abs_path(entry), points=points, channel=channel))


@app.get("/api/audio/<file_id>/samples")
def api_samples(file_id: str):
    """Return raw mono samples for a small window (used by the waveform editor)."""
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    start = float(request.args.get("start", 0.0))
    duration = float(request.args.get("duration", 5.0))
    max_n = int(request.args.get("max", 50000))
    with audio_io.WavReader(_abs_path(entry)) as r:
        sr = r.sr
        start_f = max(0, int(start * sr))
        n = min(max_n, int(duration * sr))
        chunk = r.read_excerpt(start_f, n)
        mono = audio_io.to_mono(chunk.samples)
    return jsonify({"sr": sr, "start": start, "samples": mono})


# --------------------------------------------------------------------------- #
# Waveform edits
# --------------------------------------------------------------------------- #

def _apply_edit(src_path: str, dst_path: str, op: str, params: Dict,
                on_progress=None, is_cancelled=None) -> None:
    """Streaming waveform edit: trim / silence / fade / gain / normalize / reverse."""
    from backend.audio_io import ProcessingCancelled

    def _tick():
        if is_cancelled and is_cancelled():
            raise ProcessingCancelled("cancelled")

    with audio_io.WavReader(src_path) as r:
        sr = r.sr
        ch = r.channels
        total = r.nframes

        if op == "reverse":
            bufs = [[] for _ in range(ch)]
            done = 0
            for chunk in r.iter_chunks():
                _tick()
                for c, ch_data in enumerate(chunk):
                    bufs[c].extend(ch_data)
                done += len(chunk[0])
                if on_progress:
                    on_progress(min(0.99, done / max(1, total)))
            _tick()
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

        # Normalise needs a peak pre-scan (first ~20% of the progress budget).
        if op == "normalize":
            peak = 0.0
            scanned = 0
            with audio_io.WavReader(src_path) as rr:
                for chunk in rr.iter_chunks():
                    _tick()
                    for c in chunk:
                        for v in c:
                            a = abs(v)
                            if a > peak:
                                peak = a
                    scanned += len(chunk[0])
                    if on_progress:
                        on_progress(0.2 * min(1.0, scanned / max(1, total)))
            gain = (float(params.get("peak", 0.99)) / peak) if peak > 1e-9 else 1.0
            op = "gain"

        done = 0
        with audio_io.WavWriter(dst_path, sr, ch, 2) as w:
            pos = 0
            while True:
                _tick()
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
                    on_progress(min(0.99, done / max(1, total)))
        if on_progress:
            on_progress(1.0)
        return


@app.post("/api/audio/<file_id>/edit")
def api_edit(file_id: str):
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    data = request.get_json(force=True) or {}
    op = data.get("op", "gain")
    name = data.get("name") or f"{op}-{entry['name']}"

    file_id_new = storage.new_id()
    dst = os.path.join(store.audio_dir, file_id_new + ".wav")
    _apply_edit(_abs_path(entry), dst, op, data.get("params", {}))
    new_entry = _register_derived(entry["id"], name, dst, {"op": op})
    return jsonify(new_entry)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def _run_analysis(kind: str, path: str) -> Dict:
    if kind == "chords":
        return chords.recognize(path)
    if kind == "spectrogram":
        return analysis.analyze_spectrogram(path)
    if kind == "spectral":
        return analysis.analyze_spectral(path)
    if kind == "pitch":
        return analysis.analyze_pitch(path)
    if kind == "beats":
        return analysis.analyze_beats(path)
    raise ValueError(f"unknown analysis {kind}")


@app.get("/api/analyze/<file_id>/<kind>")
def api_analyze(file_id: str, kind: str):
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    refresh = request.args.get("refresh") == "1"
    if not refresh:
        cached = store.get_analysis(file_id, kind)
        if cached:
            return jsonify(cached)
    try:
        data = _run_analysis(kind, _abs_path(entry))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    doc = store.save_analysis(file_id, kind, data, {"kind": kind})
    return jsonify(doc)


@app.post("/api/analyze/<file_id>/chords/annotate")
def api_annotate(file_id: str):
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    data = request.get_json(force=True) or {}
    annotations = data.get("annotations", [])
    doc = store.save_analysis(file_id, "chords_annotations", {"annotations": annotations})
    return jsonify(doc)


# --------------------------------------------------------------------------- #
# Source separation
# --------------------------------------------------------------------------- #

@app.post("/api/separation/<file_id>")
def api_separate(file_id: str):
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    data = request.get_json(force=True) or {}
    mode = data.get("mode", "harmonic_percussive")
    max_seconds = float(data.get("max_seconds", 60.0))

    tmp = tempfile.mkdtemp()
    try:
        if mode == "harmonic_percussive":
            p_h = os.path.join(tmp, "harmonic.wav")
            p_p = os.path.join(tmp, "percussive.wav")
            separation.hpss_separate(_abs_path(entry), p_h, p_p, max_seconds=max_seconds)
            h = _register_derived(entry["id"], f"harmonic-{entry['name']}", p_h, {"separation": mode})
            p = _register_derived(entry["id"], f"percussive-{entry['name']}", p_p, {"separation": mode})
            return jsonify({"harmonic": h, "percussive": p})
        if mode == "vocal_accompaniment":
            p_v = os.path.join(tmp, "vocal.wav")
            p_a = os.path.join(tmp, "accompaniment.wav")
            separation.mid_side_separate(_abs_path(entry), p_v, p_a)
            v = _register_derived(entry["id"], f"vocal-{entry['name']}", p_v, {"separation": mode})
            a = _register_derived(entry["id"], f"accompaniment-{entry['name']}", p_a, {"separation": mode})
            return jsonify({"vocal": v, "accompaniment": a})
        if mode == "bands":
            p_b = os.path.join(tmp, "bass.wav")
            p_m = os.path.join(tmp, "mid.wav")
            p_t = os.path.join(tmp, "treble.wav")
            separation.bands_separate(_abs_path(entry), p_b, p_m, p_t)
            b = _register_derived(entry["id"], f"bass-{entry['name']}", p_b, {"separation": mode})
            m = _register_derived(entry["id"], f"mid-{entry['name']}", p_m, {"separation": mode})
            t = _register_derived(entry["id"], f"treble-{entry['name']}", p_t, {"separation": mode})
            return jsonify({"bass": b, "mid": m, "treble": t})
        return jsonify(error="unknown mode"), 400
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Effects
# --------------------------------------------------------------------------- #

@app.get("/api/effects/catalog")
def api_effects_catalog():
    return jsonify(effects.effect_catalog())


@app.get("/api/effects/presets")
def api_effects_presets():
    return jsonify(effects.preset_list())


@app.post("/api/effects/preview")
def api_effects_preview():
    data = request.get_json(force=True) or {}
    file_id = data.get("file_id")
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    chain = data.get("chain", [])
    start = float(data.get("start", 0))
    duration = float(data.get("duration", 6))
    wav = effects.render_preview(_abs_path(entry), chain, start, duration)
    return jsonify({"wav": base64.b64encode(wav).decode(), "size": len(wav)})


@app.post("/api/effects/apply")
def api_effects_apply():
    data = request.get_json(force=True) or {}
    file_id = data.get("file_id")
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    chain = data.get("chain", [])
    name = data.get("name") or f"fx-{entry['name']}"

    file_id_new = storage.new_id()
    dst = os.path.join(store.audio_dir, file_id_new + ".wav")
    effects.apply_chain_to_file(_abs_path(entry), dst, chain)
    new_entry = _register_derived(entry["id"], name, dst, {"effects": chain})
    return jsonify(new_entry)


# --------------------------------------------------------------------------- #
# Projects & mixer
# --------------------------------------------------------------------------- #

@app.get("/api/projects")
def api_projects():
    return jsonify(store.list_projects())


@app.post("/api/projects")
def api_create_project():
    data = request.get_json(force=True) or {}
    return jsonify(store.create_project(data.get("name", "Untitled"), data.get("tracks", [])))


@app.get("/api/projects/<project_id>")
def api_project(project_id: str):
    p = store.get_project(project_id)
    if not p:
        return jsonify(error="project not found"), 404
    return jsonify(p)


@app.put("/api/projects/<project_id>")
def api_update_project(project_id: str):
    p = store.update_project(project_id, request.get_json(force=True) or {})
    if not p:
        return jsonify(error="project not found"), 404
    return jsonify(p)


@app.delete("/api/projects/<project_id>")
def api_delete_project(project_id: str):
    if not store.delete_project(project_id):
        return jsonify(error="project not found"), 404
    return jsonify(ok=True)


@app.post("/api/projects/<project_id>/mix")
def api_mix_project(project_id: str):
    p = store.get_project(project_id)
    if not p:
        return jsonify(error="project not found"), 404
    data = request.get_json(force=True) or {}
    tracks = []
    for t in p.get("tracks", []):
        e = _entry(t.get("file_id"))
        if e:
            tracks.append({"path": _abs_path(e), "gain": t.get("gain", 1.0),
                           "pan": t.get("pan", 0.0), "muted": t.get("muted", False)})
    file_id_new = storage.new_id()
    dst = os.path.join(store.audio_dir, file_id_new + ".wav")
    master = p.get("master", {})
    result = mixer.mixdown(tracks, dst, master_gain=master.get("gain", 1.0))
    name = data.get("name") or f"mixdown-{p['name']}.wav"
    entry = _register_derived(None, name, dst, {"project_id": project_id, **result})
    return jsonify(entry)


# --------------------------------------------------------------------------- #
# Export / format conversion
# --------------------------------------------------------------------------- #

@app.post("/api/export")
def api_export():
    data = request.get_json(force=True) or {}
    file_id = data.get("file_id")
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    fmt = data.get("format", "wav")
    dst_sr = data.get("sample_rate")
    sample_width = data.get("bit_depth", "pcm16")
    channels = data.get("channels")
    bitrate = data.get("bitrate", "192k")

    ext = fmt
    file_id_new = storage.new_id()
    dst = os.path.join(store.audio_dir, file_id_new + "." + ext)
    audio_io.convert(_abs_path(entry), dst, fmt, dst_sr=dst_sr,
                     sample_width=sample_width, channels=channels, bitrate=bitrate)

    # Read back what we can about the exported file.
    sr, ch, frames, duration, size = _probe(dst, fmt)
    entry = store.add_file({
        "id": file_id_new,
        "name": f"{entry['name'].rsplit('.', 1)[0]}.{ext}",
        "original_format": fmt,
        "path": f"audio/{file_id_new}.{ext}",
        "sr": sr,
        "channels": ch,
        "frames": frames,
        "duration": duration,
        "size_bytes": size,
        "derived_from": file_id,
        "exported": True,
    })
    return jsonify(entry)


def _probe(path: str, fmt: str):
    """Best-effort metadata probe for an exported file (ffmpeg for compressed)."""
    if fmt == "wav":
        try:
            with audio_io.WavReader(path) as r:
                return r.sr, r.channels, r.nframes, r.duration, os.path.getsize(path)
        except Exception:
            pass
    size = os.path.getsize(path)
    # Use ffprobe if available for duration.
    try:
        import subprocess
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


# --------------------------------------------------------------------------- #
# Batch processing
# --------------------------------------------------------------------------- #

BATCH_OPS = ("effects", "export", "edit")
EDIT_OPS = ("gain", "normalize", "fade", "silence", "reverse")
EXPORT_FORMATS = ("wav", "aiff", "raw", "mp3", "ogg", "flac", "m4a")


def _batch_runner(op: str, params: Dict):
    """Build the streaming renderer closure for one batch operation."""

    def run(item: Dict, out_path: str, on_progress, is_cancelled) -> None:
        entry = store.get_file(item["file_id"])
        if not entry:
            raise FileNotFoundError("源文件已从音频库删除")
        src = _abs_path(entry)
        if not os.path.isfile(src):
            raise FileNotFoundError("源音频文件丢失")
        if op == "effects":
            effects.apply_chain_to_file(src, out_path, params.get("chain", []),
                                        on_progress=on_progress,
                                        is_cancelled=is_cancelled)
        elif op == "edit":
            _apply_edit(src, out_path, params.get("edit_op", "gain"),
                        params.get("edit_params", {}),
                        on_progress=on_progress, is_cancelled=is_cancelled)
        elif op == "export":
            audio_io.convert(
                src, out_path, params.get("format", "wav"),
                dst_sr=params.get("sample_rate"),
                sample_width=params.get("bit_depth", "pcm16"),
                channels=params.get("channels"),
                bitrate=params.get("bitrate", "192k"),
                on_progress=on_progress, is_cancelled=is_cancelled)

    return run


def _batch_register(op: str, params: Dict):
    """Move a successfully rendered output into the library and index it."""

    def register(item: Dict, out_path: str) -> Dict:
        entry = store.get_file(item["file_id"])
        if not entry:
            raise FileNotFoundError("源文件已从音频库删除")
        src_name = entry.get("name", "audio")
        stem = src_name.rsplit(".", 1)[0] if "." in src_name else src_name

        if op == "export":
            fmt = params.get("format", "wav")
            file_id_new = storage.new_id()
            dst = os.path.join(store.audio_dir, file_id_new + "." + fmt)
            shutil.move(out_path, dst)
            sr, ch, frames, duration, size = _probe(dst, fmt)
            return store.add_file({
                "id": file_id_new,
                "name": f"{stem}.{fmt}",
                "original_format": fmt,
                "path": f"audio/{file_id_new}.{fmt}",
                "sr": sr, "channels": ch, "frames": frames,
                "duration": duration, "size_bytes": size,
                "derived_from": entry["id"],
                "exported": True,
                "batch": True,
            })

        extra = {"batch": True}
        if op == "effects":
            name = params.get("name") or f"fx-{src_name}"
            extra["effects"] = params.get("chain", [])
        else:
            edit_op = params.get("edit_op", "gain")
            name = params.get("name") or f"{edit_op}-{src_name}"
            extra["op"] = edit_op
        return _register_derived(entry["id"], name, out_path, extra)

    return register


@app.post("/api/batch")
def api_batch_create():
    data = request.get_json(force=True) or {}
    op = data.get("op")
    if op not in BATCH_OPS:
        return jsonify(error=f"未知操作类型（支持: {', '.join(BATCH_OPS)}）"), 400

    file_ids = data.get("file_ids") or []
    if not isinstance(file_ids, list) or not file_ids:
        return jsonify(error="请至少选择一个文件"), 400
    if len(file_ids) > 200:
        return jsonify(error="单次批量任务最多 200 个文件"), 400

    sources = []
    seen = set()
    for fid in file_ids:
        if fid in seen:
            continue
        seen.add(fid)
        e = store.get_file(fid)
        if e:
            sources.append(e)
    if not sources:
        return jsonify(error="所选文件均不存在（可能已被删除）"), 400

    params = data.get("params") or {}

    if op == "effects":
        chain = params.get("chain")
        if not isinstance(chain, list) or not chain:
            return jsonify(error="请选择一条效果链"), 400
        # Validate the chain up-front so the whole job doesn't fail per item.
        try:
            effects.EffectChain(chain, 44100)
        except Exception as e:
            return jsonify(error=f"效果链无效: {e}"), 400
        op_label = "应用效果链"

    elif op == "edit":
        edit_op = params.get("edit_op")
        if edit_op not in EDIT_OPS:
            return jsonify(error=f"不支持的编辑操作（支持: {', '.join(EDIT_OPS)}）"), 400
        ep = params.get("edit_params") or {}
        try:
            if edit_op in ("gain",):
                float(ep.get("gain_db", 0))
            if edit_op == "fade":
                if float(ep.get("fade_in", 0)) < 0 or float(ep.get("fade_out", 0)) < 0:
                    raise ValueError("淡入淡出时长不能为负")
            if edit_op == "silence":
                float(ep.get("start", 0))
                float(ep.get("end", 1))
        except (TypeError, ValueError) as e:
            return jsonify(error=f"编辑参数无效: {e}"), 400
        op_label = {"gain": "增益调整", "normalize": "响度归一化",
                    "fade": "淡入淡出", "silence": "区间静音",
                    "reverse": "倒放"}.get(edit_op, "编辑")

    else:  # export
        fmt = params.get("format", "wav")
        if fmt not in EXPORT_FORMATS:
            return jsonify(error=f"不支持的导出格式（支持: {', '.join(EXPORT_FORMATS)}）"), 400
        if fmt not in ("wav", "raw") and not audio_io.has_ffmpeg():
            return jsonify(error="该格式需要 ffmpeg，服务器当前未安装"), 400
        op_label = f"导出为 {fmt.upper()}"

    job = batches.create_job(op, params, sources,
                             _batch_runner(op, params),
                             _batch_register(op, params),
                             op_label=op_label)
    return jsonify(job), 201


@app.get("/api/batch")
def api_batch_list():
    return jsonify(batches.list_jobs())


@app.get("/api/batch/<job_id>")
def api_batch_get(job_id: str):
    job = batches.get_job(job_id)
    if not job:
        return jsonify(error="batch job not found"), 404
    return jsonify(job)


@app.post("/api/batch/<job_id>/cancel")
def api_batch_cancel(job_id: str):
    job = batches.cancel(job_id)
    if not job:
        return jsonify(error="batch job not found"), 404
    return jsonify(job)


@app.delete("/api/batch/<job_id>")
def api_batch_dismiss(job_id: str):
    if not batches.dismiss(job_id):
        return jsonify(error="任务不存在或仍在进行"), 404
    return jsonify(ok=True)


# --------------------------------------------------------------------------- #
# Versions / history
# --------------------------------------------------------------------------- #

@app.get("/api/projects/<project_id>/versions")
def api_versions(project_id: str):
    return jsonify(store.list_versions(project_id))


@app.post("/api/projects/<project_id>/revert")
def api_revert(project_id: str):
    data = request.get_json(force=True) or {}
    version = int(data.get("version", 1))
    p = store.revert_project(project_id, version)
    if not p:
        return jsonify(error="version not found"), 404
    return jsonify(p)


@app.get("/api/versions/<file_id>")
def api_file_versions(file_id: str):
    """Version chain for an audio file (its derived descendants)."""
    entry = _entry(file_id)
    if not entry:
        return jsonify(error="file not found"), 404
    # Walk the derived_from chain backwards to the root.
    chain = []
    cur = entry
    seen = set()
    while cur and cur["id"] not in seen:
        seen.add(cur["id"])
        chain.append(cur)
        parent = cur.get("derived_from")
        cur = _entry(parent) if parent else None
    return jsonify(chain)


# --------------------------------------------------------------------------- #
# Realtime
# --------------------------------------------------------------------------- #

@app.post("/api/realtime/analyze")
def api_realtime():
    data = request.get_json(force=True) or {}
    samples = data.get("samples", [])
    sr = float(data.get("sr", 44100))
    if not samples:
        return jsonify(error="empty buffer"), 400
    result = rt.process(samples, sr)
    return jsonify(result)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

@app.get("/api/stats")
def api_stats():
    return jsonify(store.stats())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f" * Audio MIR system running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
