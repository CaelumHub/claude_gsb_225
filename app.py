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
import math
import os
import shutil
import tempfile
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template, request, send_file

from backend import analysis, audio_io, batch, chords, effects, mixer, processing, realtime, separation, storage

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
batches = batch.BatchManager(store)
rt = realtime.new_analyzer()


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
    return processing.register_wav(store, source_id, name, wav_path, extra)


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
    processing.apply_edit(_abs_path(entry), dst, op, data.get("params", {}))
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
    processing.apply_effects(_abs_path(entry), dst, chain)
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

    if fmt in ("mp3", "ogg", "m4a", "aiff", "flac") and not audio_io.has_ffmpeg():
        return jsonify(error=f"导出 {fmt.upper()} 需要 ffmpeg，但服务器未安装"), 400

    # Render to a scratch file first; only a finished export enters the library.
    fd, tmp = tempfile.mkstemp(suffix="." + fmt)
    os.close(fd)
    os.unlink(tmp)
    try:
        processing.export_file(_abs_path(entry), tmp, fmt, dst_sr=dst_sr,
                               sample_width=sample_width, channels=channels,
                               bitrate=bitrate)
        name_base = entry["name"].rsplit(".", 1)[0]
        new_entry = processing.register_exported(store, file_id, name_base, tmp, fmt)
        return jsonify(new_entry)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


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
# Batch processing
# --------------------------------------------------------------------------- #

@app.post("/api/batch")
def api_batch_create():
    """Queue a batch job: one operation applied to many files.

    Body: {"operation": effects|export|edit, "params": {...}, "file_ids": [...]}
    Returns the created job (with per-item state) immediately; progress is polled
    via GET /api/batch/<id>.
    """
    data = request.get_json(force=True) or {}
    operation = data.get("operation")
    params = data.get("params") or {}
    file_ids: List[str] = data.get("file_ids") or []
    if not isinstance(file_ids, list) or not file_ids:
        return jsonify(error="请至少选择一个文件"), 400

    # Resolve every id up-front so an unknown id fails the whole submission
    # rather than being reported as 30 individual "file not found" rows.
    entries = []
    missing = []
    for fid in file_ids:
        e = _entry(fid)
        if e:
            entries.append(e)
        else:
            missing.append(fid)
    if missing:
        return jsonify(error=f"部分文件已不存在: {', '.join(missing[:5])}"), 404

    try:
        job = batches.create_job(operation, params, entries)
    except batch.BatchError as e:
        return jsonify(error=str(e)), 400
    return jsonify(job), 201


@app.get("/api/batch")
def api_batch_list():
    return jsonify(batches.list_jobs())


@app.get("/api/batch/active")
def api_batch_active():
    job = batches.active_job()
    if not job:
        return jsonify(None), 204
    return jsonify(job)


@app.get("/api/batch/<job_id>")
def api_batch_get(job_id: str):
    job = batches.get_job(job_id)
    if not job:
        return jsonify(error="batch job not found"), 404
    return jsonify(job)


@app.post("/api/batch/<job_id>/cancel")
def api_batch_cancel(job_id: str):
    job = batches.cancel_job(job_id)
    if not job:
        return jsonify(error="batch job not found"), 404
    return jsonify(job)


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
