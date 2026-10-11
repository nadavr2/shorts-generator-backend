"""
Shorts Generator API — Production SaaS Backend
Fully dynamic, multi-tenant, cloud-ready FastAPI server.

Environment variables required:
  API_KEY          — shared secret between this server and n8n
  OLLAMA_URL       — e.g. http://localhost:11434  (or a cloud endpoint)
  WORKSPACE_ROOT   — e.g. /tmp/jobs  (one sub-folder per job UUID)
  OLLAMA_MODEL     — default: llama3.2:latest
"""

import asyncio
import json
import logging
import os
import random
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
API_KEY        = os.getenv("API_KEY", "change-me-in-production")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "/tmp/jobs"))
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Shorts Generator API",
    version="2.0.0",
    description="Multi-tenant YouTube Shorts automation backend",
)

# ── In-memory job registry (replace with Redis for horizontal scaling) ────────
# Structure: { job_id: { status, created_at, result?, error? } }
JOBS: dict[str, dict] = {}

# ── Content filters ───────────────────────────────────────────────────────────
BANNED_TITLE_WORDS = [
    "parody", "fan", "reaction", "amv", "edit", "compilation",
    "review", "ytp", "poop", "crack", "trailer", "opening", "ending",
    "clip", "moments", "highlight", "best of", "funniest", "mashup",
    "ranked", "top 10", "top10", "recap", "explained", "tier list",
    "every time", "every episode", "shorts", "tiktok",
    "timeline", "evolution of", "everything about", "guide to",
    "ranking", "tribute", "analysis", "theory", "easter egg",
]

MIN_DURATION = int(os.getenv("MIN_DURATION_S", "480"))   # 8 min
MAX_DURATION = int(os.getenv("MAX_DURATION_S", "1800"))  # 30 min


# ── Auth dependency ───────────────────────────────────────────────────────────
async def require_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


# ── Pydantic models ───────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    topic: str                      # e.g. "The Amazing World of Gumball"
    user_id: str                    # caller's user ID for workspace isolation
    num_clips: int = 2              # how many Shorts to produce
    clip_duration_max: int = 58     # seconds per clip
    banned_keywords: list[str] = [] # extra banned search keywords
    language: str = "en"           # subtitle / prompt language


class JobStatus(BaseModel):
    job_id: str
    status: str   # pending | processing | done | failed
    created_at: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ── Subprocess helper (list-based, no shell=True → cross-platform safe) ───────
def run(args: list[str], timeout: int = 300) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            args, shell=False, capture_output=True,
            text=True, encoding="utf-8", errors="ignore", timeout=timeout,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Timed out after {timeout}s"
    except FileNotFoundError as e:
        return 1, "", f"Executable not found: {e}"
    except Exception as e:
        return 1, "", str(e)


def workspace(job_id: str) -> Path:
    p = WORKSPACE_ROOT / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── USED-VIDEOS store per user (simple JSON file, swap for DB in production) ──
def used_videos_path(user_id: str) -> Path:
    p = WORKSPACE_ROOT / "users" / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p / "used_videos.json"


def load_used(user_id: str) -> list[str]:
    f = used_videos_path(user_id)
    return json.loads(f.read_text()) if f.exists() else []


def save_used(user_id: str, url: str):
    used = load_used(user_id)
    if url not in used:
        used.append(url)
    used_videos_path(user_id).write_text(json.dumps(used[-200:]))


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ═════════════════════════════════════════════════════════════════════════════

def step_find_video(topic: str, user_id: str, extra_banned: list[str]) -> dict:
    """Search YouTube for a suitable source video on the given topic."""
    used = load_used(user_id)
    all_banned = list(set(BANNED_TITLE_WORDS + extra_banned))

    # Build several query variants for a larger candidate pool
    queries = [
        f"{topic} full episode official",
        f"{topic} HD",
        f"{topic} full",
    ]
    negative = "-parody -fan -reaction -amv -edit -clip"

    all_entries: list[dict] = []
    seen_ids: set[str] = set()

    for q in queries:
        query = f"{q} {negative}"
        log.info(f"[find_video] Searching: {query!r}")
        rc, stdout, stderr = run(
            ["yt-dlp", f"ytsearch40:{query}", "--flat-playlist", "--dump-single-json"],
            timeout=60,
        )
        if rc != 0:
            log.warning(f"[find_video] Query failed: {stderr[:150]}")
            continue
        try:
            for item in json.loads(stdout).get("entries", []):
                vid_id = item.get("id")
                if vid_id and vid_id not in seen_ids:
                    seen_ids.add(vid_id)
                    all_entries.append(item)
        except json.JSONDecodeError:
            log.warning("[find_video] Bad JSON, skipping query")

    if not all_entries:
        raise RuntimeError("All search queries failed — check yt-dlp / network")

    random.shuffle(all_entries)
    log.info(f"[find_video] Pool: {len(all_entries)} unique candidates")

    stats = {"used": 0, "banned": 0, "short": 0, "long": 0}
    for item in all_entries:
        vid_id   = item.get("id")
        url      = f"https://www.youtube.com/watch?v={vid_id}"
        title    = (item.get("title") or "").lower()
        duration = item.get("duration") or 0

        if url in used:              stats["used"]  += 1; continue
        if any(b in title for b in all_banned): stats["banned"] += 1; continue
        if duration < MIN_DURATION:  stats["short"] += 1; continue
        if duration > MAX_DURATION:  stats["long"]  += 1; continue

        save_used(user_id, url)
        log.info(f"[find_video] Selected: {item.get('title')!r} ({duration}s)")
        return {"videoUrl": url, "videoTitle": item.get("title"), "duration": duration}

    raise RuntimeError(
        f"No suitable video among {len(all_entries)} candidates. "
        f"Rejected: {stats}. "
        f"Duration window: {MIN_DURATION}-{MAX_DURATION}s. "
        f"Clear used_videos.json for user '{user_id}' if 'used' count is high."
    )


def step_download(url: str, job_id: str) -> Path:
    """Download source video (max 1080p) to job workspace."""
    ws  = workspace(job_id)
    out = ws / "original.mp4"
    if out.exists():
        out.unlink()

    log.info(f"[download] {url}")
    fmt = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]"

    # Try with browser cookies first, fall back to anonymous
    for extra in [["--cookies-from-browser", "chrome"], []]:
        rc, _, stderr = run(
            ["yt-dlp", *extra, "-f", fmt, "--merge-output-format", "mp4",
             "-o", str(out), url],
            timeout=600,
        )
        if rc == 0 and out.exists() and out.stat().st_size > 0:
            log.info(f"[download] OK — {out.stat().st_size / 1e6:.1f} MB")
            return out
        log.warning(f"[download] Attempt failed: {stderr[:100]}")

    raise RuntimeError("Download failed after two attempts")


def step_transcribe(job_id: str) -> dict:
    """Transcribe with Whisper; returns segments + full_text."""
    ws       = workspace(job_id)
    mp4      = ws / "original.mp4"
    stem     = mp4.stem                    # "original"
    json_out = ws / f"{stem}.json"

    if not mp4.exists() or mp4.stat().st_size == 0:
        raise RuntimeError("Video file missing before transcription")

    if json_out.exists():
        json_out.unlink()

    log.info("[transcribe] Starting Whisper…")
    t0 = datetime.now()
    rc, stdout, stderr = run(
        ["whisper", str(mp4),
         "--model", "base",
         "--word_timestamps", "True",
         "--condition_on_previous_text", "False",
         "--output_format", "json",
         "--output_dir", str(ws)],
        timeout=2400,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"[transcribe] Whisper exited (rc={rc}) in {elapsed:.0f}s")

    if stdout.strip():
        log.debug(f"[transcribe] stdout tail: {stdout.strip()[-300:]}")
    if stderr.strip():
        log.debug(f"[transcribe] stderr tail: {stderr.strip()[-300:]}")

    if rc != 0:
        raise RuntimeError(f"Whisper failed: {stderr[:300]}")

    # Whisper may name the file after the stem — search for any .json in ws
    json_file = None
    if json_out.exists():
        json_file = json_out
    else:
        candidates = list(ws.glob("*.json"))
        if candidates:
            json_file = candidates[0]
            log.info(f"[transcribe] Using fallback JSON: {json_file.name}")

    if json_file is None:
        files = [f.name for f in ws.iterdir()]
        raise RuntimeError(
            f"Whisper produced no JSON. Workspace contains: {files}. "
            "Check stdout/stderr above."
        )

    transcript = json.loads(json_file.read_text(encoding="utf-8"))
    segments   = transcript.get("segments", [])
    full_text  = " | ".join(
        f"[{s['start']:.0f}s] {s['text']}" for s in segments
    ).replace('"', "'")
    duration   = segments[-1]["end"] if segments else 0

    log.info(f"[transcribe] {len(segments)} segments, {duration:.0f}s total")
    return {"segments": segments, "fullText": full_text, "duration": duration}


async def step_ai_pick(full_text: str, duration: float,
                       topic: str, num_clips: int, clip_max: int) -> list[dict]:
    """Use Ollama to select the best clip timestamps for the given topic."""

    # Auto-discover exact model name from Ollama
    model = OLLAMA_MODEL
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.get(f"{OLLAMA_URL}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                base   = OLLAMA_MODEL.split(":")[0]
                match  = next((m for m in models if m.startswith(base)), None)
                if match:
                    model = match
                    log.info(f"[ai_pick] Using model: {model}")
                elif models:
                    model = models[0]
                    log.warning(f"[ai_pick] {OLLAMA_MODEL} not found, using {model}")
    except Exception as e:
        log.warning(f"[ai_pick] Could not query /api/tags: {e}")

    prompt = f"""You are an expert YouTube Shorts editor.

TOPIC: "{topic}"

TASK: Find exactly {num_clips} consecutive clips from the transcript below.
Each clip must be {clip_max - 5}–{clip_max} seconds long.
Clips must form a COMPLETE narrative arc:
  - Clip 1: Setup / Conflict
  - Clip 2 (if requested): Resolution / Payoff
Rules:
  - Only character dialogue — no narrator/intro/outro speech
  - Clean start and end — no audio bleed from adjacent scenes
  - Return ONLY valid JSON, no other text

Transcript ({duration:.0f}s total):
{full_text[:5000]}

JSON schema (return exactly this shape, with {num_clips} clips):
{{
  "clips": [
    {{
      "start": 10.0,
      "end": 58.0,
      "title": "Catchy title for this clip",
      "description": "One sentence description",
      "hashtags": "#Topic #Shorts #Funny #Cartoon"
    }}
  ]
}}"""

    async with httpx.AsyncClient(timeout=300) as c:
        resp = await c.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "stream": False,
                  "messages": [{"role": "user", "content": prompt}]},
        )

    if resp.status_code != 200:
        try:
            err = resp.json().get("error", resp.text[:200])
        except Exception:
            err = resp.text[:200]
        installed = []
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r2 = await c.get(f"{OLLAMA_URL}/api/tags")
                installed = [m["name"] for m in r2.json().get("models", [])]
        except Exception:
            pass
        raise RuntimeError(
            f"Ollama {resp.status_code} for model '{model}': {err}. "
            f"Installed: {installed}"
        )

    content = resp.json()["message"]["content"]
    match   = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise RuntimeError(f"No JSON in AI response: {content[:200]}")

    clips = json.loads(match.group()).get("clips", [])
    if len(clips) < num_clips:
        raise RuntimeError(f"AI returned {len(clips)} clips, expected {num_clips}")

    log.info(f"[ai_pick] Selected {len(clips)} clips")
    return clips[:num_clips]


def make_ass_subtitles(segments: list, start: float, end: float,
                       output: Path, delay: float = 0.2):
    """Generate ASS subtitle file — cyan text, thick white outline."""
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial Rounded MT Bold,88,&H00FFD475,&H000000FF,"
        "&H00FFFFFF,&H00000000,-1,0,0,0,100,100,0,0,1,10,0,2,60,60,650,1\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    def fmt(t: float) -> str:
        t = max(0.0, t)
        h, rem = divmod(t, 3600)
        m, s   = divmod(rem, 60)
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"

    all_words: list[dict] = []
    for seg in segments:
        ss, se = seg.get("start", 0), seg.get("end", 0)
        if seg.get("words"):
            for w in seg["words"]:
                txt = (w.get("word") or "").strip().upper()
                if txt and len(txt) > 1:
                    all_words.append({"text": txt,
                                      "start": w.get("start", ss),
                                      "end":   w.get("end",   se)})
        else:
            words = (seg.get("text") or "").strip().split()
            if not words:
                continue
            spw = (se - ss) / max(len(words), 1)
            for i, w in enumerate(words):
                if w.strip():
                    all_words.append({"text": w.upper(),
                                      "start": ss + i * spw,
                                      "end":   ss + (i + 1) * spw})

    events: list[str] = []
    i = 0
    clip_dur = end - start
    while i < len(all_words):
        size  = min(2, len(all_words) - i)
        chunk = all_words[i:i + size]
        cs, ce = chunk[0]["start"], chunk[-1]["end"]
        text   = " ".join(w["text"] for w in chunk)
        if ce > start and cs < end:
            a_s = max(0.0, min(max(cs, start) - start + delay, clip_dur - 0.1))
            a_e = max(a_s + 0.1, min(min(ce, end) - start + delay, clip_dur))
            events.append(
                f"Dialogue: 0,{fmt(a_s)},{fmt(a_e)},Default,,0,0,0,,{text}"
            )
        i += size

    output.write_text(header + "\n".join(events), encoding="utf-8")


def step_cut(job_id: str, idx: int, start: float, end: float,
             title: str, description: str, hashtags: str) -> dict:
    """Cut one clip with blur background and professional subtitles."""
    ws      = workspace(job_id)
    mp4_in  = ws / "original.mp4"
    ass     = ws / f"subs_{idx}.ass"
    mp4_out = ws / f"short_{idx}.mp4"

    transcript = json.loads((ws / "original.json").read_text(encoding="utf-8"))
    make_ass_subtitles(transcript.get("segments", []), start, end, ass)

    # ASS path safe for FFmpeg (Windows & Linux)
    drive, rest = os.path.splitdrive(str(ass))
    ass_ffmpeg  = (drive.replace(":", "\\:") + rest).replace("\\", "/")

    fg = (
        f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,boxblur=25:5[bg];"
        f"[0:v]scale=-2:1440,crop=1080:1440[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vid];"
        f"[vid]ass='{ass_ffmpeg}'"
    )

    rc, _, stderr = run(
        ["ffmpeg",
         "-ss", str(start), "-t", str(end - start),
         "-i", str(mp4_in),
         "-filter_complex", fg,
         "-c:v", "libx264", "-preset", "fast", "-crf", "20",
         "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
         "-y", str(mp4_out), "-loglevel", "error"],
        timeout=900,
    )

    if rc != 0 or not mp4_out.exists():
        raise RuntimeError(f"FFmpeg failed for clip {idx}: {stderr[:300]}")

    log.info(f"[cut] Clip {idx} done — {mp4_out.stat().st_size / 1e6:.1f} MB")
    return {
        "clipIndex":   idx,
        "path":        str(mp4_out),
        "title":       title,
        "description": description,
        "hashtags":    hashtags,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Background pipeline
# ═════════════════════════════════════════════════════════════════════════════

async def run_pipeline(job_id: str, req: GenerateRequest):
    JOBS[job_id]["status"] = "processing"
    ws = workspace(job_id)
    try:
        log.info(f"[job:{job_id}] START — topic={req.topic!r}")

        # 1. Find
        video  = step_find_video(req.topic, req.user_id, req.banned_keywords)
        # 2. Download
        step_download(video["videoUrl"], job_id)
        # 3. Transcribe
        tx     = step_transcribe(job_id)
        # Save transcript for step_cut to reuse
        (ws / "original.json").write_text(
            json.dumps({"segments": tx["segments"]}), encoding="utf-8"
        )
        # 4. AI pick
        clips  = await step_ai_pick(
            tx["fullText"], tx["duration"],
            req.topic, req.num_clips, req.clip_duration_max,
        )
        # 5. Cut all clips
        results = []
        for i, clip in enumerate(clips, 1):
            r = step_cut(
                job_id, i,
                float(clip["start"]), float(clip["end"]),
                clip.get("title", f"{req.topic} Moment {i}"),
                clip.get("description", clip.get("title", "")),
                clip.get("hashtags", f"#{req.topic.replace(' ', '')} #Shorts"),
            )
            results.append(r)

        JOBS[job_id].update({"status": "done", "result": {
            "clips":      results,
            "videoTitle": video["videoTitle"],
            "topic":      req.topic,
        }})
        log.info(f"[job:{job_id}] DONE — {len(results)} clips")

    except Exception as e:
        log.exception(f"[job:{job_id}] FAILED")
        JOBS[job_id].update({"status": "failed", "error": str(e)})
    finally:
        # Cleanup heavy source files; keep the finished MP4s
        for fname in ["original.mp4", "original.json", "original.wav"]:
            f = ws / fname
            if f.exists():
                f.unlink(missing_ok=True)
        for ass in ws.glob("subs_*.ass"):
            ass.unlink(missing_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# REST endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/jobs", dependencies=[Depends(require_api_key)])
async def create_job(req: GenerateRequest, bg: BackgroundTasks) -> dict:
    """
    Kick off a generation job in the background.
    Returns immediately with a job_id — poll /jobs/{job_id} for status.
    """
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status":     "pending",
        "created_at": datetime.utcnow().isoformat(),
    }
    bg.add_task(run_pipeline, job_id, req)
    log.info(f"[jobs] Created job {job_id} for topic={req.topic!r}")
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job(job_id: str) -> JobStatus:
    """Poll job status and retrieve clip metadata when done."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    j = JOBS[job_id]
    return JobStatus(job_id=job_id, **j)


@app.get("/jobs/{job_id}/clips/{clip_index}",
         dependencies=[Depends(require_api_key)])
async def download_clip(job_id: str, clip_index: int):
    """Stream the final MP4 file for a completed clip."""
    ws  = workspace(job_id)
    mp4 = ws / f"short_{clip_index}.mp4"
    if not mp4.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(mp4, media_type="video/mp4",
                        filename=f"short_{clip_index}.mp4")


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def delete_job(job_id: str) -> dict:
    """Delete job files and registry entry."""
    ws = workspace(job_id)
    if ws.exists():
        shutil.rmtree(ws)
    JOBS.pop(job_id, None)
    return {"deleted": job_id}


@app.delete("/users/{user_id}/history", dependencies=[Depends(require_api_key)])
async def clear_user_history(user_id: str) -> dict:
    """Reset a user's used-videos list so they get fresh content."""
    f = used_videos_path(user_id)
    if f.exists():
        f.unlink()
    return {"cleared": user_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
