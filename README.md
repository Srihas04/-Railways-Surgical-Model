# Surgical Guardian v4 — Railway Deployment Guide

## What changed from the original?

The original `surgical_guardian_v4.py` used `cv2.imshow()` (a desktop GUI window),
keyboard controls, and audio beeps — none of which work on a cloud server.

This web edition replaces all of that with:
- **Flask web server** serving a live MJPEG video stream
- **REST API** for stats, alerts, and controls (`/stats`, `/control`, `/log.csv`)
- **Browser dashboard** with real-time alert banner, stat cards, and log table
- All inference/detection logic is **identical** to v4

---

## Files in this folder

```
surgical_guardian_web.py   ← main app (replaces surgical_guardian_v4.py)
best.pt                    ← your YOLO model (copy here)
requirements.txt
Procfile
railway.toml
.gitignore
README.md
```

---

## Step-by-Step Railway Deployment

### Step 1 — Prepare your files locally

Make sure all these files are in one folder:
```
surgical_guardian_web.py
best.pt                   ← copy your model here
requirements.txt
Procfile
railway.toml
.gitignore
```

> ⚠️ If `best.pt` is larger than 100 MB, use Git LFS:
> ```
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes
> ```

---

### Step 2 — Create a GitHub repository

1. Go to https://github.com and create a **new repository** (any name, e.g. `surgical-guardian`)
2. On your machine, open a terminal in the folder with your files:

```bash
git init
git add .
git commit -m "Initial commit - Surgical Guardian v4 web edition"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

### Step 3 — Create a Railway project

1. Go to https://railway.app and sign in (GitHub login is easiest)
2. Click **"New Project"**
3. Choose **"Deploy from GitHub repo"**
4. Select your repo (`surgical-guardian`)
5. Railway auto-detects Python via Nixpacks — click **Deploy**

---

### Step 4 — Set Environment Variables

In Railway dashboard → your service → **Variables** tab, add:

| Variable | Value | Description |
|---|---|---|
| `VIDEO_SOURCE` | `0` or a URL | `0` = webcam, or a video URL / RTSP stream |
| `MODEL_PATH` | `best.pt` | Path to your model (default is fine if in root) |
| `CONF_THRESH` | `0.30` | Detection confidence threshold |
| `FRAME_WIDTH` | `640` | Frame width |
| `FRAME_HEIGHT` | `480` | Frame height |

> ℹ️ `PORT` is set automatically by Railway — do not override it.

---

### Step 5 — Generate a Public URL

1. In Railway dashboard → your service → **Settings** tab
2. Under **Networking**, click **"Generate Domain"**
3. You'll get a URL like `https://surgical-guardian-production.up.railway.app`

Open that URL in your browser — you'll see the live dashboard!

---

### Step 6 — Scale / Performance tips

- Railway's free tier gives **500 hours/month** — enough for demos
- For continuous 24/7 use, upgrade to the **Hobby plan ($5/month)**
- For GPU inference (faster YOLO), use Railway's **GPU instance** (select in service settings)
- Use **1 worker** in Procfile (already set) — inference runs in a background thread

---

## Using a video file instead of a live camera

Set `VIDEO_SOURCE` to the path or URL of your video file:
- Upload the video to the repo, set `VIDEO_SOURCE=my_surgery_video.mp4`
- Or use a public URL: `VIDEO_SOURCE=https://example.com/laparoscopy.mp4`

---

## Web Dashboard Features

| Control | What it does |
|---|---|
| ⏸ PAUSE / ▶ RESUME | Pause/resume inference |
| ↺ RESET | Reset all session stats |
| CONF slider | Live-adjust detection confidence |
| ⬇ EXPORT LOG | Download alert_log.csv |

**API endpoints:**
- `GET /stats` → JSON with all stats + last 50 alert rows
- `POST /control` → `{"cmd": "pause"}` / `{"cmd": "reset"}` / `{"cmd": "conf", "value": 0.4}`
- `GET /log.csv` → Download full session alert log
- `GET /video_feed` → MJPEG stream (embed in any `<img src="">`)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Model not found" | Make sure `best.pt` is committed to git, or set `MODEL_PATH` env var |
| Black screen / "Waiting for frames" | Check `VIDEO_SOURCE` — webcam `0` won't work on Railway (no hardware). Use a video file or RTSP URL |
| Build fails | Check Railway build logs; usually a missing dependency |
| Slow inference | Use a smaller `imgsz` (e.g. 320) or upgrade to GPU instance |

---

## Webcam note

Railway servers have **no physical webcam**. For live surgical use:
- Stream from a phone/camera using DroidCam, OBS, or any RTSP server
- Set `VIDEO_SOURCE` to the RTSP/HTTP stream URL
- Example: `VIDEO_SOURCE=rtsp://192.168.1.100:8554/stream`
