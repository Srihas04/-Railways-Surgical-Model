"""
╔══════════════════════════════════════════════════════════════════════╗
║        SURGICAL GUARDIAN v4 — Railway Web Edition                   ║
║   Real-time laparoscopic safety monitoring — Flask MJPEG stream     ║
╚══════════════════════════════════════════════════════════════════════╝

Rewritten from desktop (cv2.imshow) to a Railway-deployable Flask app.
- MJPEG video stream served at /video_feed
- REST API for stats, alerts, control commands
- Web dashboard UI at /
"""

import csv
import io
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "bipolar", "clipper", "grasper", "hook",
    "irrigator", "scissors", "specimen_bag",
    "liver", "gallbladder", "abdominal_wall",
    "fat", "GI_tract", "connective_tissue",
    "liver_ligament",
    "CYSTIC ARTERY", "CYSTIC DUCT",
]

TOOLS   = set(range(0, 7))
ORGANS  = set(range(7, 14))
VESSELS = {14, 15}

TOOL_DANGER_WEIGHT = {
    0: 0.8, 1: 0.9, 2: 0.4, 3: 1.0,
    4: 0.2, 5: 0.85, 6: 0.1,
}

CAUTION_DIST  = 150
WARNING_DIST  = 100
CRITICAL_DIST = 60

C_TOOL     = (0, 220, 255)
C_ORGAN    = (0, 140, 255)
C_VESSEL   = (0,   0, 255)
C_CAUTION  = (0, 200, 255)
C_WARNING  = (0, 100, 255)
C_CRITICAL = (0,   0, 255)
C_OK       = (0, 200,  80)
C_HUD      = (0, 255, 180)
C_APPROACH = (0,  50, 255)

ALERT_TIERS = [
    (CRITICAL_DIST, "!! CRITICAL — STOP !!",    C_CRITICAL, 3, 1100),
    (WARNING_DIST,  "!  WARNING — Too Close",    C_WARNING,  2,  800),
    (CAUTION_DIST,  "   CAUTION — Approaching",  C_CAUTION,  1,  600),
]

TRAIL_LEN        = 25
IOU_MATCH_THRESH = 0.25
FREEZE_FRAMES    = 7

# ══════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════
app = Flask(__name__)

# ── Global shared state ───────────────────────────────────────────────
state = {
    "conf_thresh": float(os.environ.get("CONF_THRESH", "0.30")),
    "paused":      False,
    "latest_frame": None,
    "frame_lock":   threading.Lock(),
    "stats": {
        "total": 0, "critical": 0, "warning": 0, "caution": 0,
        "frames": 0, "min_dist": 9999.0, "closest_ever": 9999.0,
        "elapsed": "00:00", "approaching": False, "recording": False,
        "fps": 0.0, "alert_level": 0,
    },
    "csv_rows": [],          # in-memory CSV log (last 500 rows)
    "session_t0": time.time(),
}

# ══════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ══════════════════════════════════════════════════════════════════════
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def iou(a, b):
    xi1 = max(a["x1"], b["x1"]); yi1 = max(a["y1"], b["y1"])
    xi2 = min(a["x2"], b["x2"]); yi2 = min(a["y2"], b["y2"])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    if inter == 0:
        return 0.0
    ua = (a["x2"]-a["x1"])*(a["y2"]-a["y1"])
    ub = (b["x2"]-b["x1"])*(b["y2"]-b["y1"])
    return inter / (ua + ub - inter)

def smooth_detections(prev, curr, alpha=0.55):
    if not prev:
        return curr
    out = []
    for c in curr:
        best_iou, best_p = 0.0, None
        for p in prev:
            if p["cls"] == c["cls"]:
                sc = iou(c, p)
                if sc > best_iou:
                    best_iou, best_p = sc, p
        if best_p and best_iou > IOU_MATCH_THRESH:
            c = c.copy()
            for k in ("x1","y1","x2","y2","cx","cy"):
                c[k] = int(alpha * c[k] + (1-alpha) * best_p[k])
        out.append(c)
    return out

def compute_velocity(trail):
    pts = list(trail)
    if len(pts) < 2:
        return 0.0, 0.0, 0.0
    vx = pts[-1][0] - pts[-2][0]
    vy = pts[-1][1] - pts[-2][1]
    return vx, vy, math.hypot(vx, vy)

def approach_rate(trail, vessel_cx, vessel_cy):
    pts = list(trail)
    if len(pts) < 3:
        return 0.0
    d_now  = math.hypot(pts[-1][0]-vessel_cx, pts[-1][1]-vessel_cy)
    d_prev = math.hypot(pts[-3][0]-vessel_cx, pts[-3][1]-vessel_cy)
    return (d_now - d_prev) / 2.0

def is_inside_bbox(cx, cy, det):
    return det["x1"] <= cx <= det["x2"] and det["y1"] <= cy <= det["y2"]

def draw_label(frame, text, x1, y1, color, font_scale=0.48, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y_top = max(y1-th-8, 0)
    cv2.rectangle(frame, (x1, y_top), (x1+tw+6, y1), color, -1)
    cv2.putText(frame, text, (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)

def draw_trails(frame, tool_trails):
    for pts_dq in tool_trails.values():
        pts = list(pts_dq)
        for i in range(1, len(pts)):
            alpha = int(220 * i / len(pts))
            thick = max(1, i // 6)
            cv2.line(frame, pts[i-1], pts[i], (0, alpha, 255), thick, cv2.LINE_AA)

def draw_velocity_arrow(frame, cx, cy, vx, vy, speed):
    if speed < 1.5:
        return
    scale = min(speed * 3, 40)
    ex = int(cx + vx / max(speed, 1e-6) * scale)
    ey = int(cy + vy / max(speed, 1e-6) * scale)
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), C_APPROACH, 2, tipLength=0.4, line_type=cv2.LINE_AA)

def draw_hud(frame, tools, vessels, organs, fps, stats, conf_thresh, paused, W, H):
    panel_w = 195
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (panel_w, H), (12,12,12), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    def txt(text, x, y, color=C_HUD, scale=0.48, bold=False):
        cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    2 if bold else 1, cv2.LINE_AA)

    def row(label, val, y, val_color=C_HUD):
        txt(label, 10, y, (140,140,140))
        txt(val, panel_w-10-len(val)*8, y, val_color)

    txt("SURGICAL", 10, 26, C_HUD, 0.60, True)
    txt("GUARDIAN v4", 10, 46, C_HUD, 0.50)
    cv2.line(frame, (10,54), (panel_w-10,54), C_HUD, 1)

    status_color = (0,80,255) if paused else C_OK
    txt("PAUSED" if paused else "LIVE", panel_w//2-20, 72, status_color, 0.50, True)

    row("FPS",     f"{int(fps):3d}",       90,  C_HUD)
    row("CONF",    f"{conf_thresh:.2f}",  108,  (180,180,180))
    row("TOOLS",   str(len(tools)),       130,  C_TOOL)
    row("ORGANS",  str(len(organs)),      148,  C_ORGAN)
    row("VESSELS", str(len(vessels)),     166,  C_VESSEL)
    cv2.line(frame, (10,178), (panel_w-10,178), (50,50,50), 1)
    row("ALERTS",   str(stats["total"]),    196,  C_WARNING)
    row("CRITICAL", str(stats["critical"]), 214,  C_CRITICAL)
    row("WARNING",  str(stats["warning"]),  232,  C_WARNING)
    row("CAUTION",  str(stats["caution"]),  250,  C_CAUTION)
    cv2.line(frame, (10,262), (panel_w-10,262), (50,50,50), 1)
    row("SESSION", stats["elapsed"],       280,  C_HUD)
    row("FRAMES",  str(stats["frames"]),   298,  (140,140,140))
    cv2.line(frame, (10,312), (panel_w-10,312), (50,50,50), 1)
    txt("MIN DIST", 10, 332, (120,120,120), 0.42)
    min_d = stats["min_dist"]
    d_color = (C_CRITICAL if min_d < CRITICAL_DIST else
               C_WARNING  if min_d < WARNING_DIST  else
               C_CAUTION  if min_d < CAUTION_DIST  else C_OK)
    dist_str = f"{int(min_d)}px" if min_d < 9999 else "---"
    txt(dist_str, 10, 360, d_color, 0.72, True)
    if stats.get("approaching"):
        txt(">> APPROACHING", 6, 390, C_CRITICAL, 0.40, True)

def draw_organ_overlap_warning(frame, tools, organs, W):
    for t in tools:
        for o in organs:
            if is_inside_bbox(t["cx"], t["cy"], o):
                msg = f"TOOL INSIDE {CLASS_NAMES[o['cls']].upper()}"
                (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                mx = W//2 - mw//2
                cv2.rectangle(frame, (mx-8,108), (mx+mw+8,138), (0,0,0), -1)
                cv2.putText(frame, msg, (mx,130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            C_WARNING, 2, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════
# VIDEO CAPTURE (threaded, reconnects on drop)
# ══════════════════════════════════════════════════════════════════════
class VideoCapture:
    def __init__(self, source, w=640, h=480):
        self.source  = source
        self.w, self.h = w, h
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = False
        self._eof    = False
        src = str(source)
        self.is_file = (not src.isdigit() and
                        not src.startswith("http") and
                        not src.startswith("rtsp") and
                        os.path.isfile(src))
        self._open()

    def _open(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        self.cap = cv2.VideoCapture(src)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

    def start(self):
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    self._eof = True
                    self.running = False
                    break
                time.sleep(1.0)
                self._open()
                continue
            with self.lock:
                self.frame = frame
            if self.is_file:
                time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    @property
    def eof(self):
        return self._eof

    def stop(self):
        self.running = False
        self.cap.release()

# ══════════════════════════════════════════════════════════════════════
# INFERENCE THREAD
# ══════════════════════════════════════════════════════════════════════
def inference_loop(model, cap, W, H):
    tool_trails = {}
    prev_dets   = []
    last_dets   = []
    freeze_left = 0
    prev_time   = time.time()

    while True:
        if cap.eof:
            print("Video source ended.")
            break

        if state["paused"]:
            time.sleep(0.03)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))

        frame = enhance_frame(frame)
        stats = state["stats"]
        stats["frames"] += 1
        conf_thresh = state["conf_thresh"]

        # Inference
        results    = model(frame, conf=conf_thresh, imgsz=416, verbose=False, stream=True)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                detections.append({
                    "cls": cls_id, "conf": conf,
                    "group": ("tool" if cls_id in TOOLS else
                              "vessel" if cls_id in VESSELS else "organ"),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1+x2)//2, "cy": (y1+y2)//2,
                })

        detections = smooth_detections(prev_dets, detections, alpha=0.55)
        prev_dets  = detections
        draw_dets  = detections if detections else last_dets
        last_dets  = detections if detections else last_dets

        tools   = [d for d in draw_dets if d["group"] == "tool"]
        organs  = [d for d in draw_dets if d["group"] == "organ"]
        vessels = [d for d in draw_dets if d["group"] == "vessel"]

        # Motion trails
        seen = set()
        for t in tools:
            tid = t["cls"]
            seen.add(tid)
            tool_trails.setdefault(tid, deque(maxlen=TRAIL_LEN))
            tool_trails[tid].append((t["cx"], t["cy"]))
        for tid in list(tool_trails):
            if tid not in seen:
                del tool_trails[tid]

        if freeze_left > 0:
            freeze_left -= 1
            # push last frame again
            with state["frame_lock"]:
                pass  # keep previous frame displayed
            continue

        # Draw overlays
        draw_trails(frame, tool_trails)
        for t in tools:
            trail = tool_trails.get(t["cls"])
            if trail:
                vx, vy, speed = compute_velocity(trail)
                draw_velocity_arrow(frame, t["cx"], t["cy"], vx, vy, speed)

        for d in draw_dets:
            color = (C_TOOL if d["group"]=="tool" else
                     C_VESSEL if d["group"]=="vessel" else C_ORGAN)
            cv2.rectangle(frame, (d["x1"],d["y1"]), (d["x2"],d["y2"]), color, 2)
            draw_label(frame, f"{CLASS_NAMES[d['cls']]} {d['conf']:.2f}",
                       d["x1"], d["y1"], color)

        for v in vessels:
            for radius, color in [(CAUTION_DIST, C_CAUTION),
                                  (WARNING_DIST, C_WARNING),
                                  (CRITICAL_DIST, C_CRITICAL)]:
                cv2.circle(frame, (v["cx"],v["cy"]), radius, color, 1)
            cv2.putText(frame, f"! {CLASS_NAMES[v['cls']]}",
                        (v["cx"]-50, v["cy"]-CAUTION_DIST-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_VESSEL, 1, cv2.LINE_AA)

        draw_organ_overlap_warning(frame, tools, organs, W)

        # Proximity analysis
        frame_min_dist   = 9999.0
        frame_alert_lvl  = 0
        alert_color      = C_OK
        frame_approaching = False

        for t in tools:
            danger_w = TOOL_DANGER_WEIGHT.get(t["cls"], 0.5)
            trail    = tool_trails.get(t["cls"])
            for v in vessels:
                dist     = math.hypot(t["cx"]-v["cx"], t["cy"]-v["cy"])
                eff_dist = dist / danger_w
                if dist < frame_min_dist:
                    frame_min_dist = dist

                app_rate_val = 0.0
                if trail:
                    app_rate_val = approach_rate(trail, v["cx"], v["cy"])
                    if app_rate_val < -1.0:
                        frame_approaching = True

                for threshold, msg, color, tier, freq in ALERT_TIERS:
                    if eff_dist < threshold:
                        cv2.line(frame, (t["cx"],t["cy"]), (v["cx"],v["cy"]), color, 2, cv2.LINE_AA)
                        full_msg = f"{msg}  [{CLASS_NAMES[t['cls']]} → {CLASS_NAMES[v['cls']]}]"
                        (bw, bh), _ = cv2.getTextSize(full_msg, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2)
                        bx = W//2 - bw//2
                        cv2.rectangle(frame, (bx-10,58), (bx+bw+10,98), (0,0,0), -1)
                        cv2.putText(frame, full_msg, (bx,88),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.72, color, 2, cv2.LINE_AA)
                        mx = (t["cx"]+v["cx"])//2
                        my = (t["cy"]+v["cy"])//2
                        cv2.putText(frame, f"{int(dist)}px", (mx+4, my-4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
                        if tier > frame_alert_lvl:
                            frame_alert_lvl = tier
                            alert_color     = color

                        # Log (in-memory, last 500 rows)
                        level_name = ["","CAUTION","WARNING","CRITICAL"][tier]
                        row_data = [
                            datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            stats["frames"], level_name,
                            CLASS_NAMES[t["cls"]], f"{t['conf']:.2f}",
                            CLASS_NAMES[v["cls"]], f"{dist:.1f}",
                            f"{app_rate_val:.2f}", f"{danger_w:.2f}",
                        ]
                        state["csv_rows"].append(row_data)
                        if len(state["csv_rows"]) > 500:
                            state["csv_rows"].pop(0)
                        break

        # Stats update
        stats["min_dist"]    = frame_min_dist
        stats["approaching"] = frame_approaching
        if frame_min_dist < stats["closest_ever"]:
            stats["closest_ever"] = frame_min_dist

        if frame_alert_lvl == 3:
            stats["total"] += 1; stats["critical"] += 1
            freeze_left = FREEZE_FRAMES
        elif frame_alert_lvl == 2:
            stats["total"] += 1; stats["warning"] += 1
        elif frame_alert_lvl == 1:
            stats["total"] += 1; stats["caution"] += 1

        stats["alert_level"] = frame_alert_lvl

        if frame_approaching and frame_alert_lvl == 0 and vessels:
            cv2.putText(frame, "Approaching vessel...", (W//2-100, H-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_CAUTION, 1, cv2.LINE_AA)

        if frame_alert_lvl:
            thickness = 12 if frame_alert_lvl == 3 else 6
            cv2.rectangle(frame, (0,0), (W,H), alert_color, thickness)

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_time)
        prev_time = now
        stats["fps"] = round(fps, 1)

        # Elapsed
        elapsed = int(time.time() - state["session_t0"])
        stats["elapsed"] = f"{elapsed//60:02d}:{elapsed%60:02d}"

        draw_hud(frame, tools, vessels, organs, fps, stats, conf_thresh,
                 state["paused"], W, H)

        # Store encoded JPEG for streaming
        ret2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret2:
            with state["frame_lock"]:
                state["latest_frame"] = buf.tobytes()

# ══════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════

HTML_DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Surgical Guardian v4</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #060a0f;
    --panel: #0c1219;
    --border: #1a2a3a;
    --cyan: #00ffe0;
    --red: #ff2244;
    --orange: #ff7700;
    --yellow: #ffdd00;
    --green: #00e080;
    --dim: #3a5060;
    --text: #b0ccd8;
    --mono: 'Share Tech Mono', monospace;
    --head: 'Barlow Condensed', sans-serif;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    min-height: 100vh;
    display: grid;
    grid-template-rows: 52px 1fr;
    grid-template-columns: 1fr 280px;
  }

  /* ── Header ── */
  header {
    grid-column: 1 / -1;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 24px;
    gap: 16px;
  }
  header .logo {
    font-family: var(--head);
    font-weight: 800;
    font-size: 1.3rem;
    letter-spacing: 0.12em;
    color: var(--cyan);
    text-transform: uppercase;
  }
  header .version { color: var(--dim); font-size: 0.75rem; }
  header .pill {
    margin-left: auto;
    padding: 3px 12px;
    border-radius: 2px;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    font-weight: 600;
    border: 1px solid;
  }
  header .pill.live   { color: var(--green); border-color: var(--green); background: rgba(0,224,128,.08); }
  header .pill.paused { color: var(--orange); border-color: var(--orange); background: rgba(255,119,0,.08); }

  /* ── Video ── */
  .video-area {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .video-wrapper {
    position: relative;
    background: #000;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    flex: 1;
  }
  .video-wrapper img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .alert-overlay {
    position: absolute;
    top: 0; left: 0; right: 0;
    pointer-events: none;
    transition: background 0.2s;
  }

  /* ── Controls row ── */
  .controls {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  button {
    font-family: var(--mono);
    font-size: 0.78rem;
    padding: 6px 14px;
    border-radius: 2px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
    cursor: pointer;
    letter-spacing: 0.06em;
    transition: all 0.15s;
  }
  button:hover { border-color: var(--cyan); color: var(--cyan); }
  button.danger { border-color: var(--red); color: var(--red); }
  button.danger:hover { background: rgba(255,34,68,.15); }
  .conf-row { display: flex; align-items: center; gap: 8px; font-size:0.78rem; }
  .conf-row input[type=range] {
    accent-color: var(--cyan);
    width: 120px;
  }
  .conf-val { color: var(--cyan); min-width: 36px; }

  /* ── Side panel ── */
  aside {
    background: var(--panel);
    border-left: 1px solid var(--border);
    padding: 16px 14px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  .section-title {
    font-family: var(--head);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--dim);
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }
  .stat-box {
    background: rgba(0,0,0,0.4);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 8px 10px;
  }
  .stat-box .label { font-size: 0.58rem; color: var(--dim); letter-spacing: 0.12em; text-transform: uppercase; }
  .stat-box .value { font-size: 1.2rem; font-family: var(--head); font-weight: 800; margin-top: 2px; }
  .stat-box.critical .value { color: var(--red); }
  .stat-box.warning  .value { color: var(--orange); }
  .stat-box.caution  .value { color: var(--yellow); }
  .stat-box.ok       .value { color: var(--green); }
  .stat-box.info     .value { color: var(--cyan); }

  .alert-banner {
    padding: 8px 12px;
    border-radius: 3px;
    font-size: 0.82rem;
    font-family: var(--head);
    font-weight: 600;
    letter-spacing: 0.06em;
    text-align: center;
    transition: all 0.3s;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--dim);
  }
  .alert-banner.level-3 { background: rgba(255,34,68,.18); border-color: var(--red); color: var(--red); animation: pulse 0.5s infinite; }
  .alert-banner.level-2 { background: rgba(255,119,0,.15); border-color: var(--orange); color: var(--orange); }
  .alert-banner.level-1 { background: rgba(255,221,0,.1);  border-color: var(--yellow); color: var(--yellow); }
  .alert-banner.level-0 { background: rgba(0,224,128,.07); border-color: var(--green); color: var(--green); }

  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

  /* Log table */
  .log-table { width:100%; font-size:0.62rem; border-collapse:collapse; }
  .log-table th { color: var(--dim); text-align: left; padding: 3px 4px; border-bottom: 1px solid var(--border); }
  .log-table td { padding: 2px 4px; border-bottom: 1px solid rgba(26,42,58,0.5); }
  .log-table .CRITICAL td { color: var(--red); }
  .log-table .WARNING  td { color: var(--orange); }
  .log-table .CAUTION  td { color: var(--yellow); }
  .log-table-wrap { max-height: 180px; overflow-y: auto; }

  /* dist meter */
  .dist-meter { position: relative; height: 8px; background: rgba(0,0,0,.4); border-radius: 4px; border: 1px solid var(--border); overflow: hidden; }
  .dist-fill { height: 100%; border-radius: 4px; transition: width 0.3s, background 0.3s; }

  @media (max-width: 800px) {
    body { grid-template-columns: 1fr; grid-template-rows: 52px auto auto; }
    aside { border-left: none; border-top: 1px solid var(--border); }
  }
</style>
</head>
<body>

<header>
  <span class="logo">Surgical Guardian</span>
  <span class="version">v4 · Web Edition</span>
  <span class="pill live" id="status-pill">LIVE</span>
</header>

<main class="video-area">
  <div class="video-wrapper">
    <img id="stream" src="/video_feed" alt="Live feed">
  </div>

  <div class="controls">
    <button onclick="sendCmd('pause')" id="btn-pause">⏸ PAUSE</button>
    <button onclick="sendCmd('reset')">↺ RESET</button>
    <button onclick="downloadLog()">⬇ EXPORT LOG</button>
    <div class="conf-row">
      CONF:
      <input type="range" id="conf-slider" min="5" max="95" value="30" step="5"
             oninput="updateConf(this.value)">
      <span class="conf-val" id="conf-display">0.30</span>
    </div>
  </div>
</main>

<aside>
  <div>
    <div class="section-title">Alert Status</div>
    <div class="alert-banner level-0" id="alert-banner">✓ NO ALERT</div>
  </div>

  <div>
    <div class="section-title">Session Stats</div>
    <div class="stat-grid">
      <div class="stat-box info">
        <div class="label">FPS</div>
        <div class="value" id="stat-fps">0</div>
      </div>
      <div class="stat-box info">
        <div class="label">Session</div>
        <div class="value" id="stat-elapsed">00:00</div>
      </div>
      <div class="stat-box info">
        <div class="label">Frames</div>
        <div class="value" id="stat-frames">0</div>
      </div>
      <div class="stat-box ok">
        <div class="label">Min Dist</div>
        <div class="value" id="stat-mindist">---</div>
      </div>
    </div>
  </div>

  <div>
    <div class="section-title">Proximity Events</div>
    <div class="stat-grid">
      <div class="stat-box critical">
        <div class="label">Critical</div>
        <div class="value" id="stat-critical">0</div>
      </div>
      <div class="stat-box warning">
        <div class="label">Warning</div>
        <div class="value" id="stat-warning">0</div>
      </div>
      <div class="stat-box caution">
        <div class="label">Caution</div>
        <div class="value" id="stat-caution">0</div>
      </div>
      <div class="stat-box info">
        <div class="label">Total</div>
        <div class="value" id="stat-total">0</div>
      </div>
    </div>
    <div style="margin-top:10px;">
      <div style="font-size:0.6rem;color:var(--dim);margin-bottom:4px;">CLOSEST APPROACH</div>
      <div class="dist-meter">
        <div class="dist-fill" id="dist-fill" style="width:0%;background:var(--green)"></div>
      </div>
      <div style="font-size:0.7rem;color:var(--cyan);margin-top:3px;" id="closest-ever">---</div>
    </div>
  </div>

  <div>
    <div class="section-title">Recent Alerts</div>
    <div class="log-table-wrap">
      <table class="log-table">
        <thead><tr><th>Time</th><th>Level</th><th>Tool</th><th>Dist</th></tr></thead>
        <tbody id="log-body"></tbody>
      </table>
    </div>
  </div>
</aside>

<script>
let paused = false;

async function sendCmd(cmd, data={}) {
  await fetch('/control', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({cmd, ...data})
  });
}

function updateConf(val) {
  const v = (parseInt(val)/100).toFixed(2);
  document.getElementById('conf-display').textContent = v;
  sendCmd('conf', {value: parseFloat(v)});
}

async function downloadLog() {
  const res = await fetch('/log.csv');
  const text = await res.text();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type:'text/csv'}));
  a.download = `alert_log_${Date.now()}.csv`;
  a.click();
}

function sendCmd_pause() {
  paused = !paused;
  document.getElementById('btn-pause').textContent = paused ? '▶ RESUME' : '⏸ PAUSE';
  document.getElementById('status-pill').textContent = paused ? 'PAUSED' : 'LIVE';
  document.getElementById('status-pill').className = paused ? 'pill paused' : 'pill live';
  sendCmd('pause');
}
document.getElementById('btn-pause').onclick = sendCmd_pause;

const LEVELS = [
  {cls:'level-0', text:'✓ NO ALERT'},
  {cls:'level-1', text:'⚠ CAUTION — Approaching'},
  {cls:'level-2', text:'⚠ WARNING — Too Close'},
  {cls:'level-3', text:'🛑 CRITICAL — STOP'},
];

let lastLogLen = 0;

async function pollStats() {
  try {
    const res = await fetch('/stats');
    const d   = await res.json();
    const s   = d.stats;

    document.getElementById('stat-fps').textContent     = s.fps;
    document.getElementById('stat-elapsed').textContent = s.elapsed;
    document.getElementById('stat-frames').textContent  = s.frames;
    document.getElementById('stat-critical').textContent= s.critical;
    document.getElementById('stat-warning').textContent = s.warning;
    document.getElementById('stat-caution').textContent = s.caution;
    document.getElementById('stat-total').textContent   = s.total;

    // Min dist
    const md = s.min_dist < 9999 ? s.min_dist : null;
    document.getElementById('stat-mindist').textContent = md ? Math.round(md)+'px' : '---';
    const ce = s.closest_ever < 9999 ? s.closest_ever : null;
    document.getElementById('closest-ever').textContent = ce ? 'Closest: '+Math.round(ce)+'px' : '---';

    // Distance bar
    const fill = document.getElementById('dist-fill');
    if (md) {
      const pct = Math.max(0, Math.min(100, 100 - (md/200)*100));
      fill.style.width = pct+'%';
      fill.style.background = md < 60 ? 'var(--red)' : md < 100 ? 'var(--orange)' : md < 150 ? 'var(--yellow)' : 'var(--green)';
    }

    // Alert banner
    const lvl = s.alert_level || 0;
    const banner = document.getElementById('alert-banner');
    banner.className = 'alert-banner ' + LEVELS[lvl].cls;
    banner.textContent = LEVELS[lvl].text;

    // Log table
    if (d.log && d.log.length !== lastLogLen) {
      lastLogLen = d.log.length;
      const tbody = document.getElementById('log-body');
      tbody.innerHTML = '';
      const rows = d.log.slice(-20).reverse();
      rows.forEach(r => {
        const tr = document.createElement('tr');
        tr.className = r[2];
        tr.innerHTML = `<td>${r[0]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[6]}px</td>`;
        tbody.appendChild(tr);
      });
    }
  } catch(e) {}
  setTimeout(pollStats, 500);
}

pollStats();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_DASHBOARD)

def gen_frames():
    """MJPEG stream generator."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for frames...", (140, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 200), 2)
    _, blank_buf = cv2.imencode(".jpg", blank)
    blank_bytes  = blank_buf.tobytes()

    while True:
        with state["frame_lock"]:
            frame_bytes = state.get("latest_frame")
        payload = frame_bytes if frame_bytes else blank_bytes
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n")
        time.sleep(0.033)   # ~30 fps ceiling for stream

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats_api():
    return jsonify({
        "stats": state["stats"],
        "log":   state["csv_rows"][-50:],
    })

@app.route("/control", methods=["POST"])
def control():
    data = request.get_json(silent=True) or {}
    cmd  = data.get("cmd", "")
    if cmd == "pause":
        state["paused"] = not state["paused"]
    elif cmd == "reset":
        s = state["stats"]
        s.update({"total":0,"critical":0,"warning":0,"caution":0,
                  "frames":0,"min_dist":9999.0,"closest_ever":9999.0,"alert_level":0})
        state["csv_rows"].clear()
        state["session_t0"] = time.time()
    elif cmd == "conf":
        val = float(data.get("value", 0.30))
        state["conf_thresh"] = max(0.05, min(0.95, val))
    return jsonify({"ok": True})

@app.route("/log.csv")
def log_csv():
    header = "timestamp,frame,level,tool,tool_conf,vessel,dist_px,approach_rate,tool_danger_weight\n"
    body   = "\n".join(",".join(str(c) for c in row) for row in state["csv_rows"])
    return Response(header + body, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=alert_log.csv"})

# ══════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════
def start_inference():
    model_path = os.environ.get("MODEL_PATH", "best.pt")
    source     = os.environ.get("VIDEO_SOURCE", "0")
    W          = int(os.environ.get("FRAME_WIDTH",  "640"))
    H          = int(os.environ.get("FRAME_HEIGHT", "480"))

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at '{model_path}'. "
              "Set MODEL_PATH env var or place best.pt alongside this file.")
        return

    print(f"[SurgicalGuardian] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[SurgicalGuardian] Opening source: {source}")
    cap = VideoCapture(source, W, H).start()
    if not cap.is_file:
        time.sleep(0.8)

    print("[SurgicalGuardian] Inference thread started.")
    t = threading.Thread(target=inference_loop, args=(model, cap, W, H), daemon=True)
    t.start()

if __name__ == "__main__":
    start_inference()
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)
