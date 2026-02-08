"""
Proximus.ai — Blind Assistance System
========================================
Run: python app_ui.py (from Anaconda Prompt)
Requires: pip install customtkinter
"""

import os
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np
import time
import threading
import customtkinter as ctk
from PIL import Image, ImageTk

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'core')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

# ============================================================
# Imports
# ============================================================
try:
    from core.vision import EnhancedVisionSystem
    print("[+] Vision")
except ImportError as e:
    print(f"[-] Vision: {e}")
    EnhancedVisionSystem = None

try:
    from paddleocr import PaddleOCR
    print("[+] PaddleOCR")
except ImportError:
    PaddleOCR = None
    print("[-] PaddleOCR")

try:
    from utils.audio_input import listen_and_transcribe, stop as audio_stop
    print("[+] Audio in")
except ImportError as e:
    print(f"[-] Audio in: {e}")
    listen_and_transcribe = None
    audio_stop = None

try:
    from utils.audio_output import speak
    print("[+] Audio out")
except ImportError as e:
    print(f"[-] Audio out: {e}")
    speak = None

try:
    from utils.interpret import find_and_guide
    print("[+] Gemini")
except ImportError as e:
    print(f"[-] Gemini: {e}")
    find_and_guide = None

from config import *

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# OCR Engine
# ============================================================
class InlineOCREngine:
    def __init__(self, confidence_threshold=0.6, scale_factor=0.5):
        if PaddleOCR is None:
            self.ocr = None
            return
        print("Loading PaddleOCR...")
        self.ocr = PaddleOCR(
            lang='en', use_doc_orientation_classify=False,
            use_doc_unwarping=False, use_textline_orientation=False,
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='PP-OCRv5_mobile_rec',
            enable_mkldnn=False,
        )
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        print("PaddleOCR ready")

    def scan_frame(self, frame):
        if self.ocr is None:
            return OCRResult([])
        h, w = frame.shape[:2]
        small = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor) if self.scale_factor < 1.0 else frame
        start = time.time()
        results = self.ocr.predict(input=small)
        detections = []
        for res in results:
            rd = res.get('res', res) if isinstance(res, dict) else {}
            if not isinstance(rd, dict):
                try: rd = res['res']
                except: continue
            for i in range(len(rd.get('rec_texts', []))):
                text = str(rd['rec_texts'][i]).strip()
                score = float(rd['rec_scores'][i]) if i < len(rd.get('rec_scores', [])) else 0.0
                if score < self.confidence_threshold or len(text) < 2:
                    continue
                bbox = np.array(rd['rec_polys'][i], dtype=np.float32) if i < len(rd.get('rec_polys', [])) else np.array([])
                if self.scale_factor < 1.0 and bbox.size > 0:
                    bbox = bbox / self.scale_factor
                bbox = bbox.astype(np.int32) if bbox.size > 0 else bbox
                cx = int(np.mean(bbox[:, 0])) if bbox.size > 0 else 0
                third = w / 3
                pos = "left" if cx < third else ("center" if cx < 2 * third else "right")
                detections.append(Det(text, score, bbox, cx, 0, pos))
        return OCRResult(detections, (time.time() - start) * 1000)


class Det:
    def __init__(self, text, confidence, bbox, center_x, center_y, position):
        self.text, self.confidence, self.bbox = text, confidence, bbox
        self.center_x, self.center_y, self.position = center_x, center_y, position


class OCRResult:
    def __init__(self, detections, scan_time_ms=0):
        self.detections, self.scan_time_ms = detections, scan_time_ms
    def get_all_text(self):
        return " | ".join([d.text for d in self.detections])


# ============================================================
# App
# ============================================================
class ProximusApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Proximus.ai")
        self.root.geometry("1500x900")
        self.root.state("zoomed")

        # State
        self.running = True
        self.mic_active = False
        self.current_frame = None
        self.raw_frame = None
        self.latest_detections = []
        self.latest_alerts = []
        self.latest_ocr_text = ""
        self.latest_ocr_detections = []
        self.ocr_scan_time = 0
        self.fps = 0
        self.frame_lock = threading.Lock()
        self.alert_history = []

        self._init_modules()
        self._build_ui()
        self._start_threads()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_modules(self):
        self.vision = None
        if EnhancedVisionSystem:
            try: self.vision = EnhancedVisionSystem(use_midas=False)
            except: pass
        try: self.ocr = InlineOCREngine(confidence_threshold=0.6, scale_factor=0.5)
        except: self.ocr = None
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # ========================================================
    # UI
    # ========================================================
    def _build_ui(self):

        # ── Right Panel FIRST (must pack before left so it claims space) ──
        self.right_panel = ctk.CTkFrame(self.root, width=400, corner_radius=16)
        self.right_panel.pack(side="right", fill="y", padx=(6, 12), pady=12)
        self.right_panel.pack_propagate(False)

        # ── Left Panel: Video ──
        self.left_panel = ctk.CTkFrame(self.root, corner_radius=16)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=12)

        # Camera feed
        self.video_label = ctk.CTkLabel(
            self.left_panel, text="Initializing Camera...",
            font=("Helvetica", 20), text_color="gray"
        )
        self.video_label.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        # Bottom bar under camera: OCR text + FPS
        cam_bottom = ctk.CTkFrame(self.left_panel, corner_radius=10, height=40,
                                   fg_color=("#2b2b3d", "#1a1a2e"))
        cam_bottom.pack(fill="x", padx=8, pady=(0, 8))
        cam_bottom.pack_propagate(False)

        self.ocr_dot = ctk.CTkLabel(cam_bottom, text="\u2022", font=("Helvetica", 18),
                                     text_color="#56d4c8", width=20)
        self.ocr_dot.pack(side="left", padx=(12, 4))

        self.ocr_label = ctk.CTkLabel(cam_bottom, text="Scanning...",
                                       font=("Consolas", 12), text_color="gray",
                                       anchor="w")
        self.ocr_label.pack(side="left", fill="x", expand=True, padx=4)

        self.fps_label = ctk.CTkLabel(cam_bottom, text="-- FPS",
                                       font=("Consolas", 11), text_color="#50e3a4")
        self.fps_label.pack(side="right", padx=12)

        self.ocr_ms_label = ctk.CTkLabel(cam_bottom, text="",
                                          font=("Consolas", 11), text_color="gray")
        self.ocr_ms_label.pack(side="right", padx=(0, 8))

        # ── Right Panel Contents ──

        # Title
        title_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        title_frame.pack(pady=(12, 0), padx=20, fill="x")

        ctk.CTkLabel(title_frame, text="Proximus", font=("Helvetica", 30, "bold"),
                     text_color="white").pack(side="left")
        ctk.CTkLabel(title_frame, text=".ai", font=("Helvetica", 30, "bold"),
                     text_color="#7c8cf8").pack(side="left")

        ctk.CTkLabel(self.right_panel, text="AI Vision Assistant",
                     font=("Helvetica", 13), text_color="gray").pack(
                         pady=(0, 8), padx=20, anchor="w")

        # Status
        self.status_frame = ctk.CTkFrame(self.right_panel, corner_radius=10,
                                          fg_color=("#1a2e1a", "#122212"), height=36)
        self.status_frame.pack(fill="x", padx=20, pady=(0, 6))
        self.status_frame.pack_propagate(False)

        self.status_dot = ctk.CTkLabel(self.status_frame, text="\u2022",
                                        font=("Helvetica", 22), text_color="#4CAF50", width=20)
        self.status_dot.pack(side="left", padx=(12, 4))
        self.status_label = ctk.CTkLabel(self.status_frame, text="System Ready",
                                          font=("Helvetica", 13), text_color="#4CAF50")
        self.status_label.pack(side="left")

        # ── Ask Button ──
        self.nav_button = ctk.CTkButton(
            self.right_panel, text="\u2022  Ask a Question",
            font=("Helvetica", 17, "bold"), height=56, corner_radius=28,
            fg_color="#7c8cf8", hover_color="#6670d9",
            command=self._toggle_mic
        )
        self.nav_button.pack(pady=(2, 8), padx=20, fill="x")

        # ── Transcript ──
        ctk.CTkLabel(self.right_panel, text="TRANSCRIPT",
                     font=("Helvetica", 11, "bold"), text_color="gray",
                     anchor="w").pack(padx=24, pady=(4, 1), fill="x")

        self.transcript_box = ctk.CTkTextbox(
            self.right_panel, height=55, corner_radius=10,
            font=("Consolas", 11), wrap="word",
            fg_color=("#2b2b3d", "#1a1a2e"), border_width=0
        )
        self.transcript_box.pack(padx=20, pady=(0, 4), fill="x")
        self.transcript_box.insert("1.0", "Press the button or SPACE to ask...")
        self.transcript_box.configure(state="disabled")

        # ── Detected Text (OCR) ──
        ctk.CTkLabel(self.right_panel, text="DETECTED TEXT",
                     font=("Helvetica", 11, "bold"), text_color="gray",
                     anchor="w").pack(padx=24, pady=(4, 1), fill="x")

        self.ocr_box = ctk.CTkTextbox(
            self.right_panel, height=55, corner_radius=10,
            font=("Consolas", 11), wrap="word",
            fg_color=("#2b2b3d", "#1a1a2e"), border_width=0
        )
        self.ocr_box.pack(padx=20, pady=(0, 4), fill="x")
        self.ocr_box.insert("1.0", "No text detected yet")
        self.ocr_box.configure(state="disabled")

        # ── Alerts / Warnings ──
        ctk.CTkLabel(self.right_panel, text="ALERTS",
                     font=("Helvetica", 11, "bold"), text_color="#f65866",
                     anchor="w").pack(padx=24, pady=(4, 1), fill="x")

        self.alert_box = ctk.CTkTextbox(
            self.right_panel, height=55, corner_radius=10,
            font=("Consolas", 11), wrap="word",
            fg_color=("#2b2b3d", "#1a1a2e"), border_width=0,
            text_color="#f65866"
        )
        self.alert_box.pack(padx=20, pady=(0, 4), fill="x")
        self.alert_box.insert("1.0", "No warnings")
        self.alert_box.configure(state="disabled")

        # ── Objects ──
        ctk.CTkLabel(self.right_panel, text="ENVIRONMENT",
                     font=("Helvetica", 11, "bold"), text_color="#50e3a4",
                     anchor="w").pack(padx=24, pady=(4, 1), fill="x")

        self.objects_box = ctk.CTkTextbox(
            self.right_panel, height=55, corner_radius=10,
            font=("Consolas", 11), wrap="word",
            fg_color=("#2b2b3d", "#1a1a2e"), border_width=0,
            text_color="#a0aab5"
        )
        self.objects_box.pack(padx=20, pady=(0, 8), fill="both", expand=True)
        self.objects_box.insert("1.0", "No objects detected")
        self.objects_box.configure(state="disabled")

        # Keyboard
        self.root.bind("<space>", lambda e: self._toggle_mic())
        self.root.bind("<Escape>", lambda e: self._on_close())

    # ========================================================
    # Threads
    # ========================================================
    def _start_threads(self):
        threading.Thread(target=self._vision_loop, daemon=True).start()
        threading.Thread(target=self._ocr_loop, daemon=True).start()
        self._tick()

    def _vision_loop(self):
        fc, t0 = 0, time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            raw = frame.copy()
            detections, alerts, annotated = [], [], frame.copy()

            if self.vision:
                try:
                    data = self.vision.process_frame(frame)
                    detections = data.get('detections', [])
                    alerts = data.get('alerts', [])
                    annotated = self.vision.draw_detections(frame.copy(), detections)
                except:
                    pass

            # OCR boxes on feed — soft teal
            for det in self.latest_ocr_detections:
                if det.bbox.size > 0:
                    pts = det.bbox.astype(np.int32)
                    cv2.polylines(annotated, [pts], True, (200, 212, 86), 2)
                    x, y = int(pts[0][0]), int(pts[0][1]) - 6
                    label = f"{det.text}  {det.confidence:.0%}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    cv2.rectangle(annotated, (x - 2, max(y - lh - 8, 0)),
                                  (x + lw + 8, max(y + 4, lh + 8)), (20, 24, 36), -1)
                    cv2.putText(annotated, label, (x + 3, max(y - 1, 14)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 212, 86), 1, cv2.LINE_AA)

            with self.frame_lock:
                self.raw_frame = raw
                self.current_frame = annotated
                self.latest_detections = detections
                self.latest_alerts = alerts

            fc += 1
            el = time.time() - t0
            if el > 1.0:
                self.fps = fc / el
                fc, t0 = 0, time.time()
            time.sleep(0.03)

    def _ocr_loop(self):
        while self.running:
            if self.ocr and self.raw_frame is not None:
                try:
                    with self.frame_lock:
                        f = self.raw_frame.copy()
                    r = self.ocr.scan_frame(f)
                    self.latest_ocr_detections = r.detections
                    self.latest_ocr_text = r.get_all_text() if r.detections else ""
                    self.ocr_scan_time = r.scan_time_ms
                except:
                    pass
            time.sleep(OCR_INTERVAL)

    # ========================================================
    # UI Tick
    # ========================================================
    def _tick(self):
        if not self.running:
            return

        with self.frame_lock:
            frame = self.current_frame

        # Camera feed
        if frame is not None:
            try:
                cw = self.video_label.winfo_width()
                ch = self.video_label.winfo_height()
                if cw > 10 and ch > 10:
                    h, w = frame.shape[:2]
                    s = min(cw / w, ch / h)
                    nw, nh = max(1, int(w * s)), max(1, int(h * s))
                    resized = cv2.resize(frame, (nw, nh))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                                           size=(nw, nh))
                    self.video_label.configure(image=ctk_img, text="")
                    self.video_label._ctk_img = ctk_img  # keep reference
            except:
                pass

        # FPS
        fv = self.fps
        fc = "#50e3a4" if fv > 8 else ("#f0c674" if fv > 4 else "#f65866")
        self.fps_label.configure(text=f"{fv:.0f} FPS", text_color=fc)

        # Status
        if frame is not None:
            self.status_dot.configure(text_color="#4CAF50")
            self.status_label.configure(text="System Active", text_color="#4CAF50")
            self.status_frame.configure(fg_color=("#1a2e1a", "#122212"))
        else:
            self.status_dot.configure(text_color="#f65866")
            self.status_label.configure(text="No Camera", text_color="#f65866")
            self.status_frame.configure(fg_color=("#2e1a1a", "#221212"))

        # OCR strip
        if self.latest_ocr_text:
            self.ocr_label.configure(text=self.latest_ocr_text, text_color="white")
            self.ocr_ms_label.configure(text=f"{self.ocr_scan_time:.0f}ms", text_color="#56d4c8")
            self.ocr_dot.configure(text_color="#56d4c8")
        else:
            self.ocr_label.configure(text="No text detected", text_color="gray")
            self.ocr_ms_label.configure(text="")
            self.ocr_dot.configure(text_color="gray")

        # OCR box
        self._set_textbox(self.ocr_box,
            "\n".join(f"\u2022 {d.text} ({d.confidence:.0%}) [{d.position}]"
                      for d in self.latest_ocr_detections) if self.latest_ocr_detections
            else "No text detected")

        # Alerts
        alerts = self.latest_alerts
        if alerts:
            lines = []
            for a in alerts:
                sev = a.get('severity', 'INFO')
                icon = "\u26A0" if sev == "CRITICAL" else ("\u25B3" if sev == "WARNING" else "\u2022")
                lines.append(f"{icon} [{sev}] {a.get('message', '')}")
            # Add history
            for a in alerts:
                if a.get('severity') in ('CRITICAL', 'WARNING'):
                    ts = time.strftime("%H:%M:%S")
                    entry = f"{ts}  {a['message']}"
                    if entry not in self.alert_history[-10:]:
                        self.alert_history.append(entry)
            if self.alert_history:
                lines.append("")
                for e in self.alert_history[-4:]:
                    lines.append(f"  {e}")
            self._set_textbox(self.alert_box, "\n".join(lines))
        else:
            self._set_textbox(self.alert_box, "\u2022 All clear")

        # Objects
        dets = self.latest_detections
        if dets:
            counts = {}
            for d in dets:
                counts.setdefault(d.get('class_name', '?'), []).append(d)
            lines = []
            for n, objs in sorted(counts.items()):
                if len(objs) == 1:
                    o = objs[0]
                    dist = o.get('distance', 0)
                    dirn = o.get('direction', '?')
                    lines.append(f"\u2022 {n}  \u2014  {dirn}, {dist:.1f}m")
                else:
                    ds = [f"{o.get('distance',0):.1f}m" for o in objs]
                    lines.append(f"\u2022 {n} \u00D7{len(objs)}  \u2014  {', '.join(ds)}")
            self._set_textbox(self.objects_box, "\n".join(lines))
        else:
            self._set_textbox(self.objects_box, "No objects detected")

        self.root.after(33, self._tick)

    def _set_textbox(self, box, text):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")

    # ========================================================
    # Mic
    # ========================================================
    def _toggle_mic(self):
        if self.mic_active:
            return
        self.mic_active = True
        self.nav_button.configure(text="\u2022  Listening...",
                                   fg_color="#f65866", hover_color="#d94040",
                                   state="disabled")
        self.status_dot.configure(text_color="#f65866")
        self.status_label.configure(text="Listening...", text_color="#f65866")
        self.status_frame.configure(fg_color=("#2e1a1a", "#221212"))

        threading.Thread(target=self._handle_speech, daemon=True).start()

    def _handle_speech(self):
        try:
            transcript = ""
            if listen_and_transcribe:
                # Auto-stop listening after 4 seconds
                def _auto_stop():
                    time.sleep(4)
                    if audio_stop:
                        audio_stop()
                threading.Thread(target=_auto_stop, daemon=True).start()
                transcript = listen_and_transcribe()
            else:
                transcript = "where is the exit"
                time.sleep(2)

            if not transcript or len(transcript.strip()) < 3:
                self._set_textbox_safe(self.transcript_box, "Didn't catch that. Try again.")
                if speak:
                    speak("Sorry, I didn't hear anything.")
                self._mic_reset()
                return

            self._set_textbox_safe(self.transcript_box, f"You: {transcript}\n\nSearching...")

            # Update status
            self.root.after(0, lambda: self.status_label.configure(
                text="Processing...", text_color="#7c8cf8"))

            scan_start = time.time()
            result = {}
            while (time.time() - scan_start) < 5.0:
                with self.frame_lock:
                    if self.raw_frame is None:
                        time.sleep(0.1)
                        continue
                    fc = self.raw_frame.copy()

                ocr_texts = []
                if self.ocr:
                    ocr_r = self.ocr.scan_frame(fc)
                    ocr_texts = [d.text for d in ocr_r.detections]
                yolo_objs = [d.get('class_name', '') for d in self.latest_detections]

                if find_and_guide:
                    result = find_and_guide(transcript, ocr_texts, yolo_objs)
                else:
                    result = {
                        'found': bool(ocr_texts or yolo_objs),
                        'instructions': f"I see: {', '.join(ocr_texts + yolo_objs)}" if (ocr_texts or yolo_objs) else "Nothing detected."
                    }

                if result.get('found'):
                    instr = result.get('instructions', '')
                    self._set_textbox_safe(self.transcript_box,
                        f"You: {transcript}\n\nGuide: {instr}")
                    if speak:
                        speak(instr)
                    self._mic_reset()
                    return
                time.sleep(0.3)

            instr = result.get('instructions', "Couldn't find that nearby.")
            self._set_textbox_safe(self.transcript_box,
                f"You: {transcript}\n\nGuide: {instr}")
            if speak:
                speak(instr)
        except Exception as e:
            self._set_textbox_safe(self.transcript_box, f"Error: {e}")
        finally:
            self._mic_reset()

    def _set_textbox_safe(self, box, text):
        self.root.after(0, lambda: self._set_textbox(box, text))

    def _mic_reset(self):
        self.mic_active = False
        def _do():
            self.nav_button.configure(
                text="\u2022  Ask a Question",
                fg_color="#7c8cf8", hover_color="#6670d9",
                state="normal"
            )
            self.status_dot.configure(text_color="#4CAF50")
            self.status_label.configure(text="System Active", text_color="#4CAF50")
            self.status_frame.configure(fg_color=("#1a2e1a", "#122212"))
        self.root.after(0, _do)

    def _on_close(self):
        self.running = False
        time.sleep(0.2)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("=" * 40)
    print("Proximus.ai")
    print("=" * 40)
    app = ProximusApp()
    app.run()