"""
Modern GUI Version - Using CustomTkinter
Beautiful, modern interface with dark theme and smooth animations
"""

import cv2
import numpy as np
import time
import threading
import sys
import os
from PIL import Image, ImageTk
import customtkinter as ctk

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import *
from core.vision import EnhancedVisionSystem
from utils.ocr_engine import OCREngine
from utils.audio_input import listen_and_transcribe, stop as stop_audio_input
from utils.audio_output import speak
from utils.interpret import find_and_guide

# Set appearance mode and default color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# =============================================================================
# GLOBAL STATE
# =============================================================================
running = True
listening_active = False
space_is_held = False

current_frame = None
frame_lock = threading.Lock()

latest_vision_data = None
vision_lock = threading.Lock()

last_critical_alert_time = 0
CRITICAL_ALERT_COOLDOWN = 3.0

audio_output_lock = threading.Lock()

user_input_text = ""
gemini_output_text = ""
critical_warning_text = ""
ocr_detected_texts = []
text_lock = threading.Lock()


class ModernAssistiveGUI(ctk.CTk):
    def __init__(self, vision_system, ocr_engine):
        super().__init__()
        
        self.vision_system = vision_system
        self.ocr_engine = ocr_engine
        
        # Configure window
        self.title("ProximusAI")
        self.geometry("1400x800")
        
        # Configure grid layout
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # ========== LEFT SIDE - VIDEO FEED ==========
        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)
        
        # ========== RIGHT SIDE - CONTROL PANEL ==========
        self.control_frame = ctk.CTkFrame(self, corner_radius=10)
        self.control_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        
        # Header
        self.header_label = ctk.CTkLabel(
            self.control_frame,
            text="⚡ ProximusAI",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#00D9FF"
        )
        self.header_label.pack(pady=(20, 5))
        
        self.subtitle_label = ctk.CTkLabel(
            self.control_frame,
            text="AI-Powered Vision Assistant",
            font=ctk.CTkFont(size=14),
            text_color="#00D9FF"
        )
        self.subtitle_label.pack(pady=(0, 20))
        
        # FPS Display
        self.fps_frame = ctk.CTkFrame(self.control_frame, corner_radius=8, fg_color="#1a1a2e")
        self.fps_frame.pack(pady=10, padx=20, fill="x")
        
        self.fps_label = ctk.CTkLabel(
            self.fps_frame,
            text="FPS: --",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#00ff9f"
        )
        self.fps_label.pack(pady=10)
        
        # Status Indicator
        self.status_frame = ctk.CTkFrame(self.control_frame, corner_radius=8, fg_color="#1a1a2e")
        self.status_frame.pack(pady=10, padx=20, fill="x")
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="● Ready",
            font=ctk.CTkFont(size=14),
            text_color="#00D9FF"
        )
        self.status_label.pack(pady=12)
        
        # Talk Button
        self.talk_button = ctk.CTkButton(
            self.control_frame,
            text="🎤 HOLD TO TALK",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            corner_radius=10,
            fg_color="#00D9FF",
            hover_color="#00B8D4",
            text_color="#000000"
        )
        self.talk_button.pack(pady=15, padx=20, fill="x")
        self.talk_button.bind("<ButtonPress-1>", self.start_listening)
        self.talk_button.bind("<ButtonRelease-1>", self.stop_listening)
        
        # Critical Warning Box (hidden by default)
        self.warning_frame = ctk.CTkFrame(self.control_frame, corner_radius=8, fg_color="#FF1744")
        self.warning_label = ctk.CTkLabel(
            self.warning_frame,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            wraplength=300,
            text_color="#FFFFFF"
        )
        self.warning_label.pack(pady=10, padx=10)
        
        # OCR Results
        self.ocr_title = ctk.CTkLabel(
            self.control_frame,
            text="📝 OCR Detected Text",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00D9FF"
        )
        self.ocr_title.pack(pady=(20, 5), padx=20, anchor="w")
        
        self.ocr_textbox = ctk.CTkTextbox(
            self.control_frame,
            height=120,
            corner_radius=8,
            font=ctk.CTkFont(size=12),
            fg_color="#1a1a2e",
            text_color="#FFFFFF"
        )
        self.ocr_textbox.pack(pady=5, padx=20, fill="x")
        self.ocr_textbox.configure(state="disabled")
        
        # Conversation
        self.conversation_title = ctk.CTkLabel(
            self.control_frame,
            text="💬 Conversation",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00D9FF"
        )
        self.conversation_title.pack(pady=(20, 5), padx=20, anchor="w")
        
        self.conversation_textbox = ctk.CTkTextbox(
            self.control_frame,
            height=200,
            corner_radius=8,
            font=ctk.CTkFont(size=12),
            fg_color="#1a1a2e",
            text_color="#FFFFFF"
        )
        self.conversation_textbox.pack(pady=5, padx=20, fill="both", expand=True)
        self.conversation_textbox.configure(state="disabled")
        
        # Quit Button
        self.quit_button = ctk.CTkButton(
            self.control_frame,
            text="❌ Quit",
            font=ctk.CTkFont(size=14),
            height=40,
            corner_radius=8,
            fg_color="#FF1744",
            hover_color="#D50000",
            command=self.quit_app
        )
        self.quit_button.pack(pady=20, padx=20, fill="x", side="bottom")
        
        # Start camera
        self.cap = None
        self.video_running = True
        self.start_camera()
        
        # Bind keyboard
        self.bind("<space>", self.space_pressed)
        self.bind("<KeyRelease-space>", self.space_released)
        
        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
    
    def start_camera(self):
        """Initialize and start camera feed"""
        self.cap = self.vision_system.open_camera()
        if self.cap:
            self.update_video()
        else:
            self.video_label.configure(text="❌ Camera Error")
    
    def update_video(self):
        """Update video frame"""
        global current_frame, latest_vision_data
        
        if not self.video_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Store frame
            with frame_lock:
                current_frame = frame.copy()
            
            # Process with YOLO
            vision_data = self.vision_system.process_frame(frame)
            
            with vision_lock:
                latest_vision_data = vision_data
            
            # Check critical alerts
            self.check_critical_alerts(vision_data)
            
            # Draw detections
            if SHOW_DETECTIONS:
                annotated = self.vision_system.draw_detections(
                    frame,
                    vision_data['detections'],
                    None,
                    vision_data['tracking']
                )
            else:
                annotated = frame.copy()
            
            # Convert to PIL and display
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize to fit
            display_width = self.video_label.winfo_width()
            display_height = self.video_label.winfo_height()
            
            if display_width > 10 and display_height > 10:
                img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
            
            # Update FPS
            self.fps_label.configure(text=f"FPS: {vision_data['fps']:.1f}")
        
        # Schedule next update
        self.after(30, self.update_video)
    
    def check_critical_alerts(self, vision_data):
        """Check for critical warnings"""
        global last_critical_alert_time, critical_warning_text
        
        current_time = time.time()
        
        if current_time - last_critical_alert_time < CRITICAL_ALERT_COOLDOWN:
            return
        
        critical_alerts = [
            alert for alert in vision_data['alerts']
            if alert['severity'] == 'CRITICAL'
        ]
        
        if not critical_alerts:
            self.warning_frame.pack_forget()
            with text_lock:
                critical_warning_text = ""
            return
        
        alert = critical_alerts[0]
        
        # Find pan angle
        pan_angle = 0
        for det in vision_data['detections']:
            if det['class_name'] == alert['object']:
                pan_angle = det['angle']
                break
        
        # Generate warning
        from utils.interpret import generate_critical_warning
        gemini_warning = generate_critical_warning(alert)
        message = gemini_warning if gemini_warning else alert['message']
        
        # Update warning display
        with text_lock:
            critical_warning_text = message
        
        self.warning_label.configure(text=f"⚠️ {message}")
        self.warning_frame.pack(pady=10, padx=20, fill="x", after=self.status_frame)
        
        # Speak alert
        def speak_alert():
            with audio_output_lock:
                speak(message, pan=pan_angle)
        
        threading.Thread(target=speak_alert, daemon=True).start()
        last_critical_alert_time = current_time
    
    def space_pressed(self, event):
        """Handle space key press"""
        global space_is_held, listening_active
        if not space_is_held and not listening_active:
            space_is_held = True
            self.start_listening(None)
    
    def space_released(self, event):
        """Handle space key release"""
        global space_is_held
        if space_is_held:
            space_is_held = False
            self.stop_listening(None)
    
    def start_listening(self, event):
        """Start listening for user input"""
        global listening_active
        
        listening_active = True
        self.status_label.configure(text="● Listening...", text_color="#00ff9f")
        self.talk_button.configure(fg_color="#00ff9f", hover_color="#00E676", text="🎤 LISTENING...", text_color="#000000")
        
        # Start audio thread
        threading.Thread(target=self.handle_audio, daemon=True).start()
    
    def stop_listening(self, event):
        """Stop listening"""
        stop_audio_input()
    
    def handle_audio(self):
        """Handle audio input in background"""
        global listening_active, user_input_text
        
        try:
            with text_lock:
                user_input_text = "Listening..."
            
            transcript = listen_and_transcribe()
            
            listening_active = False
            self.status_label.configure(text="● Ready", text_color="#00D9FF")
            self.talk_button.configure(fg_color="#00D9FF", hover_color="#00B8D4", text="🎤 HOLD TO TALK", text_color="#000000")
            
            if not transcript or len(transcript.strip()) < 3:
                return
            
            with text_lock:
                user_input_text = transcript
            
            # Update conversation
            self.update_conversation(f"You: {transcript}\n", "user")
            
            # Process with Gemini
            self.process_query(transcript)
        
        except Exception as e:
            listening_active = False
            self.status_label.configure(text="● Ready", text_color="#00D9FF")
            self.talk_button.configure(fg_color="#00D9FF", hover_color="#00B8D4", text="🎤 HOLD TO TALK", text_color="#000000")
    
    def process_query(self, user_speech):
        """Process user query with OCR and Gemini"""
        global current_frame, latest_vision_data, gemini_output_text, ocr_detected_texts
        
        self.update_conversation("Assistant: Processing...\n", "assistant")
        
        # Get frame
        with frame_lock:
            if current_frame is None:
                response = "No camera frame available."
                self.update_conversation(f"Assistant: {response}\n", "assistant")
                speak(response)
                return
            frame = current_frame.copy()
        
        # Get vision data
        with vision_lock:
            vision_data = latest_vision_data if latest_vision_data else {'detections': []}
        
        # Run OCR
        ocr_result = self.ocr_engine.scan_frame(frame)
        ocr_texts = [d.text for d in ocr_result.detections]
        
        with text_lock:
            ocr_detected_texts = ocr_texts.copy()
        
        # Update OCR display
        self.update_ocr_display(ocr_texts)
        
        # Get YOLO objects
        yolo_objects = [
            f"{det['class_name']} at {det['distance']:.1f}m {det['direction']}"
            for det in vision_data['detections']
        ]
        
        # Query Gemini
        result = find_and_guide(user_speech, ocr_texts, yolo_objects)
        
        with text_lock:
            gemini_output_text = result['instructions']
        
        # Update conversation
        self.update_conversation(f"Assistant: {result['instructions']}\n", "assistant")
        
        # Speak
        with audio_output_lock:
            speak(result['instructions'])
    
    def update_ocr_display(self, texts):
        """Update OCR textbox"""
        self.ocr_textbox.configure(state="normal")
        self.ocr_textbox.delete("1.0", "end")
        
        if texts:
            for i, text in enumerate(texts[:5], 1):
                self.ocr_textbox.insert("end", f"{i}. {text}\n")
        else:
            self.ocr_textbox.insert("end", "No text detected")
        
        self.ocr_textbox.configure(state="disabled")
    
    def update_conversation(self, text, sender):
        """Update conversation textbox"""
        self.conversation_textbox.configure(state="normal")
        
        # Remove "Processing..." if it exists
        content = self.conversation_textbox.get("1.0", "end")
        if "Processing..." in content:
            lines = content.split('\n')
            filtered = [line for line in lines if "Processing..." not in line]
            self.conversation_textbox.delete("1.0", "end")
            self.conversation_textbox.insert("1.0", '\n'.join(filtered))
        
        self.conversation_textbox.insert("end", text)
        self.conversation_textbox.see("end")
        self.conversation_textbox.configure(state="disabled")
    
    def quit_app(self):
        """Clean shutdown"""
        global running
        running = False
        self.video_running = False
        
        if self.cap:
            self.cap.release()
        
        stop_audio_input()
        self.destroy()


def main():
    """Main entry point"""
    print("=" * 70)
    print("PROXIMUS AI - AI-POWERED VISION ASSISTANT")
    print("=" * 70)
    print("\nInitializing...")
    
    # Initialize systems
    print("📸 Loading YOLO + MiDaS...")
    vision_system = EnhancedVisionSystem(use_midas=True)
    
    print("🔤 Loading OCR engine...")
    ocr_engine = OCREngine(confidence_threshold=0.6, use_gpu=False, scale_factor=0.5)
    
    print("\n✅ ALL SYSTEMS READY")
    print("\nLaunching GUI...")
    
    # Create and run GUI
    app = ModernAssistiveGUI(vision_system, ocr_engine)
    app.mainloop()
    
    print("\n✅ SHUTDOWN COMPLETE")


if __name__ == "__main__":
    main()
