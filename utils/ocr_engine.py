"""
OCR Engine Module - Extracted from notebooks/ocr.ipynb
========================================================

This module provides OCR functionality using PaddleOCR PP-OCRv5.
It detects text in camera frames and returns structured results with
spatial positioning information.

Usage:
    from utils.ocr_engine import OCREngine
    
    engine = OCREngine(confidence_threshold=0.6)
    result = engine.scan_frame(frame)
    print(result.get_all_text())  # "REXALL PHARMACY | OPEN 9AM-9PM"
"""

import os
# CRITICAL: Must be set before other imports
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np
import time
from typing import Optional
from dataclasses import dataclass
from paddleocr import PaddleOCR


@dataclass
class TextDetection:
    """
    Represents ONE piece of text found in the camera frame.
    
    Example:
        text = "REXALL PHARMACY"
        confidence = 0.98  (98% sure it read correctly)
        position = "left"  (it's on the left side of what the camera sees)
    """
    text: str           # the actual text that was read
    confidence: float   # how sure the model is (0.0 to 1.0)
    bbox: np.ndarray    # bounding box coordinates (4 corners)
    center_x: int = 0   # center x position in the frame
    center_y: int = 0   # center y position in the frame
    position: str = ""  # "left", "center", or "right"


@dataclass
class OCRResult:
    """
    All the text found in a single camera frame.
    Contains a list of TextDetection objects + some metadata.
    """
    detections: list          # list of TextDetection objects
    timestamp: float = 0.0    # when this scan happened
    scan_time_ms: float = 0.0 # how long the scan took in milliseconds
    frame_width: int = 0
    frame_height: int = 0
    
    def get_all_text(self) -> str:
        """
        Returns all detected text as one string.
        Used when sending context to Gemini.
        
        Example output: "REXALL PHARMACY | OPEN 9AM-9PM | EXIT"
        """
        return " | ".join([d.text for d in self.detections])
    
    def get_text_with_positions(self) -> str:
        """
        Returns text WITH spatial info (where it is in the frame).
        This helps Gemini give directional instructions.
        
        Example output:
            [LEFT] 'REXALL PHARMACY' (98%)
            [CENTER] 'OPEN 9AM-9PM' (95%)
            [RIGHT] 'EXIT' (92%)
        """
        lines = []
        for d in self.detections:
            lines.append(f"[{d.position.upper()}] '{d.text}' ({d.confidence:.0%})")
        return "\n".join(lines)


class OCREngine:
    """
    Simple OCR engine — call scan_frame() when you need to read text.
    Not running in background. Just call it when needed.
    
    Usage:
        engine = OCREngine()
        result = engine.scan_frame(frame)
        print(result.get_all_text())  # "REXALL PHARMACY | OPEN 9AM-9PM"
    """
    
    def __init__(self, confidence_threshold: float = 0.6, use_gpu: bool = False,
                 scale_factor: float = 0.5):
        """
        Initialize OCR engine.
        
        Args:
            confidence_threshold: Minimum confidence to keep detections (0.0-1.0)
            use_gpu: Whether to use GPU acceleration
            scale_factor: Resize factor for speed (0.5 = half size, faster)
        """
        print("Loading PaddleOCR model (first time downloads ~200MB)...")
        start = time.time()
        
        self.ocr = PaddleOCR(
            lang='en',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='PP-OCRv5_mobile_rec',
            enable_mkldnn=False,
        )
        
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        
        elapsed = time.time() - start
        print(f"PaddleOCR loaded in {elapsed:.1f}s")
    
    
    def _get_position(self, center_x: int, frame_width: int) -> str:
        """
        Determine if text is on left, center, or right of frame.
        
        Args:
            center_x: X coordinate of text center
            frame_width: Width of the frame
            
        Returns:
            "left", "center", or "right"
        """
        third = frame_width / 3
        if center_x < third:
            return "left"
        elif center_x < 2 * third:
            return "center"
        else:
            return "right"
    
    
    def scan_frame(self, frame: np.ndarray) -> OCRResult:
        """
        Call this when user wants to read text.
        Pass in the current camera frame, get back all detected text.
        
        Args:
            frame: Camera frame (numpy array from cv2.VideoCapture)
            
        Returns:
            OCRResult with all detected text and metadata
        """
        start = time.time()
        h, w = frame.shape[:2]
        
        # Resize for speed if needed
        if self.scale_factor < 1.0:
            small_frame = cv2.resize(frame, None,
                                      fx=self.scale_factor,
                                      fy=self.scale_factor)
        else:
            small_frame = frame
        
        # Run OCR
        results = self.ocr.ocr(small_frame, cls=False)
        
        detections = []
        
        # Parse results - PaddleOCR returns [[[bbox], (text, confidence)], ...]
        if results and results[0]:
            for line in results[0]:
                if not line:
                    continue
                
                bbox_coords = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]    # (text, confidence)
                
                text = str(text_info[0]).strip()
                score = float(text_info[1])
                
                # Filter low confidence and short text
                if score < self.confidence_threshold:
                    continue
                if len(text) < 2:
                    continue
                
                # Get bounding box
                bbox = np.array(bbox_coords, dtype=np.float32)
                # Scale back to original frame size
                if self.scale_factor < 1.0:
                    bbox = bbox / self.scale_factor
                bbox = bbox.astype(np.int32)
                
                # Calculate center position
                cx = int(np.mean(bbox[:, 0]))
                cy = int(np.mean(bbox[:, 1]))
                
                # Create detection object
                det = TextDetection(
                    text=text,
                    confidence=score,
                    bbox=bbox,
                    center_x=cx,
                    center_y=cy,
                    position=self._get_position(cx, w)
                )
                detections.append(det)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return OCRResult(
            detections=detections,
            timestamp=time.time(),
            scan_time_ms=(time.time() - start) * 1000,
            frame_width=w,
            frame_height=h
        )


if __name__ == "__main__":
    # Simple test
    print("Testing OCR Engine...")
    engine = OCREngine(confidence_threshold=0.6, use_gpu=False)
    
    # Try to grab a frame from webcam
    cap = cv2.VideoCapture(0)
    ret, test_frame = cap.read()
    cap.release()
    
    if ret:
        result = engine.scan_frame(test_frame)
        print(f"\nScan took: {result.scan_time_ms:.0f}ms")
        print(f"Found {len(result.detections)} text regions\n")
        print("=== Detected Text ===")
        print(result.get_text_with_positions())
        print(f"\n=== Combined for Gemini ===")
        print(result.get_all_text())
    else:
        print("Couldn't access webcam for test.")
