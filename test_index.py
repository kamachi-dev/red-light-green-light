#!/usr/bin/env python3
"""
Test script for face detection crossing system.
Tests the logic without requiring a webcam.
"""

import cv2
import numpy as np
import time
from datetime import datetime, timedelta
from index import FaceDetectionCrossing

def create_test_frame(width=640, height=480):
    """Create a blank test frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)

def test_initialization():
    """Test initialization of FaceDetectionCrossing."""
    print("Testing initialization...")
    detector = FaceDetectionCrossing(min_faces=2, crossing_duration=10)
    
    assert detector.min_faces == 2, "min_faces not set correctly"
    assert detector.crossing_duration == 10, "crossing_duration not set correctly"
    assert detector.isCrossing == False, "isCrossing should be False initially"
    assert detector.crossing_end_time is None, "crossing_end_time should be None initially"
    
    print("✓ Initialization test passed")

def test_crossing_logic():
    """Test crossing status logic."""
    print("\nTesting crossing logic...")
    detector = FaceDetectionCrossing(min_faces=2, crossing_duration=5)
    
    # Test 1: Not enough faces, should stay False
    detector.update_crossing_status(1)
    assert detector.isCrossing == False, "isCrossing should be False with insufficient faces"
    print("✓ Test 1 passed: isCrossing stays False with insufficient faces")
    
    # Test 2: Enough faces, should become True
    detector.update_crossing_status(2)
    assert detector.isCrossing == True, "isCrossing should be True when min_faces met"
    assert detector.crossing_end_time is not None, "crossing_end_time should be set"
    print("✓ Test 2 passed: isCrossing becomes True when threshold met")
    
    # Test 3: Faces present, should stay True and timer extends
    time.sleep(1)
    first_end_time = detector.crossing_end_time
    detector.update_crossing_status(3)
    assert detector.isCrossing == True, "isCrossing should stay True with faces present"
    assert detector.crossing_end_time > first_end_time, "Timer should extend with faces present"
    print("✓ Test 3 passed: Timer extends while faces present")
    
    # Test 4: Faces gone but timer not expired, should stay True
    detector.update_crossing_status(0)
    assert detector.isCrossing == True, "isCrossing should stay True while timer active"
    print("✓ Test 4 passed: isCrossing stays True while timer active")
    
    # Test 5: Wait for timer to expire
    print("  Waiting for timer to expire (5 seconds)...")
    time.sleep(6)
    detector.update_crossing_status(0)
    assert detector.isCrossing == False, "isCrossing should be False after timer expires"
    assert detector.crossing_end_time is None, "crossing_end_time should be None after reset"
    print("✓ Test 5 passed: isCrossing becomes False after timer expires")

def test_face_cascade_loading():
    """Test that Haar Cascade loads correctly."""
    print("\nTesting Haar Cascade loading...")
    detector = FaceDetectionCrossing()
    
    assert detector.face_cascade is not None, "Face cascade should be loaded"
    assert not detector.face_cascade.empty(), "Face cascade should not be empty"
    print("✓ Haar Cascade loaded successfully")

def test_detect_faces_on_blank():
    """Test face detection on a blank frame (should detect 0 faces)."""
    print("\nTesting face detection on blank frame...")
    detector = FaceDetectionCrossing()
    frame = create_test_frame()
    
    faces = detector.detect_faces(frame)
    assert len(faces) == 0, "Should detect 0 faces on blank frame"
    print("✓ Blank frame detection test passed")

def main():
    """Run all tests."""
    print("="*50)
    print("Face Detection Crossing System - Test Suite")
    print("="*50)
    
    try:
        test_initialization()
        test_face_cascade_loading()
        test_crossing_logic()
        test_detect_faces_on_blank()
        
        print("\n" + "="*50)
        print("All tests passed! ✓")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
