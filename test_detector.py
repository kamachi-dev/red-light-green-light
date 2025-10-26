"""
Tests for the detection module.
"""

import unittest
from detector import Detector, DetectionConfig, DetectionType, DetectionResult


class TestThresholdDetection(unittest.TestCase):
    """Test threshold-based detection."""
    
    def test_threshold_exceeded(self):
        """Test detection when threshold is exceeded."""
        config = DetectionConfig(
            detection_type=DetectionType.THRESHOLD,
            threshold=10.0
        )
        detector = Detector(config)
        
        # Add values below threshold
        result = detector.add_data_point(5.0)
        self.assertFalse(result.detected)
        
        # Add value above threshold
        result = detector.add_data_point(15.0)
        self.assertTrue(result.detected)
        self.assertGreater(result.confidence, 0)
    
    def test_threshold_not_exceeded(self):
        """Test when threshold is not exceeded."""
        config = DetectionConfig(
            detection_type=DetectionType.THRESHOLD,
            threshold=10.0
        )
        detector = Detector(config)
        
        result = detector.add_data_point(5.0)
        self.assertFalse(result.detected)
        
        result = detector.add_data_point(8.0)
        self.assertFalse(result.detected)
    
    def test_threshold_exact_match(self):
        """Test detection when value exactly matches threshold."""
        config = DetectionConfig(
            detection_type=DetectionType.THRESHOLD,
            threshold=10.0
        )
        detector = Detector(config)
        
        result = detector.add_data_point(10.0)
        self.assertTrue(result.detected)


class TestPatternDetection(unittest.TestCase):
    """Test pattern detection."""
    
    def test_increasing_pattern(self):
        """Test detection of increasing pattern."""
        config = DetectionConfig(
            detection_type=DetectionType.PATTERN,
            window_size=5
        )
        detector = Detector(config)
        
        # Add increasing values
        for value in [1, 2, 3, 4, 5]:
            result = detector.add_data_point(value)
        
        self.assertTrue(result.detected)
        self.assertEqual(result.metadata["pattern"], "increasing")
    
    def test_decreasing_pattern(self):
        """Test detection of decreasing pattern."""
        config = DetectionConfig(
            detection_type=DetectionType.PATTERN,
            window_size=5
        )
        detector = Detector(config)
        
        # Add decreasing values
        for value in [5, 4, 3, 2, 1]:
            result = detector.add_data_point(value)
        
        self.assertTrue(result.detected)
        self.assertEqual(result.metadata["pattern"], "decreasing")
    
    def test_no_pattern(self):
        """Test when no pattern exists."""
        config = DetectionConfig(
            detection_type=DetectionType.PATTERN,
            window_size=5
        )
        detector = Detector(config)
        
        # Add random values
        for value in [1, 3, 2, 5, 4]:
            result = detector.add_data_point(value)
        
        self.assertFalse(result.detected)
        self.assertEqual(result.metadata["pattern"], "none")


class TestAnomalyDetection(unittest.TestCase):
    """Test anomaly detection."""
    
    def test_anomaly_detected(self):
        """Test detection of anomalous values."""
        config = DetectionConfig(
            detection_type=DetectionType.ANOMALY,
            sensitivity=2.0,
            window_size=10
        )
        detector = Detector(config)
        
        # Add normal values around 10
        for value in [10, 11, 9, 10, 11, 10, 9, 11]:
            detector.add_data_point(value)
        
        # Add anomalous value
        result = detector.add_data_point(100)
        self.assertTrue(result.detected)
        self.assertGreater(result.confidence, 0)
    
    def test_no_anomaly(self):
        """Test when values are within normal range."""
        config = DetectionConfig(
            detection_type=DetectionType.ANOMALY,
            sensitivity=2.0,
            window_size=10
        )
        detector = Detector(config)
        
        # Add values with small variance
        for value in [10, 11, 9, 10, 11, 10, 9, 11, 10]:
            result = detector.add_data_point(value)
        
        self.assertFalse(result.detected)


class TestChangeDetection(unittest.TestCase):
    """Test change detection."""
    
    def test_significant_change(self):
        """Test detection of significant change."""
        config = DetectionConfig(
            detection_type=DetectionType.CHANGE,
            sensitivity=0.5  # 50% change threshold
        )
        detector = Detector(config)
        
        detector.add_data_point(10.0)
        result = detector.add_data_point(20.0)  # 100% change
        
        self.assertTrue(result.detected)
        self.assertGreater(result.confidence, 0)
    
    def test_insignificant_change(self):
        """Test when change is not significant."""
        config = DetectionConfig(
            detection_type=DetectionType.CHANGE,
            sensitivity=0.5  # 50% change threshold
        )
        detector = Detector(config)
        
        detector.add_data_point(10.0)
        result = detector.add_data_point(11.0)  # 10% change
        
        self.assertFalse(result.detected)


class TestDetectorFeatures(unittest.TestCase):
    """Test general detector features."""
    
    def test_window_size_maintenance(self):
        """Test that buffer maintains window size."""
        config = DetectionConfig(
            detection_type=DetectionType.THRESHOLD,
            threshold=10.0,
            window_size=3
        )
        detector = Detector(config)
        
        for i in range(5):
            detector.add_data_point(i)
        
        self.assertEqual(len(detector.data_buffer), 3)
        self.assertEqual(detector.data_buffer, [2, 3, 4])
    
    def test_reset(self):
        """Test detector reset functionality."""
        config = DetectionConfig(
            detection_type=DetectionType.THRESHOLD,
            threshold=10.0
        )
        detector = Detector(config)
        
        detector.add_data_point(5.0)
        detector.add_data_point(15.0)
        
        detector.reset()
        
        self.assertEqual(len(detector.data_buffer), 0)
        self.assertEqual(len(detector.detection_history), 0)
    
    def test_statistics(self):
        """Test statistics generation."""
        config = DetectionConfig(
            detection_type=DetectionType.THRESHOLD,
            threshold=10.0
        )
        detector = Detector(config)
        
        detector.add_data_point(5.0)   # Not detected
        detector.add_data_point(15.0)  # Detected
        detector.add_data_point(20.0)  # Detected
        
        stats = detector.get_statistics()
        
        self.assertEqual(stats["total_checks"], 3)
        self.assertEqual(stats["positive_detections"], 2)
        self.assertAlmostEqual(stats["detection_rate"], 2/3)
        self.assertGreater(stats["average_confidence"], 0)
    
    def test_callback_execution(self):
        """Test that callback is executed on detection."""
        callback_called = []
        
        def callback(result: DetectionResult):
            callback_called.append(result)
        
        config = DetectionConfig(
            detection_type=DetectionType.THRESHOLD,
            threshold=10.0,
            callback=callback
        )
        detector = Detector(config)
        
        detector.add_data_point(5.0)   # Should not trigger callback
        detector.add_data_point(15.0)  # Should trigger callback
        
        self.assertEqual(len(callback_called), 1)
        self.assertTrue(callback_called[0].detected)
    
    def test_negative_threshold(self):
        """Test threshold detection with negative threshold."""
        config = DetectionConfig(
            detection_type=DetectionType.THRESHOLD,
            threshold=-10.0
        )
        detector = Detector(config)
        
        result = detector.add_data_point(-5.0)
        self.assertTrue(result.detected)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        result = detector.add_data_point(-15.0)
        self.assertFalse(result.detected)
    
    def test_change_detection_from_zero(self):
        """Test change detection when previous value is zero."""
        config = DetectionConfig(
            detection_type=DetectionType.CHANGE,
            sensitivity=0.5
        )
        detector = Detector(config)
        
        detector.add_data_point(0.0)
        result = detector.add_data_point(1.0)
        
        # Should handle division by zero gracefully
        self.assertTrue(result.detected)
        self.assertIsNotNone(result.confidence)
    
    def test_anomaly_detection_low_variance(self):
        """Test anomaly detection with very low variance data."""
        config = DetectionConfig(
            detection_type=DetectionType.ANOMALY,
            sensitivity=2.0,
            window_size=5
        )
        detector = Detector(config)
        
        # Add identical values (zero variance)
        for _ in range(4):
            detector.add_data_point(10.0)
        
        # Small deviation should still work without division errors
        result = detector.add_data_point(10.1)
        
        # Confidence should be bounded
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
