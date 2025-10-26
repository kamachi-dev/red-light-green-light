"""
Example usage of the detection module.

This script demonstrates various detection capabilities.
"""

from detector import Detector, DetectionConfig, DetectionType


def example_threshold_detection():
    """Example of threshold-based detection."""
    print("=== Threshold Detection Example ===")
    
    config = DetectionConfig(
        detection_type=DetectionType.THRESHOLD,
        threshold=10.0
    )
    detector = Detector(config)
    
    test_values = [5, 7, 9, 12, 15, 8, 11, 6]
    
    for value in test_values:
        result = detector.add_data_point(value)
        status = "DETECTED" if result.detected else "normal"
        print(f"Value: {value:5.1f} -> {status:10s} (confidence: {result.confidence:.2f})")
    
    print(f"\nStatistics: {detector.get_statistics()}\n")


def example_pattern_detection():
    """Example of pattern detection."""
    print("=== Pattern Detection Example ===")
    
    config = DetectionConfig(
        detection_type=DetectionType.PATTERN,
        window_size=5
    )
    detector = Detector(config)
    
    # Test increasing pattern
    print("Increasing sequence:")
    for value in [1, 2, 3, 4, 5]:
        result = detector.add_data_point(value)
        print(f"Value: {value} -> Pattern: {result.metadata.get('pattern', 'N/A')}")
    
    detector.reset()
    
    # Test decreasing pattern
    print("\nDecreasing sequence:")
    for value in [10, 8, 6, 4, 2]:
        result = detector.add_data_point(value)
        print(f"Value: {value} -> Pattern: {result.metadata.get('pattern', 'N/A')}")
    
    print()


def example_anomaly_detection():
    """Example of anomaly detection."""
    print("=== Anomaly Detection Example ===")
    
    config = DetectionConfig(
        detection_type=DetectionType.ANOMALY,
        sensitivity=2.0,
        window_size=10
    )
    detector = Detector(config)
    
    # Add normal values
    normal_values = [10, 11, 9, 10, 11, 10, 9, 11, 10, 12]
    anomalous_value = 50
    
    print("Normal values:")
    for value in normal_values:
        result = detector.add_data_point(value)
        print(f"Value: {value:5.1f} -> Detected: {result.detected}")
    
    print("\nAnomalous value:")
    result = detector.add_data_point(anomalous_value)
    print(f"Value: {anomalous_value:5.1f} -> Detected: {result.detected}")
    print(f"Metadata: {result.metadata}\n")


def example_change_detection():
    """Example of change detection."""
    print("=== Change Detection Example ===")
    
    config = DetectionConfig(
        detection_type=DetectionType.CHANGE,
        sensitivity=0.3  # 30% change threshold
    )
    detector = Detector(config)
    
    test_values = [10, 11, 10.5, 15, 14.8, 20]
    
    for i, value in enumerate(test_values):
        result = detector.add_data_point(value)
        if i > 0:
            prev = test_values[i-1]
            change = abs(value - prev) / prev * 100
            status = "SIGNIFICANT" if result.detected else "normal"
            print(f"Value: {value:5.1f} (change: {change:5.1f}%) -> {status}")
        else:
            print(f"Value: {value:5.1f} (first value)")
    
    print()


def example_with_callback():
    """Example using callback function."""
    print("=== Detection with Callback Example ===")
    
    detections = []
    
    def on_detection(result):
        """Callback function executed on detection."""
        detections.append(result)
        print(f"  ⚠️  ALERT: Detection occurred! Confidence: {result.confidence:.2f}")
    
    config = DetectionConfig(
        detection_type=DetectionType.THRESHOLD,
        threshold=15.0,
        callback=on_detection
    )
    detector = Detector(config)
    
    test_values = [10, 12, 18, 14, 20, 13, 16]
    
    for value in test_values:
        print(f"Adding value: {value}")
        detector.add_data_point(value)
    
    print(f"\nTotal detections via callback: {len(detections)}\n")


if __name__ == "__main__":
    example_threshold_detection()
    example_pattern_detection()
    example_anomaly_detection()
    example_change_detection()
    example_with_callback()
    
    print("=== All Examples Complete ===")
