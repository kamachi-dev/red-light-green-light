"""
Detection module for data analysis.

This module provides various detection algorithms for analyzing data patterns,
anomalies, and threshold-based events.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time


class DetectionType(Enum):
    """Types of detection algorithms available."""
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    CHANGE = "change"


@dataclass
class DetectionConfig:
    """Configuration for detection algorithms."""
    detection_type: DetectionType
    threshold: Optional[float] = None
    window_size: int = 10
    sensitivity: float = 1.0
    callback: Optional[Callable] = None


@dataclass
class DetectionResult:
    """Result of a detection operation."""
    detected: bool
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]


class Detector:
    """
    Main detector class for performing various types of detection.
    
    This class provides a unified interface for different detection algorithms
    including threshold-based detection, pattern matching, and anomaly detection.
    """
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize the detector with the given configuration.
        
        Args:
            config: DetectionConfig object with algorithm parameters
        """
        self.config = config
        self.data_buffer: List[float] = []
        self.detection_history: List[DetectionResult] = []
        self._last_detection_time = 0
    
    def add_data_point(self, value: float) -> DetectionResult:
        """
        Add a data point and perform detection.
        
        Args:
            value: The data point to analyze
            
        Returns:
            DetectionResult indicating whether detection occurred
        """
        self.data_buffer.append(value)
        
        # Maintain window size
        if len(self.data_buffer) > self.config.window_size:
            self.data_buffer.pop(0)
        
        # Perform detection based on type
        result = self._detect()
        
        # Store result in history
        self.detection_history.append(result)
        
        # Execute callback if provided and detection occurred
        if result.detected and self.config.callback:
            self.config.callback(result)
        
        return result
    
    def _detect(self) -> DetectionResult:
        """
        Internal method to perform detection based on configured type.
        
        Returns:
            DetectionResult with detection outcome
        """
        timestamp = time.time()
        
        if self.config.detection_type == DetectionType.THRESHOLD:
            return self._threshold_detection(timestamp)
        elif self.config.detection_type == DetectionType.PATTERN:
            return self._pattern_detection(timestamp)
        elif self.config.detection_type == DetectionType.ANOMALY:
            return self._anomaly_detection(timestamp)
        elif self.config.detection_type == DetectionType.CHANGE:
            return self._change_detection(timestamp)
        else:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=timestamp,
                metadata={"error": "Unknown detection type"}
            )
    
    def _threshold_detection(self, timestamp: float) -> DetectionResult:
        """
        Detect when values exceed a threshold.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            DetectionResult indicating if threshold was exceeded
        """
        if not self.data_buffer or self.config.threshold is None:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=timestamp,
                metadata={"reason": "insufficient_data"}
            )
        
        current_value = self.data_buffer[-1]
        detected = current_value >= self.config.threshold
        
        confidence = min(1.0, abs(current_value - self.config.threshold) / abs(self.config.threshold)) if self.config.threshold != 0 else 0.0
        
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            timestamp=timestamp,
            metadata={
                "current_value": current_value,
                "threshold": self.config.threshold,
                "buffer_size": len(self.data_buffer)
            }
        )
    
    def _pattern_detection(self, timestamp: float) -> DetectionResult:
        """
        Detect patterns in the data buffer.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            DetectionResult indicating if pattern was found
        """
        if len(self.data_buffer) < 3:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=timestamp,
                metadata={"reason": "insufficient_data"}
            )
        
        # Detect increasing pattern
        is_increasing = all(
            self.data_buffer[i] < self.data_buffer[i + 1]
            for i in range(len(self.data_buffer) - 1)
        )
        
        # Detect decreasing pattern
        is_decreasing = all(
            self.data_buffer[i] > self.data_buffer[i + 1]
            for i in range(len(self.data_buffer) - 1)
        )
        
        detected = is_increasing or is_decreasing
        confidence = 1.0 if detected else 0.0
        
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            timestamp=timestamp,
            metadata={
                "pattern": "increasing" if is_increasing else "decreasing" if is_decreasing else "none",
                "buffer_size": len(self.data_buffer)
            }
        )
    
    def _anomaly_detection(self, timestamp: float) -> DetectionResult:
        """
        Detect anomalies using statistical methods.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            DetectionResult indicating if anomaly was detected
        """
        if len(self.data_buffer) < 3:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=timestamp,
                metadata={"reason": "insufficient_data"}
            )
        
        # Calculate mean and standard deviation
        mean = sum(self.data_buffer) / len(self.data_buffer)
        variance = sum((x - mean) ** 2 for x in self.data_buffer) / len(self.data_buffer)
        std_dev = variance ** 0.5
        
        current_value = self.data_buffer[-1]
        
        # Detect if current value is beyond sensitivity * std_dev from mean
        threshold = self.config.sensitivity * std_dev
        detected = abs(current_value - mean) > threshold
        
        confidence = min(1.0, abs(current_value - mean) / max(threshold, 1.0)) if threshold != 0 else 0.0
        
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            timestamp=timestamp,
            metadata={
                "current_value": current_value,
                "mean": mean,
                "std_dev": std_dev,
                "deviation": abs(current_value - mean)
            }
        )
    
    def _change_detection(self, timestamp: float) -> DetectionResult:
        """
        Detect significant changes between consecutive values.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            DetectionResult indicating if significant change was detected
        """
        if len(self.data_buffer) < 2:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=timestamp,
                metadata={"reason": "insufficient_data"}
            )
        
        prev_value = self.data_buffer[-2]
        current_value = self.data_buffer[-1]
        
        # Calculate relative change
        # When previous value is zero, use absolute change as relative change cannot be computed
        if prev_value != 0:
            relative_change = abs((current_value - prev_value) / prev_value)
        else:
            # Special case: treat absolute change as relative when prev_value is 0
            relative_change = abs(current_value)
        
        # Detect if change exceeds sensitivity threshold
        detected = relative_change > self.config.sensitivity
        confidence = min(1.0, relative_change / self.config.sensitivity) if self.config.sensitivity != 0 else 0.0
        
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            timestamp=timestamp,
            metadata={
                "previous_value": prev_value,
                "current_value": current_value,
                "relative_change": relative_change
            }
        )
    
    def reset(self):
        """Reset the detector state."""
        self.data_buffer.clear()
        self.detection_history.clear()
        self._last_detection_time = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detection history.
        
        Returns:
            Dictionary containing detection statistics
        """
        total_detections = len(self.detection_history)
        positive_detections = sum(1 for r in self.detection_history if r.detected)
        
        avg_confidence = 0.0
        if self.detection_history:
            avg_confidence = sum(r.confidence for r in self.detection_history) / len(self.detection_history)
        
        return {
            "total_checks": total_detections,
            "positive_detections": positive_detections,
            "detection_rate": positive_detections / total_detections if total_detections > 0 else 0.0,
            "average_confidence": avg_confidence,
            "buffer_size": len(self.data_buffer)
        }
