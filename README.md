# red-light-green-light

## Detection Module

A comprehensive data analysis detection system supporting multiple detection algorithms.

### Features

- **Threshold Detection**: Detect when values exceed a specified threshold
- **Pattern Detection**: Identify increasing or decreasing trends in data
- **Anomaly Detection**: Statistical detection of outliers using mean and standard deviation
- **Change Detection**: Detect significant changes between consecutive values

### Installation

No external dependencies required. The module uses only Python standard library.

```bash
python3 -m pip install -r requirements.txt
```

### Usage

#### Basic Threshold Detection

```python
from detector import Detector, DetectionConfig, DetectionType

config = DetectionConfig(
    detection_type=DetectionType.THRESHOLD,
    threshold=10.0
)
detector = Detector(config)

result = detector.add_data_point(15.0)
if result.detected:
    print(f"Threshold exceeded! Confidence: {result.confidence}")
```

#### Pattern Detection

```python
config = DetectionConfig(
    detection_type=DetectionType.PATTERN,
    window_size=5
)
detector = Detector(config)

for value in [1, 2, 3, 4, 5]:
    result = detector.add_data_point(value)

print(f"Pattern detected: {result.metadata['pattern']}")
```

#### Anomaly Detection

```python
config = DetectionConfig(
    detection_type=DetectionType.ANOMALY,
    sensitivity=2.0,
    window_size=10
)
detector = Detector(config)

# Add normal values
for value in [10, 11, 9, 10, 11]:
    detector.add_data_point(value)

# Add anomalous value
result = detector.add_data_point(100)
if result.detected:
    print("Anomaly detected!")
```

#### Using Callbacks

```python
def on_detection(result):
    print(f"Alert! Detection occurred at {result.timestamp}")

config = DetectionConfig(
    detection_type=DetectionType.THRESHOLD,
    threshold=10.0,
    callback=on_detection
)
detector = Detector(config)
```

### Running Tests

```bash
python3 -m unittest test_detector.py
```

### Running Examples

```bash
python3 example.py
```

### API Reference

#### DetectionType

- `THRESHOLD`: Threshold-based detection
- `PATTERN`: Pattern recognition (increasing/decreasing)
- `ANOMALY`: Statistical anomaly detection
- `CHANGE`: Change detection between consecutive values

#### DetectionConfig

Configuration parameters for the detector:

- `detection_type`: Type of detection algorithm to use
- `threshold`: Threshold value (for THRESHOLD type)
- `window_size`: Size of the data buffer (default: 10)
- `sensitivity`: Sensitivity parameter (default: 1.0)
- `callback`: Optional callback function executed on detection

#### Detector Methods

- `add_data_point(value)`: Add a data point and perform detection
- `reset()`: Reset the detector state
- `get_statistics()`: Get detection statistics

### License

MIT License