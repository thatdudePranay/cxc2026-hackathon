# Collision Detection System - Documentation

## Overview

The Collision Detection System provides advanced trajectory monitoring and collision prediction for moving objects detected by the vision system. It tracks objects across multiple frames, analyzes their movement patterns, and predicts potential collisions before they occur.

## Key Features

### 1. **Multi-Object Trajectory Tracking**
- Assigns unique IDs to detected objects
- Tracks position history across up to 30 frames (~1 second at 30fps)
- Maintains tracking even through brief occlusions
- Automatically removes lost tracks after 15 frames

### 2. **Kinematic Analysis**
- **Velocity Calculation**: Computes velocity vectors in 3D space (x, y, depth)
- **Acceleration Estimation**: Tracks changes in velocity over time
- **Movement Classification**: Distinguishes between moving and stationary objects

### 3. **Trajectory Prediction**
- Predicts future positions using kinematic equations
- Uses position, velocity, and acceleration for accurate predictions
- Configurable lookahead time (default: 3 seconds)

### 4. **Collision Detection**
- **Time-to-Collision (TTC)**: Calculates when approaching objects will reach danger zone
- **Approach Detection**: Identifies objects moving toward the user
- **Crossing Detection**: Warns about objects crossing the path laterally
- **Multi-Priority Alerts**: Critical, Warning, and Info levels based on urgency

### 5. **Direction Analysis**
Classifies movement into categories:
- `approaching`: Object moving directly toward user
- `approaching_left`/`approaching_right`: Approaching from angles
- `receding`: Object moving away
- `crossing_left`/`crossing_right`: Lateral movement
- `stationary`: No significant movement

## Classes

### `TrajectoryTracker`

Tracks a single object's movement over time.

**Initialization:**
```python
tracker = TrajectoryTracker(
    track_id=0,
    initial_position=(x, y, distance),
    initial_timestamp=time.time(),
    object_info={'class_name': 'car', 'class_id': 2, 'is_critical': True}
)
```

**Key Methods:**
- `update(position, timestamp)`: Add new position to trajectory
- `predict_position(time_delta)`: Predict position N seconds ahead
- `get_trajectory_direction()`: Get movement direction classification
- `get_current_position()`: Get most recent position

**Attributes:**
- `position_history`: Deque of (x, y, distance, timestamp) tuples
- `velocity`: 3D velocity vector [vx, vy, vz]
- `acceleration`: 3D acceleration vector
- `is_moving`: Boolean indicating if object has significant motion

### `CollisionDetector`

Main collision detection system that manages multiple trackers.

**Initialization:**
```python
detector = CollisionDetector()
```

**Key Methods:**

#### `update(detections, timestamp=None)`
Updates all trackers with new detections and returns collision warnings.

**Parameters:**
- `detections`: List of detection dicts from VisionSystem
- `timestamp`: Current time (optional, uses time.time() if None)

**Returns:** List of collision warning dicts with structure:
```python
{
    'priority': 0,  # 0=Critical, 1=Warning, 2=Info
    'message': 'COLLISION ALERT! Car approaching rapidly in 1.2s at 1.5m',
    'object': {'class_name': 'car', 'class_id': 2, 'is_critical': True},
    'direction': 'approaching',
    'ttc': 1.2,  # Time-to-collision in seconds
    'current_distance': 1.5,  # Current distance in meters
    'predicted_distance': 0.3,  # Predicted distance
    'track_id': 0,
    'velocity': [10.5, 2.3, -1.2],
    'position': (320, 240, 1.5)
}
```

#### `get_tracked_objects()`
Returns list of all currently tracked objects with their states.

#### `draw_trajectories(frame, show_predictions=True)`
Visualizes trajectories on the video frame:
- **Magenta lines**: Object's historical path
- **Yellow arrows**: Current velocity vector
- **Yellow circles**: Predicted future positions
- **Track IDs**: Displayed near objects

## Configuration Parameters

Set these in `config.py`:

```python
# Collision detection settings
COLLISION_LOOKAHEAD_TIME = 3.0          # Seconds ahead to predict
COLLISION_DISTANCE_THRESHOLD = 2.0      # Collision warning distance (meters)
COLLISION_LATERAL_THRESHOLD = 1.5       # Lateral threshold (meters)
MAX_TRACKING_LOST_FRAMES = 15           # Frames before removing tracker
TRAJECTORY_MATCHING_THRESHOLD = 100     # Pixels for matching objects
TRAJECTORY_HISTORY_LENGTH = 30          # Frames to keep in history
MIN_MOVEMENT_SPEED = 5.0                # Pixels/sec to consider moving
MIN_DEPTH_SPEED = 0.1                   # m/s to consider approach/recede
```

## Usage Examples

### Basic Integration

```python
from core.vision import VisionSystem
from core.collision_detection import CollisionDetector
import cv2

# Initialize systems
vision = VisionSystem()
collision_detector = CollisionDetector()

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get detections from vision system
    detections = vision.detect_objects(frame)
    
    # Update collision detector
    collision_warnings = collision_detector.update(detections)
    
    # Process warnings
    for warning in collision_warnings:
        if warning['priority'] == 0:  # Critical
            print(f"🚨 CRITICAL: {warning['message']}")
            # Trigger alert sound, vibration, etc.
        elif warning['priority'] == 1:  # Warning
            print(f"⚠️ WARNING: {warning['message']}")
    
    # Draw visualizations
    annotated_frame = vision.draw_detections(frame, detections)
    annotated_frame = collision_detector.draw_trajectories(annotated_frame)
    
    cv2.imshow('Collision Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Advanced: Custom Alert Handler

```python
class CollisionAlertHandler:
    def __init__(self, collision_detector):
        self.detector = collision_detector
        self.alert_history = []
        
    def process_frame(self, detections):
        warnings = self.detector.update(detections)
        
        for warning in warnings:
            ttc = warning.get('ttc')
            
            if warning['priority'] == 0 and ttc and ttc < 2.0:
                # Critical collision imminent
                self.trigger_emergency_alert(warning)
            elif warning['priority'] == 1:
                # Regular warning
                self.trigger_warning_alert(warning)
            
            self.alert_history.append(warning)
    
    def trigger_emergency_alert(self, warning):
        # Play urgent sound, haptic feedback, etc.
        print(f"🚨 EMERGENCY: {warning['message']}")
    
    def trigger_warning_alert(self, warning):
        # Play warning sound
        print(f"⚠️ {warning['message']}")
```

### Accessing Tracking Information

```python
# Get all tracked objects
tracked_objects = collision_detector.get_tracked_objects()

for obj in tracked_objects:
    print(f"Track ID: {obj['track_id']}")
    print(f"Object: {obj['object_name']}")
    print(f"Position: {obj['position']}")
    print(f"Velocity: {obj['velocity']}")
    print(f"Direction: {obj['direction']}")
    print(f"Moving: {obj['is_moving']}")
    print(f"Frames tracked: {obj['frames_tracked']}")
    print()
```

## Algorithm Details

### Object Matching

The system uses spatial proximity to match detections across frames:
1. For each new detection, calculate distance to all existing trackers
2. Match to closest tracker within `TRAJECTORY_MATCHING_THRESHOLD` pixels
3. If no match found, create new tracker
4. Unmatched trackers increment lost frame counter

### Velocity Calculation

Velocity is computed using finite differences:
```
v = (position_t2 - position_t1) / (t2 - t1)
```

Components:
- `vx`: Horizontal velocity (pixels/sec)
- `vy`: Vertical velocity (pixels/sec)
- `vz`: Depth velocity (meters/sec)

### Acceleration Estimation

Acceleration is the rate of change of velocity:
```
a = (velocity_t2 - velocity_t1) / (t2 - t1)
```

### Position Prediction

Future position uses kinematic equations:
```
position_future = position_current + velocity * t + 0.5 * acceleration * t²
```

Where `t` is the time delta (lookahead time).

### Time-to-Collision (TTC)

For approaching objects:
```
TTC = -distance / velocity_z
```

(Negative because velocity_z is negative when approaching)

### Collision Risk Assessment

**Critical Priority (Red):**
- TTC < 1.5 seconds AND distance < 2.0m
- Fast-moving vehicles (cars, trucks, buses) approaching

**Warning Priority (Orange):**
- TTC < 2.5 seconds AND distance < 4.0m
- Objects crossing path within danger zone
- Person approaching closely

**Info Priority (Green):**
- TTC < 3.0 seconds
- Objects approaching but not yet dangerous

## Performance Considerations

### Computational Complexity
- **Per-frame cost**: O(N*M) where N = detections, M = trackers
- **Typical performance**: Can handle 20+ objects at 30fps
- **Memory usage**: ~30KB per tracked object (30 frame history)

### Optimization Tips
1. **Reduce history length** if memory is constrained
2. **Increase matching threshold** for faster movement scenarios
3. **Adjust lookahead time** based on application needs
4. **Filter non-critical objects** before processing

## Testing

### Standalone Test
Run the module directly to test with webcam:

```bash
cd core
python collision_detection.py
```

**Controls:**
- `q`: Quit
- `t`: Toggle trajectory visualization
- `s`: Save screenshot

### Unit Testing
```python
import unittest
from core.collision_detection import TrajectoryTracker, CollisionDetector
import time

class TestCollisionDetection(unittest.TestCase):
    def test_trajectory_tracking(self):
        tracker = TrajectoryTracker(
            0, (100, 100, 5.0), time.time(),
            {'class_name': 'car', 'class_id': 2, 'is_critical': True}
        )
        
        # Simulate approaching car
        for i in range(10):
            t = time.time()
            tracker.update((110 + i*5, 100, 5.0 - i*0.2), t)
            time.sleep(0.033)  # ~30fps
        
        self.assertTrue(tracker.is_moving)
        self.assertEqual(tracker.get_trajectory_direction(), 'approaching')
```

## Troubleshooting

### Objects Not Being Tracked
- Check `TRAJECTORY_MATCHING_THRESHOLD` - may be too small
- Verify detection confidence is consistent
- Ensure objects are marked as `is_critical`

### False Collision Warnings
- Increase `COLLISION_LOOKAHEAD_TIME` for earlier warnings
- Adjust `DANGER_DISTANCE` and `WARNING_DISTANCE` thresholds
- Check if velocity calculation is accurate

### Trackers Lost Too Quickly
- Increase `MAX_TRACKING_LOST_FRAMES`
- Improve detection consistency in vision system
- Check frame rate stability

### Inaccurate Predictions
- Ensure consistent frame timing
- Verify distance estimation accuracy
- Consider using more frames for velocity calculation

## Future Enhancements

Potential improvements:
1. **Kalman filtering** for smoother tracking
2. **Multi-object collision prediction** (object-to-object)
3. **Path intersection analysis** for crossing trajectories
4. **Machine learning** for behavior prediction
5. **Sensor fusion** with other inputs (GPS, IMU)
6. **Historical pattern learning** for recurring scenarios

## License & Credits

Part of the CXC2026 Hackathon project for visual assistance technology.
