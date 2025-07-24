import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Deque
from collections import deque
import time
import math
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

@dataclass
class SimplifiedHandLandmarks:
    """Simplified hand landmarks with 7 key points"""
    wrist: Tuple[float, float]  # Wrist position
    thumb_tip: Tuple[float, float]  # Thumb tip
    index_base: Tuple[float, float]  # Base of index finger
    index_tip: Tuple[float, float]  # Index fingertip
    middle_tip: Tuple[float, float]  # Middle fingertip
    ring_tip: Tuple[float, float]  # Ring fingertip
    pinky_tip: Tuple[float, float]  # Pinky fingertip

@dataclass
class BodyPartSpeed:
    """Tracks speed of body parts"""
    value: float = 0.0
    is_high_speed: bool = False

@dataclass
class PersonData:
    """Data structure to store detection and landmark information for a person"""
    id: int  # Unique identifier for tracking
    box: List[int]  # [x1, y1, x2, y2]
    pose_landmarks: Optional[List[Tuple[float, float]]] = None
    left_hand: Optional[SimplifiedHandLandmarks] = None
    right_hand: Optional[SimplifiedHandLandmarks] = None
    left_gesture: str = "None"
    right_gesture: str = "None"
    
    # Speed tracking
    body_speed: BodyPartSpeed = field(default_factory=BodyPartSpeed)
    left_hand_speed: BodyPartSpeed = field(default_factory=BodyPartSpeed)
    right_hand_speed: BodyPartSpeed = field(default_factory=BodyPartSpeed)
    
    # Previous positions
    prev_center: Optional[Tuple[float, float]] = None
    prev_left_hand: Optional[Tuple[float, float]] = None
    prev_right_hand: Optional[Tuple[float, float]] = None
    
    # Position history (for trajectory analysis)
    center_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=10))
    left_hand_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=10))
    right_hand_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=10))
    
    # Timestamps
    timestamp: float = field(default_factory=time.time)
    prev_timestamp: Optional[float] = None
    
    # Assault detection
    is_assaulter: bool = False
    is_victim: bool = False
    flagged_body_part: str = "None"  # Which body part triggered the alert
    assault_confidence: float = 0.0  # Confidence level for assault detection
    action_label: str = "None"  # Predicted action from the model
    
    # Time tracking for persistent status
    status_start_time: float = 0  # When the assault/victim status started
    
    def update_timestamp(self):
        """Update timestamps for speed calculation"""
        self.prev_timestamp = self.timestamp
        self.timestamp = time.time()
        
    def get_center(self) -> Tuple[float, float]:
        """Calculate center point of the person"""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

class AssaultDetector:
    def __init__(self, model_path, action_csv_dir, min_detection_confidence=0.5, 
                 speed_threshold=100, interaction_threshold=50):
        """
        Initialize detector for person detection with assault detection capabilities
        
        Args:
            model_path (str): Path to the H5 model file
            action_csv_dir (str): Directory containing CSV datasets
            min_detection_confidence (float): Confidence threshold for detection
            speed_threshold (float): Threshold for high-speed movement (pixels/second)
            interaction_threshold (float): Distance threshold for interaction detection (pixels)
        """
        self.conf_threshold = min_detection_confidence
        self.speed_threshold = speed_threshold
        self.interaction_threshold = interaction_threshold
        
        # Load the H5 model
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Get model input shape
        self.input_shape = self.model.input_shape[1:]
        print(f"Model input shape: {self.input_shape}")
        
        # Load action labels from CSV directory
        self.actions_df = self.load_action_datasets(action_csv_dir)
        self.action_labels = self.extract_action_labels()
        print(f"Loaded {len(self.action_labels)} action labels: {self.action_labels}")
        
        # Initialize MediaPipe Holistic for body and hand landmarks
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        
        # Important landmark indices
        self.hand_keypoints = {
            "wrist": 0,
            "thumb_tip": 4,
            "index_base": 5,
            "index_tip": 8,
            "middle_tip": 12,
            "ring_tip": 16,
            "pinky_tip": 20
        }
        
        # Selected body keypoints (subset of 33)
        self.selected_pose_keypoints = [
            0,   # nose
            11,  # left shoulder
            12,  # right shoulder
            13,  # left elbow
            14,  # right elbow
            15,  # left wrist
            16,  # right wrist
            23,  # left hip
            24,  # right hip
            25,  # left knee
            26,  # right knee
            27,  # left ankle
            28   # right ankle
        ]
        
        # Person tracking
        self.next_person_id = 0
        self.tracked_persons = {}  # Dictionary to track persons across frames
        self.person_history = {}   # History of each person's position
        
        # Flags for detected events
        self.assault_detected = False
        self.last_detection_time = 0
        self.alert_duration = 3.0  # How long to show the alert (seconds)
        
        # Status persistence duration (10 seconds)
        self.status_persistence = 10.0
        
        # Store people who were flagged but are no longer in frame
        self.flagged_people = {}
        
        # Frame timing
        self.prev_frame_time = time.time()
        self.curr_frame_time = time.time()
        
        # Assault-related action patterns
        self.assault_patterns = ["punch", "kick", "slap", "aggressive", "attack", "hit", "strike"]
    
    def load_action_datasets(self, csv_dir):
        """Load CSV datasets from the given directory"""
        dataframes = []
        
        if not os.path.exists(csv_dir):
            print(f"Warning: CSV directory {csv_dir} does not exist")
            return pd.DataFrame()
            
        for filename in os.listdir(csv_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(csv_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    dataframes.append(df)
                    print(f"Loaded {filepath}")
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        
        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        else:
            print("No CSV files loaded")
            return pd.DataFrame()
    
    def extract_action_labels(self):
        """Extract unique action labels from the loaded dataframes"""
        if self.actions_df.empty:
            # Default actions if no CSV files are loaded
            return ["neutral", "punch", "kick", "push", "normal"]
            
        # Try to find action labels column
        label_columns = [col for col in self.actions_df.columns if "action" in col.lower() or "label" in col.lower()]
        
        if label_columns:
            label_column = label_columns[0]
            return self.actions_df[label_column].unique().tolist()
        else:
            # Default actions if no label column found
            return ["neutral", "punch", "kick", "push", "normal"]
    
    def _extract_simplified_hand_landmarks(self, hand_landmarks):
        """Extract only the specified key points from hand landmarks"""
        if not hand_landmarks:
            return None
            
        points = {}
        for name, idx in self.hand_keypoints.items():
            landmark = hand_landmarks.landmark[idx]
            points[name] = (landmark.x, landmark.y)
            
        return SimplifiedHandLandmarks(
            wrist=points["wrist"],
            thumb_tip=points["thumb_tip"],
            index_base=points["index_base"],
            index_tip=points["index_tip"],
            middle_tip=points["middle_tip"],
            ring_tip=points["ring_tip"],
            pinky_tip=points["pinky_tip"]
        )
    
    def _detect_open_palm(self, hand: SimplifiedHandLandmarks) -> bool:
        """Check if hand gesture is an open palm using simplified landmarks"""
        if not hand:
            return False
        
        # Check if fingertips are higher than base (for a vertical hand)
        fingers_extended = (
            hand.index_tip[1] < hand.index_base[1] and
            hand.middle_tip[1] < hand.index_base[1] and
            hand.ring_tip[1] < hand.index_base[1] and
            hand.pinky_tip[1] < hand.index_base[1]
        )
        
        # Check thumb position (simplified)
        thumb_extended = (
            abs(hand.thumb_tip[0] - hand.wrist[0]) > 
            abs(hand.index_base[0] - hand.wrist[0])
        )
        
        return fingers_extended and thumb_extended
    
    def _detect_closed_fist(self, hand: SimplifiedHandLandmarks) -> bool:
        """Check if hand gesture is a closed fist using simplified landmarks"""
        if not hand:
            return False
        
        # Check if fingertips are lower than base (fingers curled)
        fingers_bent = (
            hand.index_tip[1] > hand.index_base[1] and
            hand.middle_tip[1] > hand.index_base[1] and
            hand.ring_tip[1] > hand.index_base[1] and
            hand.pinky_tip[1] > hand.index_base[1]
        )
        
        # Check thumb position (simplified)
        thumb_bent = (
            abs(hand.thumb_tip[0] - hand.wrist[0]) < 
            abs(hand.index_base[0] - hand.wrist[0])
        )
        
        return fingers_bent and thumb_bent
    
    def _detect_pointing(self, hand: SimplifiedHandLandmarks) -> bool:
        """Check if hand is pointing (index extended, others closed)"""
        if not hand:
            return False
        
        # Index finger extended
        index_extended = hand.index_tip[1] < hand.index_base[1]
        
        # Other fingers bent
        other_fingers_bent = (
            hand.middle_tip[1] > hand.index_base[1] and
            hand.ring_tip[1] > hand.index_base[1] and
            hand.pinky_tip[1] > hand.index_base[1]
        )
        
        return index_extended and other_fingers_bent
    
    def detect_gesture(self, hand: SimplifiedHandLandmarks) -> str:
        """Identify hand gesture from simplified landmarks"""
        if not hand:
            return "None"
        
        if self._detect_open_palm(hand):
            return "Open Palm"
        elif self._detect_closed_fist(hand):
            return "Closed Fist"
        elif self._detect_pointing(hand):
            return "Pointing"
        
        return "Unknown"
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        if not point1 or not point2:
            return float('inf')
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_speed(self, point1, point2, time_diff):
        """Calculate speed in pixels per second"""
        if not point1 or not point2 or time_diff <= 0:
            return 0.0
        distance = self.calculate_distance(point1, point2)
        return distance / time_diff
    
    def assign_person_id(self, current_boxes, prev_persons):
        """Assign IDs to detected persons based on spatial proximity"""
        if not prev_persons:
            return {i: self.next_person_id + i for i in range(len(current_boxes))}
        
        # Calculate distances between current boxes and previous persons
        distances = {}
        for curr_idx, curr_box in enumerate(current_boxes):
            curr_center = ((curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2)
            
            for prev_id, prev_person in prev_persons.items():
                prev_center = prev_person.get_center()
                distance = self.calculate_distance(curr_center, prev_center)
                distances[(curr_idx, prev_id)] = distance
        
        # Assign IDs based on minimum distance
        used_prev_ids = set()
        assignments = {}
        
        # Sort distances
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        
        for (curr_idx, prev_id), distance in sorted_distances:
            if curr_idx not in assignments and prev_id not in used_prev_ids:
                assignments[curr_idx] = prev_id
                used_prev_ids.add(prev_id)
        
        # Assign new IDs to unmatched current boxes
        for i in range(len(current_boxes)):
            if i not in assignments:
                new_id = self.next_person_id
                self.next_person_id += 1
                assignments[i] = new_id
        
        return assignments
    
    def extract_features(self, pose_landmarks, left_hand, right_hand):
        """
        Extract features from pose and hand landmarks for the model
        This function should be adapted based on how your model expects input
        """
        # Initialize feature vector with zeros
        features = np.zeros(self.input_shape)
        
        # If input shape has 1 dimension (flat array)
        if len(self.input_shape) == 1:
            feature_idx = 0
            
            # Add pose landmarks
            if pose_landmarks:
                for point in pose_landmarks:
                    if feature_idx + 1 < self.input_shape[0]:
                        features[feature_idx] = point[0]  # x
                        features[feature_idx + 1] = point[1]  # y
                        feature_idx += 2
            
            # Add left hand landmarks
            if left_hand:
                for point_name in ["wrist", "thumb_tip", "index_base", "index_tip", 
                                 "middle_tip", "ring_tip", "pinky_tip"]:
                    point = getattr(left_hand, point_name)
                    if feature_idx + 1 < self.input_shape[0]:
                        features[feature_idx] = point[0]  # x
                        features[feature_idx + 1] = point[1]  # y
                        feature_idx += 2
            
            # Add right hand landmarks
            if right_hand:
                for point_name in ["wrist", "thumb_tip", "index_base", "index_tip", 
                                 "middle_tip", "ring_tip", "pinky_tip"]:
                    point = getattr(right_hand, point_name)
                    if feature_idx + 1 < self.input_shape[0]:
                        features[feature_idx] = point[0]  # x
                        features[feature_idx + 1] = point[1]  # y
                        feature_idx += 2
                        
        # If model expects time-sequence data
        elif len(self.input_shape) == 2:
            timesteps, features_per_step = self.input_shape
            features = np.zeros((timesteps, features_per_step))
            
            # For simplicity, we'll just use the current frame's data
            # Fill the last timestep with current data
            feature_idx = 0
            
            # Add pose landmarks to the last timestep
            if pose_landmarks:
                for point in pose_landmarks:
                    if feature_idx + 1 < features_per_step:
                        features[-1, feature_idx] = point[0]  # x
                        features[-1, feature_idx + 1] = point[1]  # y
                        feature_idx += 2
                        
        # If model expects a different format, adjust accordingly
        
        return features
    
    def predict_action(self, pose_landmarks, left_hand, right_hand):
        """
        Use the loaded model to predict action based on landmarks
        """
        # Extract features for the model
        features = self.extract_features(pose_landmarks, left_hand, right_hand)
        
        # Reshape features according to model input shape
        if len(self.input_shape) == 1:
            features = features.reshape(1, -1)
        elif len(self.input_shape) == 2:
            features = features.reshape(1, self.input_shape[0], self.input_shape[1])
        
        # Get prediction
        try:
            prediction = self.model.predict(features, verbose=0)[0]
            
            # Get the predicted class index
            predicted_class_idx = np.argmax(prediction)
            confidence = prediction[predicted_class_idx]
            
            # Get the corresponding action label
            if predicted_class_idx < len(self.action_labels):
                action = self.action_labels[predicted_class_idx]
            else:
                action = "unknown"
                
            return action, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0
    
    def detect_interactions(self, persons):
        """Detect interactions between persons based on proximity and speed"""
        current_time = time.time()
        
        for person_id, person in persons.items():
            # Check if this person has a flagged status that should persist
            if person.is_assaulter or person.is_victim:
                # If this is a newly flagged person, set start time
                if person.status_start_time == 0:
                    person.status_start_time = current_time
                # Check if the status should expire
                elif current_time - person.status_start_time > self.status_persistence:
                    person.is_assaulter = False
                    person.is_victim = False
                    person.flagged_body_part = "None"
                    person.assault_confidence = 0.0
                    person.status_start_time = 0
                # If status is still valid, skip detection for this person
                else:
                    continue
            
            # Skip if not enough speed data
            if not person.prev_left_hand or not person.prev_right_hand or not person.prev_timestamp:
                continue
            
            # Get current hand positions
            left_hand_pos = person.left_hand.wrist if person.left_hand else None
            right_hand_pos = person.right_hand.wrist if person.right_hand else None
            
            # Time difference
            time_diff = person.timestamp - person.prev_timestamp
            if time_diff <= 0:
                continue
            
            # Calculate hand speeds
            left_speed = self.calculate_speed(left_hand_pos, person.prev_left_hand, time_diff) if left_hand_pos else 0
            right_speed = self.calculate_speed(right_hand_pos, person.prev_right_hand, time_diff) if right_hand_pos else 0
            
            # Calculate body speed (based on center point)
            center = person.get_center()
            body_speed = self.calculate_speed(center, person.prev_center, time_diff) if person.prev_center else 0
            
            # Update speed values
            person.left_hand_speed.value = left_speed
            person.right_hand_speed.value = right_speed
            person.body_speed.value = body_speed
            
            # Check for high-speed movements (relative to body movement)
            left_hand_relative_speed = left_speed - body_speed
            right_hand_relative_speed = right_speed - body_speed
            
            # Flag for high speed
            high_speed_threshold = self.speed_threshold
            person.left_hand_speed.is_high_speed = left_hand_relative_speed > high_speed_threshold
            person.right_hand_speed.is_high_speed = right_hand_relative_speed > high_speed_threshold
            
            # Check for potential assaulter based on model prediction
            assault_action_detected = any(pattern in person.action_label.lower() for pattern in self.assault_patterns)
            
            # Combine model prediction with speed detection
            if assault_action_detected or person.left_hand_speed.is_high_speed or person.right_hand_speed.is_high_speed:
                person.is_assaulter = True
                person.flagged_body_part = "Left Hand" if left_hand_relative_speed > right_hand_relative_speed else "Right Hand"
                
                # Calculate assault confidence based on both speed and model prediction
                model_confidence = 0.7  # Default confidence from model if assault action detected
                speed_confidence = max(left_hand_relative_speed, right_hand_relative_speed) / (high_speed_threshold * 2)
                
                # Weight both factors
                if assault_action_detected:
                    person.assault_confidence = 0.7 * model_confidence + 0.3 * speed_confidence
                else:
                    person.assault_confidence = speed_confidence
                    
                person.assault_confidence = min(max(person.assault_confidence, 0.0), 1.0)  # Clamp to [0,1]
                person.status_start_time = current_time  # Start the persistence timer
                
                # Check if any high-speed hand is close to another person (potential victim)
                for other_id, other_person in persons.items():
                    if other_id == person_id:
                        continue
                    
                    other_center = other_person.get_center()
                    
                    # Check left hand proximity
                    if person.left_hand_speed.is_high_speed and left_hand_pos:
                        distance = self.calculate_distance(left_hand_pos, other_center)
                        if distance < self.interaction_threshold:
                            other_person.is_victim = True
                            other_person.status_start_time = current_time  # Start the persistence timer
                            self.assault_detected = True
                            self.last_detection_time = current_time
                    
                    # Check right hand proximity
                    if person.right_hand_speed.is_high_speed and right_hand_pos:
                        distance = self.calculate_distance(right_hand_pos, other_center)
                        if distance < self.interaction_threshold:
                            other_person.is_victim = True
                            other_person.status_start_time = current_time  # Start the persistence timer
                            self.assault_detected = True
                            self.last_detection_time = current_time
    
    def detect_persons(self, frame):
        """
        Detect persons in the frame
        
        This replaces YOLOv8 with a simpler approach using MediaPipe
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        
        boxes = []
        
        # For simplicity, we're using pose detection to detect persons
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get bounding box from landmarks
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for landmark in landmarks:
                px, py = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, px)
                y_min = min(y_min, py)
                x_max = max(x_max, px)
                y_max = max(y_max, py)
            
            # Add some margin
            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            # Add valid boxes
            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
        
        return boxes, results
    
    def process_frame(self, frame):
        """Process frame to detect persons with landmarks and detect potential assault"""
        # Update frame timing
        self.prev_frame_time = self.curr_frame_time
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if (self.curr_frame_time - self.prev_frame_time) > 0 else 0
        
        # Detect persons using MediaPipe
        person_boxes, mp_results = self.detect_persons(frame)
        
        # Make a copy for drawing
        annotated_frame = frame.copy()
        
        # Assign IDs to detected persons
        id_assignments = self.assign_person_id(person_boxes, self.tracked_persons)
        
        # Dictionary to store current frame's persons
        current_persons = {}
        
        # Process each detected person
        for i, box in enumerate(person_boxes):
            person_id = id_assignments[i]
            
            # Retrieve existing person or create new one
            if person_id in self.tracked_persons:
                person = self.tracked_persons[person_id]
                # Update timestamps
                person.prev_timestamp = person.timestamp
                person.timestamp = time.time()
                # Store previous positions
                person.prev_center = person.get_center()
                person.prev_left_hand = person.left_hand.wrist if person.left_hand else None
                person.prev_right_hand = person.right_hand.wrist if person.right_hand else None
                # Update box
                person.box = box
            else:
                # Create new person
                person = PersonData(id=person_id, box=box)
            
            x1, y1, x2, y2 = box
            
            # Ensure box is within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Skip if box is too small
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            
            # Extract person ROI
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue
            
            # Convert ROI to RGB for MediaPipe
            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(rgb_roi)
            
            # Draw bounding box with ID
            box_color = (0, 255, 0)  # Default color (green)
            if person.is_assaulter:
                box_color = (0, 0, 255)  # Red for assaulter
            elif person.is_victim:
                box_color = (255, 0, 0)  # Blue for victim
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated_frame, f"ID: {person_id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Scale factors for mapping back to original frame
            scale_x = (x2 - x1) / person_roi.shape[1]
            scale_y = (y2 - y1) / person_roi.shape[0]
            
            # Process pose landmarks
            if results.pose_landmarks:
                pose_points = []
                
                # Draw selected pose landmarks
                for idx in self.selected_pose_keypoints:
                    landmark = results.pose_landmarks.landmark[idx]
                    # Map coordinates back to original frame
                    px = int(landmark.x * person_roi.shape[1] * scale_x) + x1
                    py = int(landmark.y * person_roi.shape[0] * scale_y) + y1
                    pose_points.append((px, py))
                    cv2.circle(annotated_frame, (px, py), 5, (0, 255, 255), -1)
                
                # Connect landmarks with lines
                connections = [
                    (0, 1), (0, 2),  # Nose to shoulders
                    (1, 3), (3, 5),  # Left arm
                    (2, 4), (4, 6),  # Right arm
                    (1, 7), (2, 8),  # Shoulders to hips
                    (7, 9), (9, 11),  # Left leg
                    (8, 10), (10, 12)  # Right leg
                ]
                
                for connection in connections:
                    if connection[0] < len(pose_points) and connection[1] < len(pose_points):
                        cv2.line(annotated_frame, 
                                pose_points[connection[0]], 
                                pose_points[connection[1]], 
                                (0, 255, 0), 2)
                
                person.pose_landmarks = pose_points
            
            # Process left hand landmarks
            if results.left_hand_landmarks:
                left_hand = self._extract_simplified_hand_landmarks(results.left_hand_landmarks)
                
                # Map simplified landmarks to original frame
                if left_hand:
                    points = []
                    for point_name in ["wrist", "thumb_tip", "index_base", "index_tip", 
                                     "middle_tip", "ring_tip",