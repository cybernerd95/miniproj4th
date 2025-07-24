import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Deque
from collections import deque
import time
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

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
    
    # Body part speeds
    limb_speeds: Dict[str, float] = field(default_factory=dict)
    
    # Previous positions
    prev_center: Optional[Tuple[float, float]] = None
    prev_left_hand: Optional[Tuple[float, float]] = None
    prev_right_hand: Optional[Tuple[float, float]] = None
    prev_pose_landmarks: Optional[List[Tuple[float, float]]] = None
    
    # Position history (for trajectory analysis)
    center_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=20))  # Increased from 10
    left_hand_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=20))  # Increased from 10
    right_hand_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=20))  # Increased from 10
    
    # Timestamps
    timestamp: float = field(default_factory=time.time)
    prev_timestamp: Optional[float] = None
    
    # Assault detection
    is_assaulter: bool = False
    is_victim: bool = False
    flagged_body_part: str = "None"  # Which body part triggered the alert
    assault_confidence: float = 0.0  # Confidence level for assault detection
    
    # Time tracking for persistent status
    status_start_time: float = 0  # When the assault/victim status started
    # Action recognition
    pose_history: Deque[List[Tuple[float, float]]] = field(default_factory=lambda: deque(maxlen=30))  # Increased from 20
    detected_action: str = "None"
    action_confidence: float = 0.0
    
    # Frame tracking (for cleanup)
    missing_frames: int = 0

    def update_timestamp(self):
        """Update timestamps for speed calculation"""
        self.prev_timestamp = self.timestamp
        self.timestamp = time.time()
        
    def get_center(self) -> Tuple[float, float]:
        """Calculate center point of the person"""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

class AssaultDetector:
    def __init__(self, yolo_model_size="n", min_detection_confidence=0.5, 
                speed_threshold=80, interaction_threshold=50,
                action_model_path=None,
                action_threshold=0.65):
        """
        Initialize detector for person detection with assault detection capabilities
        
        Args:
            yolo_model_size (str): Size of YOLOv8 model (n, s, m, l, x)
            min_detection_confidence (float): Confidence threshold for detection
            speed_threshold (float): Threshold for high-speed movement (pixels/second)
            interaction_threshold (float): Distance threshold for interaction detection (pixels)
            action_model_path (str): Path to action recognition model file
            action_threshold (float): Threshold for action recognition confidence
        """
        self.conf_threshold = min_detection_confidence
        self.speed_threshold = speed_threshold
        self.interaction_threshold = interaction_threshold
        self.action_threshold = action_threshold

        # Find H5 file in current directory if not specified
        if action_model_path is None:
            h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
            if h5_files:
                action_model_path = h5_files[0]
                print(f"Found model file: {action_model_path}")
            else:
                print("No H5 model file found in current directory")
                action_model_path = "action_model.h5"  # Default fallback

        # Load the action recognition model
        try:
            self.action_model = load_model(action_model_path)
            self.action_model_loaded = True
            print(f"Action recognition model loaded from {action_model_path}")
            # Get output shape to determine number of action classes
            output_shape = self.action_model.outputs[0].shape
            self.num_action_classes = output_shape[-1]
            print(f"Model predicts {self.num_action_classes} action classes")
            
            # Try to find class names file (same name as model but with _classes.txt)
            class_file = action_model_path.replace('.h5', '_classes.txt')
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    self.action_classes = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(self.action_classes)} action classes from {class_file}")
            else:
                # Default action classes
                self.action_classes = ["Normal", "Punch", "Kick", "Push", "Slap", "Other"]
                if self.num_action_classes != len(self.action_classes):
                    # Generate generic class names based on model output
                    self.action_classes = [f"Action_{i}" for i in range(self.num_action_classes)]
                print(f"Using default action classes: {self.action_classes}")
        except Exception as e:
            print(f"Error loading action model: {e}")
            print("Will run without action recognition")
            self.action_model_loaded = False
            self.action_classes = ["Normal", "Punch", "Kick", "Push", "Slap", "Other"]

        # Sequence length for action recognition model - try to determine from model input shape
        try:
            if self.action_model_loaded:
                input_shape = self.action_model.inputs[0].shape
                if len(input_shape) >= 2 and input_shape[1] is not None:
                    self.sequence_length = input_shape[1] 
                    print(f"Detected sequence length: {self.sequence_length}")
                else:
                    self.sequence_length = 10  # Default
            else:
                self.sequence_length = 10
        except:
            self.sequence_length = 10  # Default if can't determine
            
        print(f"Using sequence length: {self.sequence_length}")
        
        # Initialize YOLOv8 for person detection
        try:
            self.yolo_model = YOLO(f"yolov8{yolo_model_size}.pt")
            print(f"YOLOv8 model loaded: yolov8{yolo_model_size}.pt")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise ValueError("Failed to load YOLO model. Please check the model path.")
        
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
        
        # Define body part pairs for speed calculation
        self.limb_pairs = {
            "left_arm": (11, 15),    # left shoulder (landmark 11) to left wrist (landmark 15)
            "right_arm": (12, 16),   # right shoulder (landmark 12) to right wrist (landmark 16)
            "left_leg": (23, 27),    # left hip (landmark 23) to left ankle (landmark 27)
            "right_leg": (24, 28),   # right hip (landmark 24) to right ankle (landmark 28)
            "torso": (0, 24),        # nose (landmark 0) to right hip (landmark 24)
        } 
        
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
        
        # Frame processing capacity
        self.max_frames_to_process = 30  # Increased from default
    
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
    
    def calculate_body_part_speeds(self, person):
        """Calculate speeds of individual body parts relative to body center"""
        if not person.pose_landmarks or not person.prev_pose_landmarks or not person.prev_timestamp:
            return {}
            
        # Time difference
        time_diff = person.timestamp - person.prev_timestamp
        if time_diff <= 0:
            return {}
            
        # Calculate overall body speed (center of box)
        center = person.get_center()
        body_speed = self.calculate_speed(center, person.prev_center, time_diff) if person.prev_center else 0
        
        limb_speeds = {}
        
        # Calculate speeds for defined limb pairs
        for limb_name, (start_idx, end_idx) in self.limb_pairs.items():
            # Make sure the indices exist in the landmarks
            if (start_idx < len(self.selected_pose_keypoints) and 
                end_idx < len(self.selected_pose_keypoints)):
                
                # Get current landmark indices in the selected keypoints list
                curr_start_idx = self.selected_pose_keypoints.index(start_idx) if start_idx in self.selected_pose_keypoints else -1
                curr_end_idx = self.selected_pose_keypoints.index(end_idx) if end_idx in self.selected_pose_keypoints else -1
                
                if curr_start_idx >= 0 and curr_end_idx >= 0:
                    # Get current and previous positions
                    curr_start = person.pose_landmarks[curr_start_idx]
                    curr_end = person.pose_landmarks[curr_end_idx]
                    
                    prev_start = person.prev_pose_landmarks[curr_start_idx]
                    prev_end = person.prev_pose_landmarks[curr_end_idx]
                    
                    # Calculate midpoint of limb
                    curr_mid = ((curr_start[0] + curr_end[0])/2, (curr_start[1] + curr_end[1])/2)
                    prev_mid = ((prev_start[0] + prev_end[0])/2, (prev_start[1] + prev_end[1])/2)
                    
                    # Calculate speed
                    limb_speed = self.calculate_speed(curr_mid, prev_mid, time_diff)
                    
                    # Calculate relative speed (compared to body center)
                    relative_speed = limb_speed - body_speed
                    
                    limb_speeds[limb_name] = relative_speed
        
        # Add hand speeds if available
        left_hand_pos = person.left_hand.wrist if person.left_hand else None
        right_hand_pos = person.right_hand.wrist if person.right_hand else None
        
        if left_hand_pos and person.prev_left_hand:
            left_hand_speed = self.calculate_speed(left_hand_pos, person.prev_left_hand, time_diff)
            limb_speeds["left_hand"] = left_hand_speed - body_speed  # Relative to body
            
        if right_hand_pos and person.prev_right_hand:
            right_hand_speed = self.calculate_speed(right_hand_pos, person.prev_right_hand, time_diff)
            limb_speeds["right_hand"] = right_hand_speed - body_speed  # Relative to body
            
        return limb_speeds
    
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
            
            # Calculate limb speeds relative to body
            person.limb_speeds = self.calculate_body_part_speeds(person)
            
            # Skip if not enough speed data
            if not person.limb_speeds:
                continue
                
            # Check for high-speed body parts 
            has_high_speed_part = False
            fastest_part = "None"
            max_speed = 0
            
            for part_name, speed in person.limb_speeds.items():
                if speed > self.speed_threshold:
                    has_high_speed_part = True
                    if speed > max_speed:
                        max_speed = speed
                        fastest_part = part_name
            
            # Check for potential assaulter
            if has_high_speed_part:
                person.is_assaulter = True
                person.flagged_body_part = fastest_part
                person.assault_confidence = max_speed / (self.speed_threshold * 2)
                person.assault_confidence = min(max(person.assault_confidence, 0.0), 1.0)  # Clamp to [0,1]
                person.status_start_time = current_time  # Start the persistence timer
                
                # Check if high-speed body part is close to another person (potential victim)
                for other_id, other_person in persons.items():
                    if other_id == person_id:
                        continue
                    
                    other_center = other_person.get_center()
                    
                    # Get position of flagged body part (approximate)
                    flagged_part_position = None
                    
                    if fastest_part == "left_hand" and person.left_hand:
                        flagged_part_position = person.left_hand.wrist
                    elif fastest_part == "right_hand" and person.right_hand:
                        flagged_part_position = person.right_hand.wrist
                    elif person.pose_landmarks:
                        # For other body parts, use pose landmarks
                        if fastest_part == "left_arm" and len(person.pose_landmarks) > 5:
                            # Left wrist is usually index 5
                            flagged_part_position = person.pose_landmarks[5]
                        elif fastest_part == "right_arm" and len(person.pose_landmarks) > 6:
                            # Right wrist is usually index 6
                            flagged_part_position = person.pose_landmarks[6]
                        elif fastest_part == "left_leg" and len(person.pose_landmarks) > 11:
                            # Left ankle is usually index 11
                            flagged_part_position = person.pose_landmarks[11]
                        elif fastest_part == "right_leg" and len(person.pose_landmarks) > 12:
                            # Right ankle is usually index 12
                            flagged_part_position = person.pose_landmarks[12]
                    
                    # Check proximity
                    if flagged_part_position:
                        distance = self.calculate_distance(flagged_part_position, other_center)
                        if distance < self.interaction_threshold:
                            other_person.is_victim = True
                            other_person.status_start_time = current_time  # Start the persistence timer
                            self.assault_detected = True
                            self.last_detection_time = current_time

    def prepare_action_sequence(self, history):
        """Prepare pose sequence for action recognition model"""
        if len(history) < self.sequence_length:
            return None  # Not enough frames

        # Use only the last sequence_length frames
        recent_history = list(history)[-self.sequence_length:]

        sequence = []
        for frame_landmarks in recent_history:
            if not frame_landmarks:
                return None   # Invalid frame
                
            # Ensure we have exactly 33 landmarks (MediaPipe pose default)
            if len(frame_landmarks) != 33:
                return None   # Incorrect landmark count

            # Flatten x, y, z coordinates
            flat = []
            for point in frame_landmarks:
                if isinstance(point, tuple) or isinstance(point, list):
                    if len(point) >= 3:  # (x, y, z)
                        flat.extend([point[0], point[1], point[2]])
                    elif len(point) == 2:  # (x, y)
                        flat.extend([point[0], point[1], 0.0])
                    else:
                        return None  # Invalid point format
                else:
                    return None  # Invalid point format
            sequence.append(flat)  # Shape: (10, 99)
        
        # Convert to numpy array with proper shape
        try:
            return np.array([sequence], dtype=np.float32)  # Shape: (1, 10, 99)
        except ValueError:
            return None  # Something went wrong with the conversion
    
    def process_frame(self, frame):
        """Process frame to detect persons with landmarks and detect potential assault"""
        # Update frame timing - adjust for consistent delay
        self.prev_frame_time = self.curr_frame_time
        self.curr_frame_time = time.time()
        
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Warning: Empty frame received")
            return np.zeros((720, 1280, 3), dtype=np.uint8)  # Increased resolution
        
        # Resize large frames for better performance
        frame_height, frame_width = frame.shape[:2]
        scaling_factor = 1.0
        
        if frame_width > 1920:  # Increased from 1280
            scaling_factor = 1920 / frame_width
            frame = cv2.resize(frame, (0, 0), fx=scaling_factor, fy=scaling_factor)
        
        # Account for the 10ms delay when calculating FPS to get more accurate measurements
        actual_frame_time = self.curr_frame_time - self.prev_frame_time
        adjusted_time = max(actual_frame_time, 0.01)  # Ensure minimum 10ms (0.01s)
        fps = 1 / adjusted_time
        
        # Detect persons using YOLOv8
        try:
            yolo_results = self.yolo_model(frame, conf=self.conf_threshold, classes=[0])  # 0 is person class
            
            # Get person bounding boxes
            person_boxes = []
            for result in yolo_results:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    person_boxes.append([x1, y1, x2, y2])
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            person_boxes = []
      
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
                person.prev_pose_landmarks = person.pose_landmarks  # Store previous pose landmarks
                # Reset missing frames counter
                person.missing_frames = 0
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
            try:
                results = self.holistic.process(rgb_roi)
            except Exception as e:
                print(f"Error in MediaPipe processing: {e}")
                continue
            
            # Draw bounding box with ID
            box_color = (0, 255, 0)  # Default color (green)
            if person.is_assaulter:
                box_color = (0, 0, 255)  # Red for assaulter (BGR format)
            elif person.is_victim:
                box_color = (255, 0, 0)  # Blue for victim (BGR format)
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated_frame, f"ID: {person_id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Scale factors for mapping back to original frame
            # Scale factors for mapping back to original frame
            scale_x = frame_width / person_roi.shape[1]
            scale_y = frame_height / person_roi.shape[0]
            
            offset_x, offset_y = x1, y1
            
            # Process pose landmarks
            if results.pose_landmarks:
                # Extract selected pose landmarks
                pose_landmarks = []
                for idx in self.selected_pose_keypoints:
                    lm = results.pose_landmarks.landmark[idx]
                    # Scale to ROI size, then add offset to map back to full frame
                    x = lm.x * (x2 - x1) + offset_x
                    y = lm.y * (y2 - y1) + offset_y
                    z = lm.z
                    pose_landmarks.append((x, y, z))
                
                person.pose_landmarks = pose_landmarks
                
                # Append to pose history for action recognition
                if len(pose_landmarks) == len(self.selected_pose_keypoints):
                    full_pose = []
                    # Create full 33-point pose estimation (needed for action model)
                    for i in range(33):  # MediaPipe provides 33 pose landmarks
                        if i in self.selected_pose_keypoints:
                            idx = self.selected_pose_keypoints.index(i)
                            full_pose.append(pose_landmarks[idx])
                        else:
                            # For missing landmarks, use zeros or interpolate
                            full_pose.append((0.0, 0.0, 0.0))
                            
                    person.pose_history.append(full_pose)
                    
                    # Keep only the most recent frames based on sequence_length
                    if len(person.pose_history) > self.sequence_length * 2:
                        person.pose_history = deque(list(person.pose_history)[-self.sequence_length*2:], 
                                                  maxlen=self.sequence_length*2)
                
                # Draw pose landmarks on the annotated frame
                for i, (x, y, _) in enumerate(pose_landmarks):
                    cv2.circle(annotated_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                    
                    # Connect landmarks to show skeleton (simplified)
                    if i > 0 and i < len(pose_landmarks) - 1:
                        prev_point = (int(pose_landmarks[i-1][0]), int(pose_landmarks[i-1][1]))
                        curr_point = (int(x), int(y))
                        cv2.line(annotated_frame, prev_point, curr_point, (0, 255, 0), 2)
            
            # Process left hand landmarks
            if results.left_hand_landmarks:
                person.left_hand = self._extract_simplified_hand_landmarks(results.left_hand_landmarks)
                person.left_gesture = self.detect_gesture(person.left_hand)
                
                # Store in history
                if person.left_hand:
                    person.left_hand_history.append(person.left_hand.wrist)
                    
                # Draw gesture text
                if person.left_gesture != "None":
                    gesture_text = f"Left: {person.left_gesture}"
                    cv2.putText(annotated_frame, gesture_text, (x1, y2 + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Process right hand landmarks
            if results.right_hand_landmarks:
                person.right_hand = self._extract_simplified_hand_landmarks(results.right_hand_landmarks)
                person.right_gesture = self.detect_gesture(person.right_hand)
                
                # Store in history
                if person.right_hand:
                    person.right_hand_history.append(person.right_hand.wrist)
                    
                # Draw gesture text
                if person.right_gesture != "None":
                    gesture_text = f"Right: {person.right_gesture}"
                    cv2.putText(annotated_frame, gesture_text, (x1, y2 + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Store center position in history
            person.center_history.append(person.get_center())
            
            # Run action recognition model if available and enough pose history
            if self.action_model_loaded and len(person.pose_history) >= self.sequence_length:
                try:
                    # Check if any body part is moving fast enough compared to the rest of the body
                    body_part_speeds = self.calculate_body_part_speeds(person)
                    fast_body_part = False
                    
                    for part_name, speed in body_part_speeds.items():
                        # Check if body part speed exceeds threshold (100 px/s)
                        if speed > 100:  # Increased from original threshold
                            fast_body_part = True
                            break
                    
                    # Only run action recognition if a body part is moving fast
                    if fast_body_part:
                        # Prepare sequence for action recognition
                        sequence = self.prepare_action_sequence(person.pose_history)
                        
                        if sequence is not None:
                            # Make prediction
                            prediction = self.action_model.predict(sequence, verbose=0)
                            
                            # Get predicted class and confidence
                            predicted_class_idx = np.argmax(prediction[0])
                            confidence = prediction[0][predicted_class_idx]
                            
                            # Update person's detected action if confidence is high enough
                            if confidence > self.action_threshold:
                                person.detected_action = self.action_classes[predicted_class_idx]
                                person.action_confidence = float(confidence)
                                
                                # If detected action is assault-related, flag as assaulter
                                assault_actions = ["Punch", "Kick", "Push", "Slap"]
                                if person.detected_action in assault_actions:
                                    person.is_assaulter = True
                                    person.flagged_body_part = "Action: " + person.detected_action
                                    person.assault_confidence = confidence
                                    person.status_start_time = time.time()  # Reset the 10-second timer
                                    
                                    # Set assault detected flag
                                    self.assault_detected = True
                                    self.last_detection_time = time.time()
                                    
                                    # Look for potential victims (closest person)
                                    min_distance = float('inf')
                                    closest_person_id = None
                                    
                                    for other_id, other_person in current_persons.items():
                                        if other_id != person_id:
                                            other_center = other_person.get_center()
                                            distance = self.calculate_distance(person.get_center(), other_center)
                                            
                                            if distance < min_distance and distance < self.interaction_threshold * 3:
                                                min_distance = distance
                                                closest_person_id = other_id
                                    
                                    # Mark closest person as victim
                                    if closest_person_id is not None and closest_person_id in current_persons:
                                        victim = current_persons[closest_person_id]
                                        victim.is_victim = True
                                        victim.status_start_time = time.time()  # Reset the 10-second timer
                
                except Exception as e:
                    print(f"Error in action recognition: {e}")
            
            # Draw action text if an action was detected
            if person.detected_action != "None":
                action_text = f"{person.detected_action}: {person.action_confidence:.2f}"
                cv2.putText(annotated_frame, action_text, (x1, y2 + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Store updated person data
            current_persons[person_id] = person
        
        # Update tracked persons with current persons
        self.tracked_persons = current_persons.copy()
        
        # Process missing persons
        for person_id in list(self.tracked_persons.keys()):
            if person_id not in current_persons:
                self.tracked_persons[person_id].missing_frames += 1
                
                # Remove person if missing for too many frames
                if self.tracked_persons[person_id].missing_frames > 30:  # Increased from default
                    # If this person was flagged, store them in flagged_people
                    if self.tracked_persons[person_id].is_assaulter or self.tracked_persons[person_id].is_victim:
                        self.flagged_people[person_id] = {
                            "status": "assaulter" if self.tracked_persons[person_id].is_assaulter else "victim",
                            "time": time.time()
                        }
                    
                    # Remove from tracking
                    del self.tracked_persons[person_id]
        
        # Calculate additional metrics for interaction detection
        self.detect_interactions(self.tracked_persons)
        
        # Check if we should show assault alert
        if self.assault_detected and time.time() - self.last_detection_time < self.alert_duration:
            cv2.putText(annotated_frame, "ASSAULT DETECTED!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            self.assault_detected = False
        
        # Print stats
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"People: {len(self.tracked_persons)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw legend
        cv2.putText(annotated_frame, "Green: Normal", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Red: Assaulter", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, "Blue: Victim", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return annotated_frame
    
    def release(self):
        """Release resources"""
        self.holistic.close()


def main():
    """Main function to run assault detection on video"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run assault detection on video')
    parser.add_argument('--input', type=str, default='0', help='Input video file or camera index (default: 0)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video file (default: output.mp4)')
    parser.add_argument('--yolo-size', type=str, default='n', help='YOLOv8 model size (n, s, m, l, x) (default: n)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--speed-threshold', type=float, default=100, help='Speed threshold for assault detection (default: 100)')
    parser.add_argument('--interaction-distance', type=float, default=150, help='Distance threshold for interaction detection (default: 150)')
    parser.add_argument('--model', type=str, default=None, help='Path to action recognition model file')
    parser.add_argument('--output-size', type=str, default='1920x1080', help='Output video size (default: 1920x1080)')
    parser.add_argument('--max-frames', type=int, default=30, help='Maximum frames to process (default: 30)')
    
    args = parser.parse_args()
    
    # Parse input source
    if args.input.isdigit():
        input_source = int(args.input)
    else:
        input_source = args.input
    
    # Parse output size
    width, height = map(int, args.output_size.split('x'))
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        return
    
    # Set capture resolution higher if it's a camera
    if isinstance(input_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30  # Default FPS if not available
    
    # Initialize detector
    detector = AssaultDetector(
        yolo_model_size=args.yolo_size,
        min_detection_confidence=args.conf,
        speed_threshold=args.speed_threshold,
        interaction_threshold=args.interaction_distance,
        action_model_path=args.model,
        action_threshold=0.65
    )
    
    # Update max frames to process
    detector.max_frames_to_process = args.max_frames
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"Processing video with output size {width}x{height} at {fps} FPS")
    print(f"Using YOLOv8{args.yolo_size} model with confidence threshold {args.conf}")
    print(f"Speed threshold: {args.speed_threshold}, Interaction distance: {args.interaction_distance}")
    print(f"Action model: {args.model if args.model else 'Auto-detected'}")
    print(f"Max frames to process: {args.max_frames}")
    
    try:
        start_time = time.time()
        frame_count = 0
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = detector.process_frame(frame)
            
            # Resize processed frame to desired output size
            processed_frame = cv2.resize(processed_frame, (width, height))
            
            # Write frame to output
            out.write(processed_frame)
            
            # Display frame
            cv2.imshow('Assault Detection', processed_frame)
            
            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                break
            
            frame_count += 1
        
        # Print stats
        elapsed_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({frame_count/elapsed_time:.2f} FPS)")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        detector.release()
        print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()