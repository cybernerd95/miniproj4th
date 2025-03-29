import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Deque
from collections import deque
import time
import math

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
    def __init__(self, yolo_model_size="n", min_detection_confidence=0.5, 
                 speed_threshold=100, interaction_threshold=50):
        """
        Initialize detector for person detection with assault detection capabilities
        
        Args:
            yolo_model_size (str): Size of YOLOv8 model (n, s, m, l, x)
            min_detection_confidence (float): Confidence threshold for detection
            speed_threshold (float): Threshold for high-speed movement (pixels/second)
            interaction_threshold (float): Distance threshold for interaction detection (pixels)
        """
        self.conf_threshold = min_detection_confidence
        self.speed_threshold = speed_threshold
        self.interaction_threshold = interaction_threshold
        
        # Initialize YOLOv8 for person detection
        self.yolo_model = YOLO(f"yolov8{yolo_model_size}.pt")
        
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
            
            # Check for potential assaulter
            if person.left_hand_speed.is_high_speed or person.right_hand_speed.is_high_speed:
                person.is_assaulter = True
                person.flagged_body_part = "Left Hand" if left_hand_relative_speed > right_hand_relative_speed else "Right Hand"
                person.assault_confidence = max(left_hand_relative_speed, right_hand_relative_speed) / (high_speed_threshold * 2)
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
    
    def process_frame(self, frame):
        """Process frame to detect persons with landmarks and detect potential assault"""
        # Update frame timing
        self.prev_frame_time = self.curr_frame_time
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if (self.curr_frame_time - self.prev_frame_time) > 0 else 0
        
        # Detect persons using YOLOv8
        yolo_results = self.yolo_model(frame, conf=self.conf_threshold, classes=[0])  # 0 is person class
        
        # Get person bounding boxes
        person_boxes = []
        for result in yolo_results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                person_boxes.append([x1, y1, x2, y2])
      
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
                box_color = (255, 0, 0)  # Orange for victim
                
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
                                     "middle_tip", "ring_tip", "pinky_tip"]:
                        point = getattr(left_hand, point_name)
                        px = int(point[0] * person_roi.shape[1] * scale_x) + x1
                        py = int(point[1] * person_roi.shape[0] * scale_y) + y1
                        points.append((px, py))
                    
                    # Determine hand color based on speed
                    hand_color = (255, 0, 0)  # Default blue
                    if person.left_hand_speed.is_high_speed:
                        hand_color = (0, 0, 255)  # Red for high speed
                    
                    # Draw simplified hand landmarks
                    for point in points:
                        cv2.circle(annotated_frame, point, 5, hand_color, -1)
                    
                    # Connect points with lines for better visualization
                    connections = [(0, 1), (0, 2), (2, 3), (0, 4), (0, 5), (0, 6)]
                    for connection in connections:
                        cv2.line(annotated_frame, points[connection[0]], 
                               points[connection[1]], hand_color, 2)
                    
                    # Detect gesture and display
                    person.left_gesture = self.detect_gesture(left_hand)
                    
                    # Display speed
                    if person.left_hand_speed.value > 0:
                        speed_text = f"L: {person.left_hand_speed.value:.1f} px/s"
                        cv2.putText(annotated_frame, speed_text, 
                                  (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, hand_color, 2)
                    
                    # Store mapped hand landmarks
                    remapped_hand = SimplifiedHandLandmarks(
                        wrist=points[0],
                        thumb_tip=points[1],
                        index_base=points[2],
                        index_tip=points[3],
                        middle_tip=points[4],
                        ring_tip=points[5],
                        pinky_tip=points[6]
                    )
                    person.left_hand = remapped_hand
                    
                    # Store hand position in history
                    person.left_hand_history.append(points[0])  # Wrist position
            
            # Process right hand landmarks
            if results.right_hand_landmarks:
                right_hand = self._extract_simplified_hand_landmarks(results.right_hand_landmarks)
                
                # Map simplified landmarks to original frame
                if right_hand:
                    points = []
                    for point_name in ["wrist", "thumb_tip", "index_base", "index_tip", 
                                     "middle_tip", "ring_tip", "pinky_tip"]:
                        point = getattr(right_hand, point_name)
                        px = int(point[0] * person_roi.shape[1] * scale_x) + x1
                        py = int(point[1] * person_roi.shape[0] * scale_y) + y1
                        points.append((px, py))
                    
                    # Determine hand color based on speed
                    hand_color = (0, 0, 255)  # Default red
                    if person.right_hand_speed.is_high_speed:
                        hand_color = (255, 0, 0)  # Blue for high speed
                    
                    # Draw simplified hand landmarks
                    for point in points:
                        cv2.circle(annotated_frame, point, 5, hand_color, -1)
                    
                    # Connect points with lines for better visualization
                    connections = [(0, 1), (0, 2), (2, 3), (0, 4), (0, 5), (0, 6)]
                    for connection in connections:
                        cv2.line(annotated_frame, points[connection[0]], 
                               points[connection[1]], hand_color, 2)
                    
                    # Detect gesture and display
                    person.right_gesture = self.detect_gesture(right_hand)
                    
                    # Display speed
                    if person.right_hand_speed.value > 0:
                        speed_text = f"R: {person.right_hand_speed.value:.1f} px/s"
                        text_x = min(x1 + 100, frame.shape[1] - 100)
                        cv2.putText(annotated_frame, speed_text, 
                                  (text_x, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.6, hand_color, 2)
                    
                    # Store mapped hand landmarks
                    remapped_hand = SimplifiedHandLandmarks(
                        wrist=points[0],
                        thumb_tip=points[1],
                        index_base=points[2],
                        index_tip=points[3],
                        middle_tip=points[4],
                        ring_tip=points[5],
                        pinky_tip=points[6]
                    )
                    person.right_hand = remapped_hand
                    
                    # Store hand position in history
                    person.right_hand_history.append(points[0])  # Wrist position
            
            # Add person data to current frame's dictionary
            current_persons[person_id] = person
            
            # Add flags and text for assaulter/victim
            status_y = y1 - 50
            if person.is_assaulter:
                # Calculate remaining time
                time_left = self.status_persistence - (time.time() - person.status_start_time)
                time_left = max(0, time_left)
                
                cv2.putText(annotated_frame, f"ASSAULTER ({person.flagged_body_part}) [{time_left:.1f}s]", 
                          (x1, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show confidence level
                conf_text = f"Conf: {person.assault_confidence:.2f}"
                cv2.putText(annotated_frame, conf_text, 
                          (x1, status_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            elif person.is_victim:
                # Calculate remaining time
                time_left = self.status_persistence - (time.time() - person.status_start_time)
                time_left = max(0, time_left)
                
                cv2.putText(annotated_frame, f"VICTIM [{time_left:.1f}s]", 
                          (x1, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Detect interactions between persons
        self.detect_interactions(current_persons)
        
        # Update tracked persons
        self.tracked_persons = current_persons
        
        # Check if assault alert should be displayed
        if self.assault_detected:
            if time.time() - self.last_detection_time < self.alert_duration:
                # Draw warning on frame
                cv2.putText(annotated_frame, "⚠️ ASSAULT DETECTED ⚠️", 
                          (int(frame.shape[1]/2) - 200, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                self.assault_detected = False
        
        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display person count
        cv2.putText(annotated_frame, f"Persons: {len(current_persons)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display speed threshold
        cv2.putText(annotated_frame, f"Speed Threshold: {self.speed_threshold} px/s", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Check for expired flags in tracked persons
        current_time = time.time()
        for person_id, person in list(self.tracked_persons.items()):
            # If person is not in current frame
            if person_id not in current_persons:
                # If person was flagged, store in flagged_people
                if person.is_assaulter or person.is_victim:
                    # Only add if not already expired
                    if person.status_start_time > 0 and current_time - person.status_start_time < self.status_persistence:
                        self.flagged_people[person_id] = person
                # Remove from tracked persons
                del self.tracked_persons[person_id]
        
        # Process flagged people who are no longer in frame
        for person_id, person in list(self.flagged_people.items()):
            # Check if flag should expire
            if current_time - person.status_start_time > self.status_persistence:
                del self.flagged_people[person_id]
            # Check if person reappeared in current frame
            elif person_id in current_persons:
                # Transfer flag status to current person
                current_persons[person_id].is_assaulter = person.is_assaulter
                current_persons[person_id].is_victim = person.is_victim
                current_persons[person_id].flagged_body_part = person.flagged_body_part
                current_persons[person_id].assault_confidence = person.assault_confidence
                current_persons[person_id].status_start_time = person.status_start_time
                # Remove from flagged people
                del self.flagged_people[person_id]
                
        return annotated_frame

    def run_detection(self, video_source=0):
        """Run assault detection on video source"""
        cap = cv2.VideoCapture(video_source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            result_frame = self.process_frame(frame)
            
            # Display result
            cv2.imshow('Assault Detection', result_frame)
            
            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize detector with default settings
    detector = AssaultDetector(
        yolo_model_size="n",
        min_detection_confidence=0.5,
        speed_threshold=100,
        interaction_threshold=50
    )
    
    # Run detection on webcam (0) or video file
    detector.run_detection(0)  # Use webcam
    # Alternative: detector.run_detection("path/to/video.mp4")  # Use video file