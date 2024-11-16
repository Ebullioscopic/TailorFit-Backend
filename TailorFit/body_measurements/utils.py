import cv2
import mediapipe as mp
import numpy as np
from django.conf import settings
import os

class BodyMeasurementProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera parameters
        self.real_distance_to_person = 60.0
        self.focal_length_mm = 5.0
        self.sensor_height_mm = 3.6
        
        self.length_adjustment_factor = 0.95
        self.circumference_adjustment_factors = {
            "chest": 0.1,
            "waist": 1.49,
            "hips": 1.43,
            "biceps": 0.0001,
            "forearm": 0.55,
            "thigh": 0.40,
            "calf": 0.35
        }

    def calculate_scale_factor(self, distance_to_person, focal_length, sensor_height, image_height_pixels):
        focal_length_inches = focal_length / 25.4
        sensor_height_inches = sensor_height / 25.4
        return (distance_to_person * sensor_height_inches) / (focal_length_inches * image_height_pixels)

    def get_landmark(self, landmarks, idx, image_shape):
        landmark = landmarks[idx]
        return (int(landmark.x * image_shape[1]), int(landmark.y * image_shape[0]))

    def calculate_circumference(self, width_in_pixels, scale_factor, body_part_factor=0.6):
        depth_in_pixels = width_in_pixels * body_part_factor
        circumference_in_pixels = np.pi * np.sqrt(0.5 * (width_in_pixels**2 + depth_in_pixels**2))
        return circumference_in_pixels * scale_factor

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def process_image(self, image_path):
        # Read and process image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None, None
            
        landmarks = results.pose_landmarks.landmark
        
        # Get landmarks
        landmark_points = {
            'shoulder_left': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, image.shape),
            'shoulder_right': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, image.shape),
            'hip_left': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP.value, image.shape),
            'hip_right': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP.value, image.shape),
            'elbow_left': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW.value, image.shape),
            'elbow_right': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value, image.shape),
            'wrist_left': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST.value, image.shape),
            'wrist_right': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST.value, image.shape),
            'knee_left': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE.value, image.shape),
            'knee_right': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE.value, image.shape),
            'ankle_left': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE.value, image.shape),
            'ankle_right': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value, image.shape),
            'head': self.get_landmark(landmarks, self.mp_pose.PoseLandmark.NOSE.value, image.shape)
        }
        
        # Calculate height and scale factor
        height_in_pixels = (self.calculate_distance(landmark_points['head'], landmark_points['ankle_left']) + 
                          self.calculate_distance(landmark_points['head'], landmark_points['ankle_right'])) / 2
        
        scale_factor = self.calculate_scale_factor(
            self.real_distance_to_person,
            self.focal_length_mm,
            self.sensor_height_mm,
            height_in_pixels
        )
        
        # Calculate measurements
        measurements = {}
        
        # Basic widths
        shoulder_width = self.calculate_distance(landmark_points['shoulder_left'], landmark_points['shoulder_right'])
        chest_width = shoulder_width * 1.2
        waist_width = self.calculate_distance(landmark_points['hip_left'], landmark_points['hip_right']) * 1.1
        hip_width = self.calculate_distance(landmark_points['hip_left'], landmark_points['hip_right']) * 1.15
        
        # Calculate circumferences
        measurements['chest_circumference'] = self.calculate_circumference(
            chest_width, scale_factor, self.circumference_adjustment_factors["chest"])
        measurements['waist_circumference'] = self.calculate_circumference(
            waist_width, scale_factor, self.circumference_adjustment_factors["waist"])
        measurements['hip_circumference'] = self.calculate_circumference(
            hip_width, scale_factor, self.circumference_adjustment_factors["hips"])
        
        # Arms
        measurements['left_bicep_circumference'] = self.calculate_circumference(
            self.calculate_distance(landmark_points['shoulder_left'], landmark_points['elbow_left']),
            scale_factor=0.045, body_part_factor=self.circumference_adjustment_factors["biceps"])
        measurements['right_bicep_circumference'] = self.calculate_circumference(
            self.calculate_distance(landmark_points['shoulder_right'], landmark_points['elbow_right']),
            scale_factor=0.045, body_part_factor=self.circumference_adjustment_factors["biceps"])
        
        # Legs
        measurements['left_thigh_circumference'] = self.calculate_circumference(
            self.calculate_distance(landmark_points['hip_left'], landmark_points['knee_left']),
            scale_factor=0.055, body_part_factor=self.circumference_adjustment_factors["thigh"])
        measurements['right_thigh_circumference'] = self.calculate_circumference(
            self.calculate_distance(landmark_points['hip_right'], landmark_points['knee_right']),
            scale_factor=0.055, body_part_factor=self.circumference_adjustment_factors["thigh"])
        
        # Draw landmarks and save processed image
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        # Add measurements to image
        y_position = 30
        for key, value in measurements.items():
            cv2.putText(annotated_image, f"{key}: {value:.2f} in", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_position += 30
        print(measurements)
        
        return measurements, annotated_image