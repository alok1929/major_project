import cv2
import numpy as np
from ultralytics import YOLO

def calculate_angle(point1, point2, point3):
    """
    Calculate angle between three points.
    point2 is the vertex of the angle.
    Returns angle in degrees.
    """
    # Convert points to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    # Calculate vectors
    vector1 = p1 - p2
    vector2 = p3 - p2
    
    # Calculate angle using dot product
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def get_joint_angles(keypoints):
    """
    Calculate angles for major joints from COCO keypoints.
    COCO format: 0-Nose, 1-LEye, 2-REye, 3-LEar, 4-REar, 
                 5-LShoulder, 6-RShoulder, 7-LElbow, 8-RElbow,
                 9-LWrist, 10-RWrist, 11-LHip, 12-RHip,
                 13-LKnee, 14-RKnee, 15-LAnkle, 16-RAnkle
    """
    angles = {}
    
    # Left elbow angle (shoulder-elbow-wrist)
    if all(keypoints[[5, 7, 9], 2] > 0.5):  # Check confidencex
        angles['left_elbow'] = calculate_angle(
            keypoints[5, :2], keypoints[7, :2], keypoints[9, :2]
        )
    
    # Right elbow angle
    if all(keypoints[[6, 8, 10], 2] > 0.5):
        angles['right_elbow'] = calculate_angle(
            keypoints[6, :2], keypoints[8, :2], keypoints[10, :2]
        )
    
    # Left knee angle (hip-knee-ankle)
    if all(keypoints[[11, 13, 15], 2] > 0.5):
        angles['left_knee'] = calculate_angle(
            keypoints[11, :2], keypoints[13, :2], keypoints[15, :2]
        )
    
    # Right knee angle
    if all(keypoints[[12, 14, 16], 2] > 0.5):
        angles['right_knee'] = calculate_angle(
            keypoints[12, :2], keypoints[14, :2], keypoints[16, :2]
        )
    
    # Left shoulder angle (hip-shoulder-elbow)
    if all(keypoints[[11, 5, 7], 2] > 0.5):
        angles['left_shoulder'] = calculate_angle(
            keypoints[11, :2], keypoints[5, :2], keypoints[7, :2]
        )
    
    # Right shoulder angle
    if all(keypoints[[12, 6, 8], 2] > 0.5):
        angles['right_shoulder'] = calculate_angle(
            keypoints[12, :2], keypoints[6, :2], keypoints[8, :2]
        )
    
    # Left hip angle (shoulder-hip-knee)
    if all(keypoints[[5, 11, 13], 2] > 0.5):
        angles['left_hip'] = calculate_angle(
            keypoints[5, :2], keypoints[11, :2], keypoints[13, :2]
        )
    
    # Right hip angle
    if all(keypoints[[6, 12, 14], 2] > 0.5):
        angles['right_hip'] = calculate_angle(
            keypoints[6, :2], keypoints[12, :2], keypoints[14, :2]
        )
    
    return angles

def main():
    # Load the pre-trained YOLOv8 pose model
    print("Loading model...")
    model = YOLO('yolov8n-pose.pt')  # 'n' for nano (fastest), use 'm' or 'l' for better accuracy
    
    # Open camera (0 for default webcam, or provide video file path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("Starting pose tracking... Press 'q' to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Run pose detection
        results = model(frame, verbose=False)
        
        # Process results
        if results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.data.cpu().numpy()
            
            # Process each detected person (up to 2 people)
            num_people = min(len(keypoints_data), 4)
            
            print(f"\n--- Frame {frame_count} ---")
            
            for person_idx in range(num_people):
                keypoints = keypoints_data[person_idx]
                angles = get_joint_angles(keypoints)
                
                print(f"Person {person_idx + 1}:")
                for joint, angle in angles.items():
                    print(f"  {joint}: {angle:.1f}°")
        
        # Display the frame with pose overlay (optional)
        annotated_frame = results[0].plot()
        cv2.imshow('Pose Tracking', annotated_frame)
        
        frame_count += 1
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nTracking stopped.")

if __name__ == "__main__":
    main()