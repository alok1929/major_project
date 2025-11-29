import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from utils.utils import get_joint_angles, calculate_angle


model = YOLO("yolov8n-pose.pt")
clf = joblib.load("models/model_ex1.pkl")

print("🔍 Model expects these features in this order:")
if hasattr(clf, 'feature_names_in_'):
    print(clf.feature_names_in_)
    expected_features = clf.feature_names_in_
else:
    print("Model doesn't have feature names stored")
    expected_features = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
                        'left_hip', 'right_hip', 'left_knee', 'right_knee']
print()


video_path = "/Users/alokhegde/major_project/rehab24_6/videos/Ex1/PM_000-Camera17-30fps.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Cannot open video: {video_path}")
    exit()

print(f"✅ Testing on video: {video_path}")
print("Press 'q' to quit, 's' to slow down, 'f' to speed up\n")

frame_count = 0
correct_count = 0
incorrect_count = 0
delay = 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n📹 Video ended or no more frames")
        break

    frame_count += 1

    # Run YOLO pose
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()[0]
        angles = get_joint_angles(keypoints)

        if len(angles) >= 4:
            # ✅ Create DataFrame with features in the EXACT order the model expects
            features_dict = {feature: angles.get(feature, 0) for feature in expected_features}
            features = pd.DataFrame([features_dict])
            
            # 🔍 DEBUG: Print features on first frame
            if frame_count == 1:
                print("🔍 Features being sent to model:")
                print(features.columns.tolist())
                print(features.iloc[0].to_dict())
                print()
            
            # Predict posture correctness
            pred = clf.predict(features)[0]
            pred_proba = clf.predict_proba(features)[0]
            
            label = f"✅ Correct ({pred_proba[1]:.2f})" if pred == 1 else f"❌ Incorrect ({pred_proba[0]:.2f})"
            print(pred_proba)
            print(label)

            if pred == 1:
                correct_count += 1
            else:
                incorrect_count += 1

            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            
            # Display prediction and angles
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display some angle values for debugging
            y_pos = 100
            for angle_name, angle_val in list(angles.items())[:4]:
                print(f"{angle_name}: {angle_val:.1f}")
                y_pos += 25
    
    cv2.imshow("Testing on Training Video", annotated)
    
    key = cv2.waitKey(delay)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📊 Results:")
print(f"Total frames processed: {frame_count}")
print(f"✅ Correct: {correct_count}")
print(f"❌ Incorrect: {incorrect_count}")
if correct_count + incorrect_count > 0:
    print(f"Accuracy: {correct_count/(correct_count+incorrect_count)*100:.1f}%")
print("\n Session ended.")

import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from utils.utils import get_joint_angles, calculate_angle


model = YOLO("yolov8n-pose.pt")
clf = joblib.load("models/model_ex1.pkl")

print("🔍 Model expects these features in this order:")
if hasattr(clf, 'feature_names_in_'):
    print(clf.feature_names_in_)
    expected_features = clf.feature_names_in_
else:
    print("Model doesn't have feature names stored")
    expected_features = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
                        'left_hip', 'right_hip', 'left_knee', 'right_knee']
print()


video_path = "/Users/alokhegde/major_project/rehab24_6/videos/Ex1/PM_000-Camera17-30fps.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Cannot open video: {video_path}")
    exit()

print(f"✅ Testing on video: {video_path}")
print("Press 'q' to quit, 's' to slow down, 'f' to speed up\n")

frame_count = 0
correct_count = 0
incorrect_count = 0
delay = 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n📹 Video ended or no more frames")
        break

    frame_count += 1

    # Run YOLO pose
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()[0]
        angles = get_joint_angles(keypoints)

        if len(angles) >= 4:
            # ✅ Create DataFrame with features in the EXACT order the model expects
            features_dict = {feature: angles.get(feature, 0) for feature in expected_features}
            features = pd.DataFrame([features_dict])
            
            # 🔍 DEBUG: Print features on first frame
            if frame_count == 1:
                print("🔍 Features being sent to model:")
                print(features.columns.tolist())
                print(features.iloc[0].to_dict())
                print()
            
            # Predict posture correctness
            pred = clf.predict(features)[0]
            pred_proba = clf.predict_proba(features)[0]
            
            label = f"✅ Correct ({pred_proba[1]:.2f})" if pred == 1 else f"❌ Incorrect ({pred_proba[0]:.2f})"
            print(pred_proba)
            print(label)

            if pred == 1:
                correct_count += 1
            else:
                incorrect_count += 1

            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            
            # Display prediction and angles
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display some angle values for debugging
            y_pos = 100
            for angle_name, angle_val in list(angles.items())[:4]:
                print(f"{angle_name}: {angle_val:.1f}")
                y_pos += 25
    
    cv2.imshow("Testing on Training Video", annotated)
    
    key = cv2.waitKey(delay)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📊 Results:")
print(f"Total frames processed: {frame_count}")
print(f"✅ Correct: {correct_count}")
print(f"❌ Incorrect: {incorrect_count}")
if correct_count + incorrect_count > 0:
    print(f"Accuracy: {correct_count/(correct_count+incorrect_count)*100:.1f}%")
print("\n Session ended.")

import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from utils.utils import get_joint_angles, calculate_angle


model = YOLO("yolov8n-pose.pt")
clf = joblib.load("models/model_ex1.pkl")

print("🔍 Model expects these features in this order:")
if hasattr(clf, 'feature_names_in_'):
    print(clf.feature_names_in_)
    expected_features = clf.feature_names_in_
else:
    print("Model doesn't have feature names stored")
    expected_features = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
                        'left_hip', 'right_hip', 'left_knee', 'right_knee']
print()


video_path = "/Users/alokhegde/major_project/rehab24_6/videos/Ex1/PM_000-Camera17-30fps.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Cannot open video: {video_path}")
    exit()

print(f"✅ Testing on video: {video_path}")
print("Press 'q' to quit, 's' to slow down, 'f' to speed up\n")

frame_count = 0
correct_count = 0
incorrect_count = 0
delay = 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n📹 Video ended or no more frames")
        break

    frame_count += 1

    # Run YOLO pose
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()[0]
        angles = get_joint_angles(keypoints)

        if len(angles) >= 4:
            # ✅ Create DataFrame with features in the EXACT order the model expects
            features_dict = {feature: angles.get(feature, 0) for feature in expected_features}
            features = pd.DataFrame([features_dict])
            
            # 🔍 DEBUG: Print features on first frame
            if frame_count == 1:
                print("🔍 Features being sent to model:")
                print(features.columns.tolist())
                print(features.iloc[0].to_dict())
                print()
            
            # Predict posture correctness
            pred = clf.predict(features)[0]
            pred_proba = clf.predict_proba(features)[0]
            
            label = f"✅ Correct ({pred_proba[1]:.2f})" if pred == 1 else f"❌ Incorrect ({pred_proba[0]:.2f})"
            print(pred_proba)
            print(label)

            if pred == 1:
                correct_count += 1
            else:
                incorrect_count += 1

            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            
            # Display prediction and angles
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display some angle values for debugging
            y_pos = 100
            for angle_name, angle_val in list(angles.items())[:4]:
                print(f"{angle_name}: {angle_val:.1f}")
                y_pos += 25
    
    cv2.imshow("Testing on Training Video", annotated)
    
    key = cv2.waitKey(delay)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📊 Results:")
print(f"Total frames processed: {frame_count}")
print(f"✅ Correct: {correct_count}")
print(f"❌ Incorrect: {incorrect_count}")
if correct_count + incorrect_count > 0:
    print(f"Accuracy: {correct_count/(correct_count+incorrect_count)*100:.1f}%")
print("\n Session ended.")

import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from utils.utils import get_joint_angles, calculate_angle


model = YOLO("yolov8n-pose.pt")
clf = joblib.load("models/model_ex1.pkl")

print("🔍 Model expects these features in this order:")
if hasattr(clf, 'feature_names_in_'):
    print(clf.feature_names_in_)
    expected_features = clf.feature_names_in_
else:
    print("Model doesn't have feature names stored")
    expected_features = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
                        'left_hip', 'right_hip', 'left_knee', 'right_knee']
print()


video_path = "/Users/alokhegde/major_project/rehab24_6/videos/Ex1/PM_000-Camera17-30fps.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Cannot open video: {video_path}")
    exit()

print(f"✅ Testing on video: {video_path}")
print("Press 'q' to quit, 's' to slow down, 'f' to speed up\n")

frame_count = 0
correct_count = 0
incorrect_count = 0
delay = 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n📹 Video ended or no more frames")
        break

    frame_count += 1

    # Run YOLO pose
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()[0]
        angles = get_joint_angles(keypoints)

        if len(angles) >= 4:
            # ✅ Create DataFrame with features in the EXACT order the model expects
            features_dict = {feature: angles.get(feature, 0) for feature in expected_features}
            features = pd.DataFrame([features_dict])
            
            # 🔍 DEBUG: Print features on first frame
            if frame_count == 1:
                print("🔍 Features being sent to model:")
                print(features.columns.tolist())
                print(features.iloc[0].to_dict())
                print()
            
            # Predict posture correctness
            pred = clf.predict(features)[0]
            pred_proba = clf.predict_proba(features)[0]
            
            label = f"✅ Correct ({pred_proba[1]:.2f})" if pred == 1 else f"❌ Incorrect ({pred_proba[0]:.2f})"
            print(pred_proba)
            print(label)

            if pred == 1:
                correct_count += 1
            else:
                incorrect_count += 1

            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            
            # Display prediction and angles
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display some angle values for debugging
            y_pos = 100
            for angle_name, angle_val in list(angles.items())[:4]:
                print(f"{angle_name}: {angle_val:.1f}")
                y_pos += 25
    
    cv2.imshow("Testing on Training Video", annotated)
    
    key = cv2.waitKey(delay)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📊 Results:")
print(f"Total frames processed: {frame_count}")
print(f"✅ Correct: {correct_count}")
print(f"❌ Incorrect: {incorrect_count}")
if correct_count + incorrect_count > 0:
    print(f"Accuracy: {correct_count/(correct_count+incorrect_count)*100:.1f}%")
print("\n Session ended.")

