import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import json

print("Loading Face Recognition System...")
print()

# Download model if needed
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face model (first time only)...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Done!")

# Initialize detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=2,
    min_face_detection_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)
print("Camera starting...")

# ============================================================================
# FACE RECOGNITION WITH QUALITY CHECKS
# ============================================================================

# Define key facial landmarks (64 points)
FIXED_LANDMARKS = [
    1, 2, 3, 4, 5,       # Nose
    33, 133,             # Left eye
    362, 263,            # Right eye  
    61, 291,             # Mouth
    234, 454,            # Jaw
    46, 276,             # Eyebrows
    127, 356,            # Cheeks
    152, 10,             # Chin, Forehead
    78, 308, 87, 317,    # Lips
    70, 88, 66, 105,     # Eye details
    195, 197, 199, 201,  # Nose sides
    129, 130, 131, 132,  # Nose bridge
    147, 148, 149, 150,  # Under eyes
    175, 176, 177, 178,  # Left cheek
    395, 396, 397, 398,  # Right cheek
]

# Remove duplicates and sort
FIXED_LANDMARKS = sorted(set(FIXED_LANDMARKS))
NUM_FEATURES = len(FIXED_LANDMARKS) * (len(FIXED_LANDMARKS) - 1) // 2

print(f"Using {len(FIXED_LANDMARKS)} facial landmarks")
print(f"Feature vector size: {NUM_FEATURES}")

# Face storage
known_faces = {}

# Load saved faces
database_file = "faces.json"
if os.path.exists(database_file):
    try:
        with open(database_file, "r") as f:
            data = json.load(f)
            for name, vectors in data.items():
                known_faces[name] = [np.array(v) for v in vectors]
        print(f"✅ Loaded {len(known_faces)} registered faces")
    except Exception as e:
        print(f"Error loading: {e}")

# ============================================================================
# QUALITY CHECK FUNCTIONS
# ============================================================================

def check_face_quality(landmarks, frame):
    """Check if face is suitable for registration"""
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    face_width = max(xs) - min(xs)
    face_height = max(ys) - min(ys)
    
    messages = []
    
    # Check face size (not too far, not too close)
    if face_width < 100 or face_height < 100:
        messages.append("Move closer")
    elif face_width > w * 0.7 or face_height > h * 0.7:
        messages.append("Move back")
    
    # Check if face is centered
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    if abs(center_x - w/2) > w * 0.2:
        messages.append("Center face horizontally")
    if abs(center_y - h/2) > h * 0.2:
        messages.append("Center face vertically")
    
    return messages

def are_eyes_open(landmarks):
    """Check if eyes are open using eye aspect ratio"""
    try:
        # Left eye landmarks
        left_eye_top = landmarks[159].y
        left_eye_bottom = landmarks[145].y
        left_eye_left = landmarks[33].x
        left_eye_right = landmarks[133].x
        
        left_eye_height = abs(left_eye_top - left_eye_bottom)
        left_eye_width = abs(left_eye_left - left_eye_right)
        left_ear = left_eye_height / left_eye_width if left_eye_width > 0 else 0
        
        # Right eye landmarks
        right_eye_top = landmarks[386].y
        right_eye_bottom = landmarks[374].y
        right_eye_left = landmarks[362].x
        right_eye_right = landmarks[263].x
        
        right_eye_height = abs(right_eye_top - right_eye_bottom)
        right_eye_width = abs(right_eye_left - right_eye_right)
        right_ear = right_eye_height / right_eye_width if right_eye_width > 0 else 0
        
        # If EAR < 0.15, eye is considered closed
        return left_ear > 0.15 and right_ear > 0.15
    except:
        return True  # If can't detect, assume eyes are open

def is_face_straight(landmarks):
    """Check if face is looking straight at camera"""
    try:
        # Nose tip vs nose bridge
        nose_tip_x = landmarks[1].x
        nose_bridge_x = landmarks[168].x
        
        # Left and right eye visibility (using Z coordinate)
        left_eye_z = landmarks[33].z
        right_eye_z = landmarks[362].z
        
        # Both eyes should be visible (Z close to same value)
        eyes_visible = abs(left_eye_z - right_eye_z) < 0.05
        
        # Nose should be centered
        nose_centered = abs(nose_tip_x - nose_bridge_x) < 0.05
        
        return eyes_visible and nose_centered
    except:
        return True  # If can't detect, assume face is straight

def check_brightness(frame):
    """Check if lighting is good"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    if brightness < 50:
        return False, f"Too dark ({brightness:.0f})"
    if brightness > 200:
        return False, f"Too bright ({brightness:.0f})"
    return True, f"Good lighting ({brightness:.0f})"

def is_duplicate(features, threshold=0.05):
    """Check if face is already registered"""
    for name, vectors in known_faces.items():
        for stored in vectors:
            distance = np.linalg.norm(features - stored)
            if distance < threshold:
                return True, name
    return False, None

def check_multi_faces(face_count):
    """Warn if multiple faces detected"""
    if face_count > 1:
        return False, f"{face_count} faces detected - only one person at a time"
    return True, ""

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def get_face_features(landmarks):
    """Extract features from face landmarks"""
    features = []
    
    # Get coordinates for all fixed landmarks
    points = []
    for idx in FIXED_LANDMARKS:
        if idx < len(landmarks):
            points.append([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        else:
            points.append([0, 0, 0])
    
    points = np.array(points)
    
    # Calculate distances between all pairs of points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            features.append(dist)
    
    # Normalize
    features = np.array(features)
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    
    return features

def draw_face_landmarks(frame, landmarks):
    """Draw face landmarks on frame"""
    h, w = frame.shape[:2]
    
    # Draw all landmarks as small dots
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    
    # Highlight the key landmarks in red
    for idx in FIXED_LANDMARKS:
        if idx < len(landmarks):
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
    
    return frame

def recognize_face(features, threshold=0.05):
    """
    Recognize face using EUCLIDEAN DISTANCE.
    Lower distance = more similar.
    """
    if not known_faces:
        return "Unknown", 0.0
    
    best_name = "Unknown"
    best_distance = float('inf')
    
    for name, vectors in known_faces.items():
        for stored_features in vectors:
            distance = np.linalg.norm(features - stored_features)
            
            if distance < best_distance and distance < threshold:
                best_distance = distance
                best_name = name
    
    if best_name != "Unknown":
        confidence = max(0, (1 - best_distance / threshold)) * 100
        return best_name, confidence
    return "Unknown", 0.0

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Configuration
SHOW_LANDMARKS = True
THRESHOLD = 0.05  # VERY STRICT
debug_mode = False

print("\n" + "="*60)
print("FACE RECOGNITION SYSTEM - PROFESSIONAL EDITION")
print("="*60)
print("\nControls:")
print("  Q - Quit")
print("  R - Register face (look at camera, then type name)")
print("  S - Save database to file")
print("  L - Load database from file")
print("  C - Clear all faces from memory")
print("  D - Toggle debug mode")
print("  +/- - Adjust threshold (current: 0.05, lower = stricter)")
print("  H - Hide/show landmarks")
print("="*60 + "\n")

print("⚠️  TIPS FOR BEST RESULTS:")
print("   - Look straight at the camera")
print("   - Keep your eyes open")
print("   - Ensure good lighting")
print("   - Center your face")
print("   - Only one person in frame when registering")
print()

registering = False
reg_name = ""
registration_quality_ok = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    
    # Quality check variables
    quality_messages = []
    face_count = len(result.face_landmarks) if result.face_landmarks else 0
    
    if result.face_landmarks:
        for landmarks in result.face_landmarks:
            # Extract features
            features = get_face_features(landmarks)
            
            if features is not None and len(features) > 0:
                # Recognize
                name, confidence = recognize_face(features, THRESHOLD)
                
                # ============================================
                # REGISTRATION WITH QUALITY CHECKS
                # ============================================
                if registering:
                    # Check multiple faces
                    multi_ok, multi_msg = check_multi_faces(face_count)
                    if not multi_ok:
                        quality_messages.append(multi_msg)
                        registration_quality_ok = False
                    else:
                        # Check face quality (size, position)
                        pos_messages = check_face_quality(landmarks, frame)
                        if pos_messages:
                            quality_messages.extend(pos_messages)
                            registration_quality_ok = False
                        else:
                            # Check eyes open
                            if not are_eyes_open(landmarks):
                                quality_messages.append("Open your eyes")
                                registration_quality_ok = False
                            # Check face straight
                            elif not is_face_straight(landmarks):
                                quality_messages.append("Look straight at camera")
                                registration_quality_ok = False
                            # Check brightness
                            else:
                                bright_ok, bright_msg = check_brightness(frame)
                                if not bright_ok:
                                    quality_messages.append(bright_msg)
                                    registration_quality_ok = False
                                else:
                                    registration_quality_ok = True
                    
                    # Auto-register if quality is good
                    if registration_quality_ok and name == "Unknown":
                        if reg_name:
                            # Check for duplicate before registering
                            is_dup, dup_name = is_duplicate(features, THRESHOLD)
                            if is_dup:
                                print(f"\n⚠️ This face is already registered as '{dup_name}'")
                                registering = False
                                reg_name = ""
                                registration_quality_ok = False
                            else:
                                if reg_name not in known_faces:
                                    known_faces[reg_name] = []
                                known_faces[reg_name].append(features)
                                if len(known_faces[reg_name]) > 3:
                                    known_faces[reg_name].pop(0)
                                print(f"\n✅ Registered '{reg_name}' successfully!")
                                print(f"   Features: {len(features)}")
                                print(f"   Total samples for {reg_name}: {len(known_faces[reg_name])}")
                                registering = False
                                reg_name = ""
                                registration_quality_ok = False
                        else:
                            reg_name = input("\n📝 Enter name: ").strip()
                            if not reg_name:
                                registering = False
                                registration_quality_ok = False
                
                # Draw on frame
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                if SHOW_LANDMARKS:
                    frame = draw_face_landmarks(frame, landmarks)
                
                # Draw bounding box and name
                h, w = frame.shape[:2]
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{name} ({confidence:.1f}%)" if name != "Unknown" else "Unknown Person"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Debug info
                if debug_mode and name != "Unknown":
                    cv2.putText(frame, f"Distance: {best_distance:.4f}", 
                               (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    
    # ============================================
    # DISPLAY OVERLAYS
    # ============================================
    
    # Registration mode indicator with quality messages
    if registering:
        # Green bar at top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (0, 255, 0), -1)
        
        if reg_name:
            cv2.putText(frame, f"REGISTERING: {reg_name}", 
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.putText(frame, "REGISTRATION MODE - Look at camera", 
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Show quality messages
        if quality_messages:
            y_offset = 65
            for msg in quality_messages:
                cv2.putText(frame, f"⚠️ {msg}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_offset += 25
        elif registration_quality_ok and not reg_name:
            cv2.putText(frame, "✅ Face quality GOOD - Press 'R' or type name", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Show face count warning
        if face_count > 1:
            cv2.putText(frame, f"⚠️ {face_count} faces detected - Only one person allowed", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Show threshold level indicator
    if THRESHOLD <= 0.07:
        threshold_status = "VERY STRICT"
        threshold_color = (0, 0, 255)
    elif THRESHOLD <= 0.12:
        threshold_status = "STRICT"
        threshold_color = (0, 165, 255)
    elif THRESHOLD <= 0.18:
        threshold_status = "NORMAL"
        threshold_color = (0, 255, 255)
    else:
        threshold_status = "LOOSE"
        threshold_color = (0, 255, 0)
    
    cv2.putText(frame, f"Threshold: {THRESHOLD:.3f} ({threshold_status})", 
                (frame.shape[1] - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, threshold_color, 1)
    
    # Instructions overlay
    cv2.putText(frame, f"Q:Quit R:Register S:Save L:Load C:Clear D:Debug H:Landmarks +/-:Thresh", 
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Known: {len(known_faces)} people", 
                (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Face Recognition System', frame)
    
    # ============================================
    # HANDLE KEYBOARD INPUT
    # ============================================
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27:
        break
    
    elif key == ord('r'):
        registering = not registering
        reg_name = ""
        registration_quality_ok = False
        print(f"\n🔴 Registration mode: {'ON' if registering else 'OFF'}")
        if registering:
            print("   Center your face, keep eyes open, look straight")
            print("   Quality checks will appear on screen")
    
    elif key == ord('s'):
        try:
            save_data = {}
            for name, vectors in known_faces.items():
                save_data[name] = [v.tolist() for v in vectors]
            with open(database_file, "w") as f:
                json.dump(save_data, f)
            print(f"\n💾 Saved {len(known_faces)} faces to {database_file}")
        except Exception as e:
            print(f"\n❌ Error saving: {e}")
    
    elif key == ord('l'):
        try:
            if os.path.exists(database_file):
                with open(database_file, "r") as f:
                    data = json.load(f)
                    known_faces = {}
                    for name, vectors in data.items():
                        known_faces[name] = [np.array(v) for v in vectors]
                print(f"\n✅ Loaded {len(known_faces)} faces from {database_file}")
            else:
                print(f"\n❌ No database file found: {database_file}")
        except Exception as e:
            print(f"\n❌ Error loading: {e}")
    
    elif key == ord('c'):
        known_faces = {}
        print(f"\n🗑️ Cleared all faces from memory")
    
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"\n🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    elif key == ord('h'):
        SHOW_LANDMARKS = not SHOW_LANDMARKS
        print(f"\n👁️ Landmarks: {'ON' if SHOW_LANDMARKS else 'OFF'}")
    
    elif key == ord('=') or key == ord('+'):
        THRESHOLD = min(0.30, THRESHOLD + 0.01)
        status = "VERY STRICT" if THRESHOLD <= 0.07 else "STRICT" if THRESHOLD <= 0.12 else "NORMAL" if THRESHOLD <= 0.18 else "LOOSE"
        print(f"\n📊 Threshold: {THRESHOLD:.3f} ({status})")
    
    elif key == ord('-') or key == ord('_'):
        THRESHOLD = max(0.03, THRESHOLD - 0.01)
        status = "VERY STRICT" if THRESHOLD <= 0.07 else "STRICT" if THRESHOLD <= 0.12 else "NORMAL" if THRESHOLD <= 0.18 else "LOOSE"
        print(f"\n📊 Threshold: {THRESHOLD:.3f} ({status})")

cap.release()
cv2.destroyAllWindows()
detector.close()
print("\n👋 Application closed")