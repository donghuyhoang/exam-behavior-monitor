import cv2
import os
import urllib.request
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# 3D Head Pose Model Reference Points
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (Landmark 1)
    (0.0, -330.0, -65.0),        # Chin (Landmark 152)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (Landmark 33)
    (225.0, 170.0, -135.0),      # Right eye outer corner (Landmark 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (Landmark 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (Landmark 291)
], dtype=np.float64)

def get_head_pose(face_landmarks, img_w, img_h):
    landmark_ids = [1, 152, 33, 263, 61, 291]
    image_points = np.array([
        (face_landmarks[idx].x * img_w, face_landmarks[idx].y * img_h)
        for idx in landmark_ids
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return "UNKNOWN", 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    x = float(euler_angles[0].item())
    y = float(euler_angles[1].item())
    z = float(euler_angles[2].item())

    # Chuẩn hóa góc Pitch (X) về quanh 0 độ (do trục Y của mô hình 3D ngược với camera OpenCV)
    if x > 90:
        x = x - 180
    elif x < -90:
        x = x + 180

   
    if x > 10:
        direction = "DOWN"
    elif x < -10:
        direction = "UP"
    elif y < -15:
        direction = "TURN LEFT"
    elif y > 15:
        direction = "TURN RIGHT"
    else:
        direction = "CENTER"

    return direction, x, y, z

def detect_and_draw_hand(img, hand_detector):
    """
    Phát hiện bàn tay và vẽ landmark bàn tay lên khung hình.
    
    Args:
        img: Ảnh đầu vào BGR từ OpenCV (được vẽ trực tiếp).
        hand_detector: Đối tượng MediaPipe HandLandmarker.
        
    Returns:
        is_present (bool): True nếu phát hiện bàn tay.
        detection_result: Kết quả nhận diện từ MediaPipe HandLandmarker.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = hand_detector.detect(mp_image)
    
    is_present = len(detection_result.hand_landmarks) > 0
    if is_present:
        h, w, _ = img.shape
        for hand_landmarks in detection_result.hand_landmarks:
            # Vẽ các đường nối giữa điểm landmark bàn tay
            for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                pt1_lm = hand_landmarks[conn.start]
                pt2_lm = hand_landmarks[conn.end]
                pt1 = (int(pt1_lm.x * w), int(pt1_lm.y * h))
                pt2 = (int(pt2_lm.x * w), int(pt2_lm.y * h))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            # Vẽ các điểm landmark
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                
    return is_present, detection_result


def detect_and_draw_face(img, face_detector, face_state):
    """
    Phát hiện khuôn mặt, vẽ landmark, hiển thị góc x,y,z và theo dõi hành vi gian lận (quay đầu quá 5s).
    Phân biệt giữa gian lận tĩnh và xoay cổ/thư giãn bằng độ biến động góc quay (variance).
    
    Args:
        img: Ảnh đầu vào BGR từ OpenCV.
        face_detector: Đối tượng MediaPipe FaceLandmarker.
        face_state: Dictionary lưu trạng thái thời gian và lịch sử góc quay.
        
    Returns:
        is_present (bool): True nếu phát hiện khuôn mặt.
        detection_result: Kết quả nhận diện từ MediaPipe FaceLandmarker.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = face_detector.detect(mp_image)
    
    is_present = len(detection_result.face_landmarks) > 0
    if is_present:
        h, w, _ = img.shape
        for face_landmarks in detection_result.face_landmarks:
            # Tính hướng quay đầu và góc euler x, y, z
            direction, x, y, z = get_head_pose(face_landmarks, w, h)
            
            suspicious_directions = ["TURN LEFT", "TURN RIGHT", "DOWN"]
            current_time = time.time()
            
            if direction in suspicious_directions:
                # Nếu đổi hướng hoặc vừa bắt đầu quay nghi ngờ -> reset timer
                if face_state["last_direction"] != direction or face_state["start_time"] is None:
                    face_state["start_time"] = current_time
                    face_state["last_direction"] = direction
                    face_state["alert_logged"] = False

                #face_state["angle_history"].append((x, y, z))
                elapsed_time = current_time - face_state["start_time"]
                
                # Tính độ biến động chuyển động (standard deviation) để kiểm tra lắc/xoay cổ
                # Kiểm tra ngưỡng gian lận 5 giây
                if elapsed_time >= 5.0 :
                    color = (0, 0, 255) # Đỏ
                    status_text = f"ALERT: CHEATING DETECTED! ({direction} > 5s)"
                    
                    # In thông báo ra terminal 1 lần mỗi đợt gian lận
                    if not face_state["alert_logged"]:
                        print(f"⚠️ [CẢNH BÁO GIAN LẬN] Phát hiện nhìn hướng {direction} tĩnh trong {elapsed_time:.1f} giây!")
                        face_state["alert_logged"] = True
                    
                    # Vẽ khung cảnh báo đỏ nổi bật lên toàn bộ ảnh
                    cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 8)
                    cv2.putText(img, status_text, (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    color = (0, 255, 255) # Vàng
                    
                    status_text = f"Timer: {elapsed_time:.1f}s / 5.0s ({direction})"
                    
                    cv2.putText(img, status_text, (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                # Trở về CENTER hoặc UP -> Reset toàn bộ timer
                face_state["start_time"] = None
                face_state["last_direction"] = direction

                face_state["alert_logged"] = False
                color = (0, 255, 0) # Xanh lá
            
            # Hiển thị thông tin cơ bản lên màn hình
            cv2.putText(img, f"Head: {direction}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(img, f"X (Pitch): {x:.1f} deg", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Y (Yaw):   {y:.1f} deg", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Z (Roll):  {z:.1f} deg", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            # Vẽ các điểm landmark khuôn mặt
            for lm in face_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 1, (255, 255, 255), cv2.FILLED)
                
    return is_present, detection_result


# 1. Tự động tải file nhận diện của Google (Chỉ tải 1 lần)
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Đang tải file hệ thống bàn tay... Vui lòng đợi vài giây...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", MODEL_PATH)

FaceModel_Path = "face_landmarker.task"
if not os.path.exists(FaceModel_Path):
    print("Đang tải file hệ thống khuôn mặt... Vui lòng đợi vài giây...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", FaceModel_Path)

# 2. Cấu hình AI
hand_base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

face_base_options = python.BaseOptions(model_asset_path=FaceModel_Path)
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    num_faces=1,
    min_face_detection_confidence=0.5
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

# Trạng thái quản lý gian lận theo thời gian
face_state = {
    "start_time": None,
    "last_direction": "CENTER",
    "alert_logged": False
}

# 3. Khởi động Camera (Dùng CAP_DSHOW để chống treo trên Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
hand_was_present = False
face_was_present = False
print("Đang bật Camera... Bấm phím 'q' để thoát.")

while cap.isOpened():
    success, img = cap.read()
    if not success or img is None:
        break

    # 4. Phát hiện & vẽ landmark bàn tay và khuôn mặt bằng 2 hàm chuyên biệt
    hand_currently_present, hand_result = detect_and_draw_hand(img, hand_detector)
    face_currently_present, face_result = detect_and_draw_face(img, face_detector, face_state)

    # 5. Logic in thông báo 1 lần duy nhất khi xuất hiện
    if hand_currently_present and not hand_was_present:
        print("=> THÔNG BÁO: Hand detected!")
    hand_was_present = hand_currently_present

    if face_currently_present and not face_was_present:
        print("=> THÔNG BÁO: Face detected!")
    face_was_present = face_currently_present

    cv2.imshow("He thong giam sat", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()