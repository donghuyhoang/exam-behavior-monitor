import cv2
import numpy as np
# Nâng cấp cách import trực tiếp module thay vì gọi qua mp.solutions
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# Cài đặt MediaPipe
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=5, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

face_3d = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), 
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), 
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

class DummyDeformableDETR:
    def predict(self, frame):
        return [{"class": "phone", "bbox": [150, 200, 50, 80], "conf": 0.85}]

detr_model = DummyDeformableDETR()

video_path = "video_cctv_demo.mp4" # Đảm bảo file mp4 nằm cùng thư mục
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Không thể mở video tại đường dẫn: {video_path}")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Đã phát hết video hoặc lỗi đọc frame.")
        break
    
    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. Chạy AI phát hiện vật thể (Module 2 giả lập)
    detections = detr_model.predict(frame)
    for det in detections:
        x, y, bw, bh = det["bbox"]
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
        cv2.putText(frame, det["class"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 2. Chạy AI phát hiện tư thế đầu (Module 1)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_2d = []
            for idx in [1, 199, 33, 263, 61, 291]:
                lm = face_landmarks.landmark[idx]
                x_2d, y_2d = int(lm.x * w), int(lm.y * h)
                face_2d.append([x_2d, y_2d])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
            _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, None)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            yaw = angles[1] * 360
            
            # Đánh dấu Bounding Box bao quanh mặt
            x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
            
            if yaw < -20: 
                text = "Quay trai!"
                color = (0, 0, 255)
            elif yaw > 20: 
                text = "Quay phai!"
                color = (0, 0, 255)
            else: 
                text = "Hop le"
                color = (0, 255, 0)
                
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Demo He Thong', frame)
    if cv2.waitKey(30) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()