import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükleme
model = YOLO('/home/samet/Desktop/yolo-seg/training/segmentation_runs/yolo_segmentation/weights/best.pt')  # 'yolov8n.pt', 'yolov8s.pt' gibi seçenekler kullanılabilir

# Video kaynağı (Dosya veya Kamera)
video_path = '/home/samet/Desktop/yolo-seg/personal_test/videoplayback.mp4'  # Videonun yolunu belirleyin
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 ile tespit yapma
    results = model(frame)

    # Sonuçları görselleştirme
    annotated_frame = results[0].plot()  # Tespit edilen nesneleri çizer

    # Çıktıyı göster
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
