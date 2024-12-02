# Gerekli kütüphaneyi içe aktar
from ultralytics import YOLO
import cv2

# YOLOv8 modelini yükle
model = YOLO('/home/samet/Desktop/yolo-seg/train/segmentation_runs/yolo_segmentation4/weights/best.pt')  # 'yolov8n.pt' küçük, hızlı bir modeldir. 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt' gibi daha büyük modeller de mevcuttur.

# Resmi yükle
#image_path = '/home/samet/Desktop/yolo-seg/train/dataset/train/images/892.png'  # Burada analiz etmek istediğiniz resmin yolunu belirleyin.
# image_path = '/home/samet/Desktop/yolo-seg/train/dataset/train/images/842.jpg'  # Burada analiz etmek istediğiniz resmin yolunu belirleyin.
image_path = '/home/samet/Desktop/yolo-seg/personal_test/m.png'  # Burada analiz etmek istediğiniz resmin yolunu belirleyin.
image = cv2.imread(image_path)

# Obje tespiti yap
results = model(image)

# Sonuçları görselleştirme
annotated_frame = results[0].plot()  # Tespit edilen nesneleri çizer

# Çıktıyı göster
cv2.imshow('YOLOv8 Detection', annotated_frame)
cv2.waitKey(0)