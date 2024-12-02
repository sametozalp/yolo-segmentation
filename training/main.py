from ultralytics import YOLO

# Modeli yükle
model = YOLO("yolov8n-seg.pt")  # YOLOv8 segmentasyon modelinin başlangıç ağırlıkları

# Modeli eğit
model.train(
    data="/home/samet/Desktop/yolo-seg/training/data.yaml",           # Veri kümesi yapılandırması
    epochs=100,                  # Eğitim için epoch sayısı
    batch=16,                   # Batch boyutu
    imgsz=640,                  # Görüntü boyutu
    workers=4,                  # Paralel iş parçacığı sayısı
    name="yolo_segmentation",   # Eğitimden sonra oluşturulacak model adı
    project="/home/samet/Desktop/yolo-seg/training/segmentation_runs", # Eğitim sonuçlarının kaydedileceği proje dizini
    device=0                    # GPU kullanımı (CPU için "cpu")
)

# Eğitim tamamlandıktan sonra sonuçları kontrol edin
print("Eğitim tamamlandı. Modeller 'segmentation_runs' dizininde saklandı.")
