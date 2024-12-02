import cv2
import os
import numpy as np

# Resim ve etiket dosyalarının yolu
image_path = "/home/samet/Desktop/yolo-seg/train/dataset/train/images/20.png"  # Resmin tam yolu
label_path = "/home/samet/Desktop/yolo-seg/train/dataset/train/labels/20.txt"  # Aynı isimli etiket dosyasının tam yolu

# Resmi yükle
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Resim bulunamadı: {image_path}")

# Etiket dosyasını aç ve verileri oku
if not os.path.exists(label_path):
    raise FileNotFoundError(f"Etiket dosyası bulunamadı: {label_path}")

with open(label_path, 'r') as f:
    lines = f.readlines()

# Görüntü boyutlarını al
height, width, _ = image.shape

# Etiket dosyasındaki her satırı işle
for line in lines:
    data = line.strip().split()
    class_id = int(data[0])  # Sınıf kimliği
    coordinates = list(map(float, data[1:]))  # Poligon koordinatları

    # Normalleştirilmiş koordinatları orijinal boyutlara dönüştür
    polygon = []
    for i in range(0, len(coordinates), 2):
        x = int(coordinates[i] * width)
        y = int(coordinates[i + 1] * height)
        polygon.append((x, y))

    # Poligonu çiz
    polygon = np.array(polygon, dtype=np.int32)
    cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    # Sınıf bilgisini poligonun ilk noktasına yaz
    cv2.putText(image, f"Class {class_id}", (polygon[0][0], polygon[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Görüntüyü göster ve kaydet
output_path = "path/to/output_39.jpg"
cv2.imshow("Segmented Image", image)
cv2.imwrite(output_path, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
