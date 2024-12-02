import json
import os

def normalize_coordinates(points, img_width, img_height):
    """Koordinatları normalize eder (0-1 arası değerlere çevirir)."""
    normalized = []
    for i in range(0, len(points), 2):
        x = points[i] / img_width
        y = points[i + 1] / img_height
        normalized.extend([x, y])
    return normalized

def coco_to_yolo_segmentation(coco_json_path, output_dir):
    # COCO JSON dosyasını yükle
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Kategori ID'lerini ve isimlerini al
    category_map = {category['id']: idx for idx, category in enumerate(coco_data['categories'])}
    
    # Görselleri id ile eşleştir ve boyut bilgilerini sakla
    image_map = {image['id']: (image['file_name'].rsplit('.', 1)[0], image['width'], image['height']) 
                 for image in coco_data['images']}
    
    # Çıkış dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Tüm anotasyonları işle
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']
        
        # Görüntü bilgilerini al
        image_name, img_width, img_height = image_map[image_id]
        txt_file_path = os.path.join(output_dir, f"{image_name}.txt")
        
        # YOLO formatında sınıf etiketi
        class_id = category_map[category_id]
        
        # Her segmentasyon poligonunu işlemeye başla
        with open(txt_file_path, 'a') as f:
            for segment in segmentation:
                # Segmentasyon koordinatlarını normalize et
                normalized_segment = normalize_coordinates(segment, img_width, img_height)
                segment_str = ' '.join(map(str, normalized_segment))
                f.write(f"{class_id} {segment_str}\n")

    print(f"YOLO formatındaki etiketler {output_dir} dizinine kaydedildi.")

# COCO JSON dosyasının yolu ve çıktı dizini
coco_json_path = "/home/samet/Desktop/yolo-seg/yolo_labels2/val_labels.json"  # Güncel JSON dosya yolu
output_dir = "/home/samet/Desktop/yolo-seg/yolo_labels2/labels"  # Çıkış dizini

# Dönüştürme işlemini başlat
coco_to_yolo_segmentation(coco_json_path, output_dir)
