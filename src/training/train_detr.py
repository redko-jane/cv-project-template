import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torchvision.transforms as T

# --- Создаем папку для результатов ---
os.makedirs("results/plots", exist_ok=True)

# --- Загружаем DETR через torch.hub (официальный репозиторий Facebook) ---
print("Загружаю DETR из официального репозитория Facebook...")
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
model.eval()
print("✅ Модель загружена!")

# --- Загружаем картинку из KITTI ---
image_path = "D:/redko/practice/kitti/training/image_2/000000.jpg"

if not os.path.exists(image_path):
    print(f"❌ Файл не найден: {image_path}")
    print("Проверь путь к картинке!")
    exit()

image = Image.open(image_path).convert("RGB")

# --- Подготовка изображения для модели ---
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = transform(image).unsqueeze(0)

# --- Запускаем детекцию ---
with torch.no_grad():
    outputs = model(img)

# --- Извлекаем результаты ---
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # убираем класс "no object"
keep = probas.max(-1).values > 0.7  # порог уверенности

boxes = outputs['pred_boxes'][0, keep].cpu().numpy()
scores = probas[keep].max(-1).values.cpu().numpy()
labels = probas[keep].argmax(-1).cpu().numpy()

# --- COCO классы ---
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# --- Визуализация ---
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)
ax.set_title("DETR (предобученная на COCO) на KITTI", fontsize=14)

# Конвертируем координаты из [0,1] в пиксели
w, h = image.size
detected_classes = []

for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
    
    class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)
    detected_classes.append(f"{class_name} ({score:.2f})")
    
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-10, f'{class_name}: {score:.2f}', color='white', fontsize=10,
            bbox=dict(facecolor='red', alpha=0.7, pad=2))

print(f"Обнаружено объектов: {len(detected_classes)}")
print(f"Классы: {', '.join(detected_classes) if detected_classes else 'нет объектов'}")

plt.tight_layout()
plt.savefig('results/plots/detr_detection.png', dpi=150, bbox_inches='tight')
print("✅ Результат сохранен в results/plots/detr_detection.png")
plt.show()
print("\n🎉 Готово! DETR успешно запущен.")
