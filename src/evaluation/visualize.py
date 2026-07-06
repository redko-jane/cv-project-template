import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

# --- ПУТИ (ИСПРАВЛЕНЫ) ---
image_dir = "D:/redko/practice/kitti/training/image_2"
label_dir = "D:/redko/practice/kitti/label_2"

# --- Находим первый попавшийся файл ---
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

if not image_files:
    print("❌ В папке нет изображений!")
    exit()

first_image = image_files[210]
image_path = os.path.join(image_dir, first_image)
label_path = os.path.join(label_dir, first_image.replace('.jpg', '.txt').replace('.png', '.txt'))

print(f"📂 Изображение: {image_path}")
print(f"📂 Аннотации: {label_path}")
print(f"📂 Файл аннотаций существует: {os.path.exists(label_path)}")

# --- Загружаем картинку ---
img = Image.open(image_path)

# --- Рисуем ---
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img)
ax.axis('off')

# --- Читаем аннотации и рисуем рамки ---
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        print(f"📂 Найдено строк в аннотациях: {len(lines)}")
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
                
            cls = parts[0]
            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])
            
            width = x2 - x1
            height = y2 - y1
            
            # Пропускаем слишком маленькие объекты
            if width < 10 or height < 10:
                continue
            
            print(f"  - {cls}: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, cls, color='red', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))
else:
    print("⚠️ Файл аннотаций не найден!")

plt.title(f'Пример размеченного изображения из датасета KITTI\n{first_image}', fontsize=14)
plt.savefig('results/plots/kitti_sample.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Скриншот сохранен в results/plots/kitti_sample.png")