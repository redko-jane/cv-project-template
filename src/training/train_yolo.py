from ultralytics import YOLO

print("=" * 60)
print("ОБУЧЕНИЕ YOLOv8 НА ПОЛНОМ ДАТАСЕТЕ KITTI (8 ЭПОХ)")
print("=" * 60)

# Загружаем модель
model = YOLO('yolov8n.pt')
print("Модель загружена")

# Обучаем на ВСЕХ данных
results = model.train(
    data='kitti_yolo_fixed.yaml',
    epochs=5,            # 8 эпох для баланса времени и качества
    batch=8,            
    imgsz=640,          
    device='cpu',       
    project='runs/yolo',
    name='kitti_full',
    exist_ok=True,
    verbose=True,
    patience=20,        
    save=True,          
    plots=True          
)

print("\n" + "=" * 60)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print(f"Результаты сохранены в: runs/yolo/kitti_full")
print("=" * 60)

# Оценка модели
print("\nОЦЕНКА МОДЕЛИ НА ВАЛИДАЦИИ:")
metrics = model.val()
print(f"  mAP@0.5: {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall: {metrics.box.mr:.4f}")