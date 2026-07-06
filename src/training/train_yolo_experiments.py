from ultralytics import YOLO
import os

# --- Пути ---
DATA_YAML = "D:/redko/practice/kitti_yolo_fixed/data.yaml"
EPOCHS = 4
BATCH = 4

# --- Эксперимент 1: lr = 0.01 ---
print("=" * 60)
print("🚀 Эксперимент 1: lr = 0.01, batch = 4")
print("=" * 60)

model = YOLO("yolov8n.pt")
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH,
    lr0=0.01,
    device="cpu",
    project="experiments/yolo_lr_001",
    name="exp",
    exist_ok=True,
)

# --- Эксперимент 2: lr = 0.001 (уже есть, но переобучим для чистоты) ---
print("\n" + "=" * 60)
print("🚀 Эксперимент 2: lr = 0.001, batch = 4")
print("=" * 60)

model = YOLO("yolov8n.pt")
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH,
    lr0=0.001,
    device="cpu",
    project="experiments/yolo_lr_0001",
    name="exp",
    exist_ok=True,
)

# --- Эксперимент 3: lr = 0.0001 ---
print("\n" + "=" * 60)
print("🚀 Эксперимент 3: lr = 0.0001, batch = 4")
print("=" * 60)

model = YOLO("yolov8n.pt")
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH,
    lr0=0.0001,
    device="cpu",
    project="experiments/yolo_lr_00001",
    name="exp",
    exist_ok=True,
)

print("\n✅ Все эксперименты завершены!")
print("📁 Веса сохранены в папках:")
print("   - experiments/yolo_lr_001/exp/weights/best.pt")
print("   - experiments/yolo_lr_0001/exp/weights/best.pt")
print("   - experiments/yolo_lr_00001/exp/weights/best.pt")