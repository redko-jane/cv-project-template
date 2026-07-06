# C:\users\nargi\practice\train_ssd.py
import os
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================ ВАШ ТЕКУЩИЙ ПУТЬ ================
BASE_DIR = Path(r'C:\users\nargi\practice')
KITTI_DIR = BASE_DIR / 'data' / 'raw' / 'kitti'

IMAGES_DIR = KITTI_DIR / 'training' / 'image_2'
LABELS_DIR = KITTI_DIR / 'label_2'

PROCESSED_DIR = BASE_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("🚀 SSD TRAINING - KITTI (ВСЕ 8 КЛАССОВ)")
print("="*60)

# ================ 8 КЛАССОВ ================
ALL_CLASSES = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 
               'Person_sitting', 'Tram', 'Misc']
CLASS_TO_ID = {cls: i for i, cls in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)

print(f"\n📊 Количество классов: {NUM_CLASSES}")
print(f"   Классы: {ALL_CLASSES}")

# ================ ПАРСИНГ ================

def parse_kitti_label(label_path):
    bboxes = []
    classes = []
    
    if not os.path.exists(label_path):
        return [], []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            
            cls = parts[0]
            if cls in ALL_CLASSES:
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])
                
                if x1 < x2 and y1 < y2:
                    bboxes.append([x1, y1, x2, y2])
                    classes.append(cls)
    
    return bboxes, classes

def prepare_kitti_data():
    print("\n📂 Подготовка данных...")
    
    data = []
    image_files = list(IMAGES_DIR.glob('*.png'))
    
    class_counts = {cls: 0 for cls in ALL_CLASSES}
    
    for img_path in tqdm(image_files, desc="Обработка"):
        label_path = LABELS_DIR / f'{img_path.stem}.txt'
        bboxes, classes = parse_kitti_label(label_path)
        
        if len(bboxes) > 0:
            data.append({
                'image_path': str(img_path),
                'bboxes': bboxes,
                'classes': classes
            })
            for cls in classes:
                if cls in class_counts:
                    class_counts[cls] += 1
    
    print(f"\n✅ Подготовлено {len(data)} изображений")
    print("\n📊 Распределение объектов по классам:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    if len(data) == 0:
        return None, None, None
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    
    for name, dataset in [('train', train), ('val', val), ('test', test)]:
        with open(PROCESSED_DIR / f'{name}.json', 'w') as f:
            json.dump(dataset, f)
        print(f"  {name}: {len(dataset)}")
    
    return train, val, test

# ================ DATASET ================

class KITTIDataset(Dataset):
    def __init__(self, data_file, img_size=512):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.img_size = img_size
        self.class_map = CLASS_TO_ID
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        img = cv2.imread(item['image_path'])
        if img is None:
            return {
                'image': torch.zeros(3, self.img_size, self.img_size),
                'boxes': torch.zeros((0, 4)),
                'labels': torch.zeros((0,), dtype=torch.long)
            }
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        scale_x = self.img_size / w
        scale_y = self.img_size / h
        
        boxes = []
        labels = []
        
        for bbox, cls in zip(item['bboxes'], item['classes']):
            x1 = bbox[0] * scale_x
            y1 = bbox[1] * scale_y
            x2 = bbox[2] * scale_x
            y2 = bbox[3] * scale_y
            
            if x1 < x2 and y1 < y2:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_map[cls])
        
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = np.transpose(img, (2, 0, 1))
        
        return {
            'image': torch.FloatTensor(img),
            'boxes': torch.FloatTensor(boxes) if boxes else torch.zeros((0, 4)),
            'labels': torch.LongTensor(labels) if labels else torch.zeros((0,), dtype=torch.long)
        }

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    targets = []
    for i in range(len(images)):
        target = {}
        if len(boxes[i]) > 0:
            target['boxes'] = boxes[i]
            target['labels'] = labels[i] + 1
        else:
            target['boxes'] = torch.zeros((0, 4))
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
        targets.append(target)
    
    return images, targets

# ================ МОДЕЛЬ ================

def replace_ssd_head(model, num_classes):
    in_channels_list = []
    for module in model.head.classification_head.module_list:
        in_channels_list.append(module.in_channels)
    
    num_anchors = [6] * len(in_channels_list)
    num_classes_with_bg = num_classes + 1
    
    new_head = SSDClassificationHead(
        in_channels=in_channels_list,
        num_anchors=num_anchors,
        num_classes=num_classes_with_bg
    )
    
    model.head.classification_head = new_head
    return model

# ================ ОБУЧЕНИЕ ================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    valid_batches = 0
    
    for images, targets in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        
        for target in targets:
            if len(target['boxes']) > 0:
                target['boxes'] = target['boxes'].to(device)
                target['labels'] = target['labels'].to(device)
            else:
                target['boxes'] = torch.zeros((0, 4), device=device)
                target['labels'] = torch.zeros((0,), dtype=torch.int64, device=device)
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        valid_batches += 1
    
    return total_loss / valid_batches if valid_batches > 0 else 0

def evaluate(model, dataloader, device):
    """Только сбор loss без постпроцессинга"""
    model.eval()
    total_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            
            for target in targets:
                if len(target['boxes']) > 0:
                    target['boxes'] = target['boxes'].to(device)
                    target['labels'] = target['labels'].to(device)
                else:
                    target['boxes'] = torch.zeros((0, 4), device=device)
                    target['labels'] = torch.zeros((0,), dtype=torch.int64, device=device)
            
            # Для валидации используем только forward без постпроцессинга
            try:
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                total_loss += loss.item()
                valid_batches += 1
            except:
                # Если ошибка - пропускаем этот батч
                continue
    
    return total_loss / valid_batches if valid_batches > 0 else float('inf')

# ================ MAIN ================

def main():
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        print("\n❌ ОШИБКА СТРУКТУРЫ!")
        return
    
    train_json = PROCESSED_DIR / 'train.json'
    if not train_json.exists():
        prepare_kitti_data()
    else:
        print("\n✅ Данные уже подготовлены")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n💻 Device: {device}")
    
    print("\n📦 Загрузка SSD...")
    model = ssd300_vgg16(pretrained=True)
    model = replace_ssd_head(model, NUM_CLASSES)
    model.to(device)
    print(f"   Классов: {NUM_CLASSES} + background = {NUM_CLASSES + 1}")
    
    train_dataset = KITTIDataset(PROCESSED_DIR / 'train.json')
    val_dataset = KITTIDataset(PROCESSED_DIR / 'val.json')
    
    batch_size = 2 if device == 'cpu' else 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"📊 Train: {len(train_dataset)}")
    print(f"📊 Val: {len(val_dataset)}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("\n" + "="*60)
    print("🏋️ ОБУЧЕНИЕ (4 ЭПОХИ, 8 КЛАССОВ)")
    print("="*60)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(4):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/4")
        print(f"{'='*50}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"\n📊 Train Loss: {train_loss:.4f}")
        
        # Валидация
        val_loss = evaluate(model, val_loader, device)
        print(f"📊 Val Loss:   {val_loss:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, RESULTS_DIR / f'ssd_epoch_{epoch+1}.pth')
        print(f"💾 Сохранено: ssd_epoch_{epoch+1}.pth")
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', marker='o')
        plt.plot(val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'SSD Training on KITTI ({NUM_CLASSES} classes)')
        plt.legend()
        plt.grid(True)
        plt.savefig(RESULTS_DIR / 'training_plot.png')
        plt.close()
    
    print("\n" + "="*60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print(f"\n📊 Итоговые результаты:")
    print(f"   Train Loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    print(f"   Val Loss:   {val_losses[0]:.4f} → {val_losses[-1]:.4f}")

if __name__ == "__main__":
    main()