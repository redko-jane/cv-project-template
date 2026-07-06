import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
import os
import json
import cv2
import numpy as np
import time

# ============================================================
# 1. ДАТАСЕТ
# ============================================================

class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, ann_file):
        self.data_dir = data_dir
        self.img_size = (640, 384)
        
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        self.image_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_to_anns:
                self.image_to_anns[img_id] = []
            self.image_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.data_dir, 'training', 'image_2', img_info['file_name'])
        
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((375, 1242, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        new_w, new_h = self.img_size
        
        image = cv2.resize(image, (new_w, new_h))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        anns = self.image_to_anns.get(img_info['id'], [])
        
        boxes = []
        labels = []
        
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            x = x * scale_x
            y = y * scale_y
            w = w * scale_x
            h = h * scale_y
            
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'] + 1)
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return image, target


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================
# 2. ОБУЧЕНИЕ (БЕЗ ЗАМОРОЗКИ, 5 ЭПОХ)
# ============================================================

def train_faster_rcnn():
    print("=" * 60)
    print("ОБУЧЕНИЕ FASTER R-CNN (БЕЗ ЗАМОРОЗКИ, 5 ЭПОХ)")
    print("=" * 60)
    
    data_dir = "D:/redko/practice/kitti"
    ann_file = "D:/redko/practice/kitti_coco.json"
    
    if not os.path.exists(ann_file):
        print(f"Ошибка: файл {ann_file} не найден!")
        return
    
    dataset = KITTIDataset(data_dir, ann_file)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"Загружено {len(dataset)} изображений")
    print(f"Размер изображения: 640×384")
    
    print("\nСоздание модели...")
    backbone = torchvision.models.mobilenet_v2(weights='DEFAULT').features
    backbone.out_channels = 1280
    
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    model = FasterRCNN(
        backbone,
        num_classes=9,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=384,
        max_size=640
    )
    
    device = torch.device('cpu')
    model.to(device)
    
    # ===== БЕЗ ЗАМОРОЗКИ — ВСЕ СЛОИ ОБУЧАЮТСЯ =====
    print("Backbone НЕ заморожен — вся модель будет дообучаться")
    
    # Оптимизатор для ВСЕХ параметров
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.005, 
        momentum=0.9, 
        weight_decay=0.0005
    )
    
    num_epochs = 4
    print(f"\nНачинаем обучение на {num_epochs} эпох...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_loss_dict = {
            'loss_classifier': 0, 
            'loss_box_reg': 0, 
            'loss_rpn_box_reg': 0, 
            'loss_objectness': 0
        }
        
        for i, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            for k in epoch_loss_dict:
                epoch_loss_dict[k] += loss_dict[k].item()
            
            if (i + 1) % 10 == 0:
                progress = (i + 1) / len(dataloader) * 100
                print(f"  Эпоха {epoch+1}, шаг {i+1}/{len(dataloader)} ({progress:.1f}%), потери: {losses.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nЭпоха {epoch+1} завершена")
        print(f"  Средние потери: {avg_loss:.4f}")
        print(f"  Classifier: {epoch_loss_dict['loss_classifier']/len(dataloader):.4f}")
        print(f"  Box reg: {epoch_loss_dict['loss_box_reg']/len(dataloader):.4f}")
        print("=" * 60)
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    print(f"\nОбучение завершено за {hours}ч {minutes}мин")
    
    os.makedirs('runs/faster_rcnn', exist_ok=True)
    torch.save(model.state_dict(), 'runs/faster_rcnn/model.pth')
    print("Модель сохранена в runs/faster_rcnn/model.pth")
    
    return model


if __name__ == "__main__":
    train_faster_rcnn()