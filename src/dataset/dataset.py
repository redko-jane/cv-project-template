import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

class KITTIDataset(Dataset):
    """Датасет KITTI для детекции объектов (2D bounding boxes)"""
    
    CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    NUM_CLASSES = 8
    
    def __init__(self, data_dir, split='training', transform=None, target_transform=None):
        """
        Args:
            data_dir (str): путь к папке с kitti (например, D:/redko/practice/kitti)
            split (str): 'training' или 'test'
            transform: трансформации для изображений
            target_transform: трансформации для аннотаций
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.image_dir = os.path.join(data_dir, split, 'image_2')
        self.label_dir = os.path.join(data_dir, split, 'label_2')
        
        self.image_files = [f for f in os.listdir(self.image_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()
        
        print(f"Загружено {len(self.image_files)} изображений из {split}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
        
        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        
        # Загружаем аннотации
        boxes, labels = self._load_annotations(label_path, width, height)
        
        if self.transform:
            image = self.transform(image)
        
        # Преобразуем в тензоры
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': img_name,
            'width': width,
            'height': height
        }
    
    def _load_annotations(self, label_path, width, height):
        """Загружает аннотации из KITTI label файла"""
        boxes = []
        labels = []
        
        if not os.path.exists(label_path):
            return boxes, labels
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                
                cls = parts[0]
                if cls not in self.CLASSES:
                    continue
                
                # KITTI формат: x1, y1, x2, y2 (пиксели)
                x1 = float(parts[4])
                y1 = float(parts[5])
                x2 = float(parts[6])
                y2 = float(parts[7])
                
                # Пропускаем слишком маленькие объекты
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                
                class_id = self.CLASSES.index(cls)
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
        
        return boxes, labels

    def get_class_names(self):
        return self.CLASSES