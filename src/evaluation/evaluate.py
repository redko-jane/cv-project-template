import os
import torch
import argparse
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import cv2

from src.dataset.dataset import KITTIDataset
from src.evaluation.metrics import calculate_metrics, compute_iou

# ============================================================
# ЗАГРУЗЧИКИ МОДЕЛЕЙ
# ============================================================

def load_yolo(model_path="models/weights/yolo_best.pt"):
    """Загружает обученную модель YOLO"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model
    except ImportError:
        print("Ultralytics не установлена!")
        return None

def load_faster_rcnn(model_path="models/weights/faster_rcnn.pth"):
    """Загружает обученную модель Faster R-CNN"""
    try:
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=9)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"Ошибка загрузки Faster R-CNN: {e}")
        return None

def load_ssd(model_path="models/weights/ssd.pth"):
    """Загружает обученную модель SSD"""
    try:
        from torchvision.models.detection import ssd300_vgg16
        from torchvision.models.detection.ssd import SSD300_VGG16_Weights
        
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"Ошибка загрузки SSD: {e}")
        return None

def load_efficientdet(model_path="models/weights/efficientdet.pth"):
    """Загружает обученную модель EfficientDet"""
    try:
        import torch.nn as nn
        import torchvision
        from efficientnet_pytorch import EfficientNet
        
        class SimpleEfficientDet(nn.Module):
            def __init__(self, num_classes=9):
                super().__init__()
                self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
                self.classifier = nn.Conv2d(1280, num_classes, kernel_size=1)
                self.bbox_regressor = nn.Conv2d(1280, 4, kernel_size=1)
                
            def forward(self, x):
                features = self.backbone.extract_features(x)
                classes = self.classifier(features)
                bboxes = self.bbox_regressor(features)
                return classes, bboxes
        
        model = SimpleEfficientDet(num_classes=9)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"Ошибка загрузки EfficientDet: {e}")
        return None

def load_detr(model_path="models/weights/detr.pth"):
    """Загружает обученную модель DETR"""
    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
        
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"Ошибка загрузки DETR: {e}")
        return None

# ============================================================
# ФУНКЦИИ ПРЕДСКАЗАНИЙ ДЛЯ КАЖДОЙ МОДЕЛИ
# ============================================================

def predict_yolo(model, image, conf_threshold=0.5):
    """YOLO предсказание"""
    results = model(image, conf=conf_threshold)
    boxes = []
    scores = []
    labels = []
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy().astype(int)
    
    return boxes, scores, labels

def predict_faster_rcnn(model, image_tensor, device):
    """Faster R-CNN предсказание"""
    with torch.no_grad():
        predictions = model([image_tensor.to(device)])
    
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    
    return boxes, scores, labels

def predict_ssd(model, image_tensor, device):
    """SSD предсказание"""
    with torch.no_grad():
        predictions = model([image_tensor.to(device)])
    
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    
    return boxes, scores, labels

def predict_efficientdet(model, image_tensor, device):
    """EfficientDet предсказание"""
    with torch.no_grad():
        classes, bboxes = model(image_tensor.to(device))
    
    # Получаем предсказания
    if isinstance(classes, tuple):
        classes = classes[0]
        bboxes = bboxes[0]
    
    # Преобразуем классы в вероятности
    scores = torch.softmax(classes, dim=1).cpu().numpy()
    labels = np.argmax(scores, axis=1)
    scores = np.max(scores, axis=1)
    boxes = bboxes.cpu().numpy()
    
    return boxes, scores, labels

def predict_detr(model, image_tensor, device):
    """DETR предсказание"""
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor.to(device))
    
    logits = outputs['logits'][0]
    boxes = outputs['pred_boxes'][0].cpu().numpy()
    probas = torch.softmax(logits, dim=-1).cpu().numpy()
    scores = np.max(probas, axis=1)
    labels = np.argmax(probas, axis=1)
    
    return boxes, scores, labels

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ ОЦЕНКИ
# ============================================================

def evaluate_model(model, dataloader, model_name, device, predict_func, conf_threshold=0.5):
    """Оценивает модель и возвращает метрики"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print(f"\nОценка {model_name} на {len(dataloader)} изображениях...")
    
    for batch in tqdm(dataloader, desc=f"Evaluating {model_name}"):
        images = batch['image']
        boxes = batch['boxes']
        labels = batch['labels']
        image_ids = batch.get('image_id', [])
        
        # Для каждой картинки в батче
        for idx in range(len(images)):
            img_tensor = images[idx].unsqueeze(0)
            gt_boxes = boxes[idx].cpu().numpy()
            gt_labels = labels[idx].cpu().numpy()
            
            # Предсказание
            pred_boxes, pred_scores, pred_labels = predict_func(
                model, img_tensor, device
            )
            
            # Фильтруем по порогу уверенности
            keep = pred_scores > conf_threshold
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
            
            all_predictions.append({
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels,
                'image': image_ids[idx] if image_ids else f"img_{idx}"
            })
            
            all_targets.append({
                'boxes': gt_boxes,
                'labels': gt_labels,
                'image': image_ids[idx] if image_ids else f"img_{idx}"
            })
    
    # Вычисляем метрики
    print(f"\nВычисление метрик для {model_name}...")
    metrics = calculate_metrics(all_predictions, all_targets, iou_threshold=0.5, num_classes=8)
    
    # Выводим результаты
    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ ОЦЕНКИ {model_name}")
    print(f"{'='*60}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-мера:    {metrics['f1']:.4f}")
    print(f"  mAP@0.5:    {metrics['map']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Оценка моделей детекции на KITTI")
    parser.add_argument('--model', type=str, 
                        choices=['yolo', 'faster', 'ssd', 'efficientdet', 'detr', 'all'],
                        help='Модель для оценки')
    parser.add_argument('--data_dir', type=str, default="D:/redko/practice/kitti",
                        help='Путь к датасету KITTI')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Количество изображений для оценки')
    args = parser.parse_args()
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загружаем датасет
    print(f"\nЗагрузка датасета из {args.data_dir}")
    dataset = KITTIDataset(args.data_dir, split='training')
    
    # Берем подвыборку для оценки
    if args.num_samples and args.num_samples < len(dataset):
        dataset.image_files = dataset.image_files[:args.num_samples]
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: {
            'image': torch.stack([b['image'] for b in batch]),
            'boxes': [b['boxes'] for b in batch],
            'labels': [b['labels'] for b in batch],
            'image_id': [b['image_id'] for b in batch]
        }
    )
    
    print(f"Оценка на {len(dataloader)} изображениях")
    
    models_to_evaluate = []
    if args.model == 'all':
        models_to_evaluate = ['yolo', 'faster', 'ssd', 'efficientdet', 'detr']
    else:
        models_to_evaluate = [args.model]
    
    results = {}
    
    for model_name in models_to_evaluate:
        print(f"\n{'='*60}")
        print(f"ЗАГРУЗКА {model_name.upper()}")
        print(f"{'='*60}")
        
        if model_name == 'yolo':
            model = load_yolo()
            predict_func = predict_yolo
        elif model_name == 'faster':
            model = load_faster_rcnn()
            predict_func = predict_faster_rcnn
        elif model_name == 'ssd':
            model = load_ssd()
            predict_func = predict_ssd
        elif model_name == 'efficientdet':
            model = load_efficientdet()
            predict_func = predict_efficientdet
        elif model_name == 'detr':
            model = load_detr()
            predict_func = predict_detr
        else:
            continue
        
        if model is None:
            print(f"Не удалось загрузить {model_name}")
            continue
        
        model.to(device)
        metrics = evaluate_model(model, dataloader, model_name.upper(), device, predict_func)
        results[model_name] = metrics
    
    # Выводим сводную таблицу
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("СВОДНАЯ ТАБЛИЦА МЕТРИК")
        print(f"{'='*70}")
        print(f"{'Модель':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'mAP@0.5':<12}")
        print(f"{'-'*70}")
        for name, metrics in results.items():
            print(f"{name.upper():<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['map']:<12.4f}")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()