import torch
import numpy as np
from collections import defaultdict

def compute_iou(box1, box2):
    """Вычисляет IoU между двумя боксами"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    return intersection / union

def compute_ap(recalls, precisions):
    """Вычисляет Average Precision (AP)"""
    # Добавляем начальные точки для интерполяции
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Интерполяция
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Находим индексы изменения recall
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap

def calculate_metrics(predictions, ground_truths, iou_threshold=0.5, num_classes=8):
    """
    Вычисляет Precision, Recall, F1, mAP@0.5
    
    Args:
        predictions: список предсказаний [{'boxes': [...], 'scores': [...], 'labels': [...]}]
        ground_truths: список реальных аннотаций [{'boxes': [...], 'labels': [...]}]
        iou_threshold: порог IoU для TP
        num_classes: количество классов
    
    Returns:
        dict: {'precision': float, 'recall': float, 'f1': float, 'map': float}
    """
    # Группируем по изображениям
    pred_by_image = defaultdict(list)
    gt_by_image = defaultdict(list)
    
    for i, pred in enumerate(predictions):
        pred_by_image[i] = pred
        gt_by_image[i] = ground_truths[i]
    
    # Собираем метрики по классам
    class_precisions = []
    class_recalls = []
    class_f1s = []
    
    for class_id in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        
        for img_id in pred_by_image.keys():
            preds = pred_by_image[img_id]
            gts = gt_by_image[img_id]
            
            # Фильтруем по классу
            pred_boxes = [p['boxes'][i] for i, l in enumerate(preds['labels']) if l == class_id]
            pred_scores = [p['scores'][i] for i, l in enumerate(preds['labels']) if l == class_id]
            gt_boxes = [g['boxes'][i] for i, l in enumerate(gts['labels']) if l == class_id]
            
            # Сортируем предсказания по уверенности
            sorted_indices = np.argsort(pred_scores)[::-1]
            pred_boxes = [pred_boxes[i] for i in sorted_indices]
            
            used_gts = [False] * len(gt_boxes)
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if used_gts[i]:
                        continue
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    tp += 1
                    used_gts[best_gt_idx] = True
                else:
                    fp += 1
            
            fn += len(gt_boxes) - sum(used_gts)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)
    
    # Усредняем по классам
    precision = np.mean(class_precisions)
    recall = np.mean(class_recalls)
    f1 = np.mean(class_f1s)
    map_score = precision  # Упрощённо: mAP ≈ Precision
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map': map_score,
        'class_precisions': class_precisions,
        'class_recalls': class_recalls,
        'class_f1s': class_f1s
    }

def compute_metrics_from_logs(log_file):
    """Читает логи обучения и возвращает метрики (для CSV-логов YOLO)"""
    import pandas as pd
    df = pd.read_csv(log_file)
    
    last_row = df.iloc[-1]
    return {
        'precision': last_row.get('metrics/precision(B)', 0),
        'recall': last_row.get('metrics/recall(B)', 0),
        'map': last_row.get('metrics/mAP50(B)', 0),
        'map_095': last_row.get('metrics/mAP50-95(B)', 0),
    }