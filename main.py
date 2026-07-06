import argparse
import yaml
from src.models.yolo import train_yolo
from src.models.faster_rcnn import train_faster_rcnn
from src.models.ssd import train_ssd
from src.models.efficientdet import train_efficientdet
from src.models.detr import train_detr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['yolo', 'faster', 'ssd', 'efficientdet', 'detr', 'all'], 
                        help='Model to train')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.model == 'yolo' or args.model == 'all':
        train_yolo(config)
    if args.model == 'faster' or args.model == 'all':
        train_faster_rcnn(config)
    if args.model == 'ssd' or args.model == 'all':
        train_ssd(config)
    if args.model == 'efficientdet' or args.model == 'all':
        train_efficientdet(config)
    if args.model == 'detr' or args.model == 'all':
        train_detr(config)

if __name__ == '__main__':
    main()