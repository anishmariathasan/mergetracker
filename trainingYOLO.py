# a simple script to train my YOLOv11n model with the labelled images in the merge_detect folder
import os
from ultralytics import YOLO

#gets absolute path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'merge_detect', 'data.yaml')

model = YOLO('yolo11n.pt')

if __name__ == '__main__':
    # Train the model
    model.train(data=data_path, epochs=60, imgsz=640, batch=16, device='0')

    # Validate after training (includes val and test defined in data.yaml)
    model.val(data=data_path, imgsz=640, batch=16, device='0')
