# a simple script using ultralytics to train my YOLOv11n model with the labelled images in the merge_detect folder
import os
from ultralytics import YOLO

#gets absolute path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
#select the dataset we want to train on
data_path = os.path.join(script_dir, 'augmented_dataset', 'data.yaml')

# Choosing model variety
model = YOLO('yolo11n.pt')

if __name__ == '__main__':
    # Train the model (ultralytics handles the training process)
    model.train(data=data_path, epochs=100, imgsz=640, batch=16, device='0')

    # Validate after training (only on val as defined in data.yaml)
    model.val(data=data_path, imgsz=640, batch=16, device='0')

    #test the model on the test dataset 
    print("testing the model on the test dataset...")
    model.val(data=data_path, split='test', imgsz=640, batch=16, device='0')