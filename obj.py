from ultralytics import YOLO

# Load pre-trained model and start training
model = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy
model.train(data='weapon.yaml', epochs=7, imgsz=416, batch=16)
