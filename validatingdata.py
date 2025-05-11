from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run validation to see metrics (on the validation set defined in weapon.yaml)
# Run inference on test images
metrics = model.val(data='weapon.yaml', split='test')

