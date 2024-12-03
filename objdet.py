import torch
from ultralytics import YOLO

# Check and set the device (MPS for M2, CPU fallback)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Nano model for limited resources

# Train the model
model.train(
    data="/Users/yaya/Desktop/Grad Project/Hcrowd/Human Crowd.v1i.yolov8/data.yaml",  # Dataset configuration
    epochs=10,                  # Start with 10 epochs, increase if stable
    batch=8,                    # Lower batch size for 8GB RAM
    imgsz=640,                  # Standard image size
    workers=0,                  # Use single-threaded data loading
                # Use MPS or CPU
    name="crowd_project"    # Training run name
)

# Validate the model
metrics = model.val(device=device)  # Perform validation using MPS

# Print validation metrics
print(metrics)

# Save the trained model
model.save("/Users/yaya/Desktop/Grad Project/Hcrowd/Hcrowded_project.pt")
