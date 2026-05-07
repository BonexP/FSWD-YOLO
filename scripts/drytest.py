import torch
# from models.yolo import Model # Adjust import path if necessary
from ultralytics import YOLO

if __name__ == "__main__":
    # Load the new model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    custom_yaml= 'ultralytics/cfg/models/11/yolo11s_CBAM.yaml'

    model = YOLO(custom_yaml).to(device)
    model.eval()

    # Print model summary to check layers
    print(model)

    # Test forward pass
    try:
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        output = model(dummy_input)
        print("Model built and forward pass successful!")
    except Exception as e:
        print(f"An error occurred: {e}")
