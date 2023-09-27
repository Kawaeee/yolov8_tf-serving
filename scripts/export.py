import argparse

from ultralytics import YOLO
import torch


def export(model_path: str, model_format: str = "onnx"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    model.export(format=model_format, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 model exporter")
    parser.add_argument("-i", "--model-path", type=str, required=True, help="Path to the YOLOv8 PyTorch model weight file")
    parser.add_argument("-f", "--model-format", type=str, default="onnx", help="Export format (default: onnx)")

    args = parser.parse_args()
    export(model_path=args.model_path, model_format=args.model_format)
