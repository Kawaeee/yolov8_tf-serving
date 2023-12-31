import argparse

from ultralytics import YOLO
import torch


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 model exporter")
    parser.add_argument("-i", "--model-path", type=str, required=True, help="Path to the YOLOv8 PyTorch model weight file")
    parser.add_argument("-f", "--model-format", type=str, default="onnx", help="Export format (default: onnx)")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.model_path)
    model.export(format=args.model_format, device=device)


if __name__ == "__main__":
    main()
