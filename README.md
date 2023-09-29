# yolov8_tf-serving

`yolov8_tf-serving` is a project designed to convert YOLOv8 models into a format compatible with TensorFlow Serving, enabling seamless deployment of these models in production environments. 

## Getting Started

* Clone the repository
```bash
git clone https://github.com/Kawaeee/yolov8_tf-serving.git
cd yolov8_tf-serving/
```

* Build Docker image
```bash
docker build -t yolov8conv .
```

* Access Docker container bash shell
```bash
# CPU
docker run -it -v $(pwd):/data --rm yolov8conv /bin/bash

# GPU
docker run -it -v $(pwd):/data --gpus all --rm yolov8conv /bin/bash
```

* Run `run.sh` with .pt model file in mounted directory
```bash
bash /app/run.sh <yolov8-model.pt>
```

```bash
# Example:
wget -O /data/yolov8l.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
bash /app/run.sh /data/yolov8l.pt
```

* After obtaining the converted model, transfer the contents from the "output" directory to the "demo/models". Then, simply follow the instructions outlined in the [next steps](https://github.com/Kawaeee/yolov8_tf-serving/blob/main/demo/README.md).
