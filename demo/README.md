# yolov8_tf-serving (demo)

In this demonstration, we will utilize a converted YOLOv8l model obtained from the previous example.

## Getting Started

* Running tensorflow-serving docker container
```bash
# CPU
docker run -it -p 8501:8501 -p 8500:8500 \
--mount type=bind,source=$PWD/models/output/,target=/models/output/ \
--mount type=bind,source=$PWD/models/models.config,target=/models/models.config \
-t tensorflow/serving:latest --model_config_file=/models/models.config

# GPU
docker run --rm --gpus all -p 8501:8501 -p 8500:8500 \
--mount type=bind,source=$PWD/models/output/,target=/models/output/ \
--mount type=bind,source=$PWD/models/models.config,target=/models/models.config \
-t tensorflow/serving-gpu:latest --model_config_file=/models/models.config
```

* After setting up the container, you can check the provided Jupyter notebooks to obtain prediction results.