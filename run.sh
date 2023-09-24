#!/bin/bash

# Define the work directory
WORK_DIRECTORY="runs/$(echo $RANDOM | md5sum | head -c 8; echo;)/"
INPUT_MODEL_PATH="${1}"
INPUT_BASE_NAME=$(basename "${INPUT_MODEL_PATH}" .pt)

# Create the work directory if it doesn't exist
mkdir -p "${WORK_DIRECTORY}"

# Copy the input model to the work directory
cp -r "${INPUT_MODEL_PATH}" "${WORK_DIRECTORY}"

# Find PyTorch model files in the work directory
PYTORCH_MODEL_PATH=$(ls "${WORK_DIRECTORY}"*.pt)

# Export PyTorch model
python scripts/export.py -i "${PYTORCH_MODEL_PATH}"

# Find ONNX model files in the work directory
ONNX_MODEL_PATH=$(ls "${WORK_DIRECTORY}"*.onnx)

# Convert ONNX model to TensorFlow
bash scripts/convert.sh "${ONNX_MODEL_PATH}"

# Define TensorFlow model paths
TF_MODEL_PATH=$(echo "${WORK_DIRECTORY%/*}/${INPUT_BASE_NAME}")
CUSTOMIZED_TF_MODEL_PATH=$(echo "${WORK_DIRECTORY%/*}/output")

# Customize TensorFlow model
python scripts/customize.py -i "${TF_MODEL_PATH}" -o "${CUSTOMIZED_TF_MODEL_PATH}"
