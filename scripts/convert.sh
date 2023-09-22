#!/bin/bash
# Function to display usage information
function show_usage {
    echo "Usage: $0 <input_onnx_path>"
    exit 1
}

# Check if an input ONNX file is provided as an argument
if [ $# -ne 1 ]; then
    show_usage
fi

INPUT_ONNX_PATH="$1"
OUTPUT_BASE_NAME=$(basename "$INPUT_ONNX_PATH")
OUTPUT_MODEL_PATH="${OUTPUT_BASE_NAME%.*}"

# Display input ONNX path and derived output names
echo "Input ONNX Path: $INPUT_ONNX_PATH"
echo "Output Model Path: $OUTPUT_MODEL_PATH"

# Convert ONNX to TensorFlow format and check the exit code
onnx2tf -i "$INPUT_ONNX_PATH" -o "$OUTPUT_MODEL_PATH" -osd
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Conversion completed. Model saved at: $OUTPUT_MODEL_PATH"
else
    echo "Conversion failed with exit code $EXIT_CODE. Please check the ONNX file and try again."
fi
