import argparse
import os
import pathlib

import tensorflow as tf


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    default_signature = model.signatures["serving_default"]
    return model, default_signature


def main():
    parser = argparse.ArgumentParser(description="Load and modify a TensorFlow model.")
    parser.add_argument("-i", "--input-directory", type=str, required=True, help="Path to the model directory")
    parser.add_argument("-o", "--output-directory", type=str, required=True, help="Path to save the modified model")
    parser.add_argument("-v", "--model-version", type=str, default="1", help="Default model version")

    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = os.path.join(args.output_directory, args.model_version)

    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")

    model, default_signature = load_model(input_directory)

    @tf.function
    def bytes_to_prediction(
        image_bytes,
        model_shape=(640, 640),
        score_threshold=0.35,
        iou_threshold=0.65,
        max_boxes=300,
    ):
        model_height, model_width = model_shape

        # Preprocess
        bytes_image_scalar = tf.reshape(image_bytes, [])
        decoded_image = tf.image.decode_image(bytes_image_scalar, channels=3)
        decoded_image_shape = tf.shape(decoded_image)
        image_height = decoded_image_shape[0]
        image_width = decoded_image_shape[1]
        channels = decoded_image_shape[2]
        resized_pad_image = tf.image.resize_with_pad(decoded_image, model_height, model_width)
        float32_image = tf.convert_to_tensor(resized_pad_image, dtype=tf.float32)
        normalized_image = float32_image / 255.0

        tf_img = tf.reshape(normalized_image, [1, model_height, model_height, 3])

        # Prediction
        y = default_signature(tf_img)
        raw_predictions = y["output0"][0]
        raw_predictions = tf.transpose(raw_predictions)
        raw_predictions = tf.cast(raw_predictions, dtype=tf.float32)

        # Post processing
        model_width = tf.cast(model_width, dtype=tf.float32)
        model_height = tf.cast(model_height, dtype=tf.float32)
        image_width = tf.cast(image_width, dtype=tf.float32)
        image_height = tf.cast(image_height, dtype=tf.float32)

        boxes = tf.zeros([0, 6])

        for row in raw_predictions:
            tf.autograph.experimental.set_loop_options(shape_invariants=[(boxes, tf.TensorShape([None, 6]))])
            xywh = row[0:4]
            class_probs = row[4:]

            max_class_idx = tf.argmax(class_probs)
            max_class_prob = class_probs[max_class_idx]

            max_class_idx = tf.cast(max_class_idx, dtype=tf.float32)

            xc, yc, w, h = tf.unstack(xywh, axis=-1)

            if max_class_prob < score_threshold:
                continue

            x1 = xc - (w / 2.0)
            y1 = yc - (h / 2.0)
            x2 = xc + (w / 2.0)
            y2 = yc + (h / 2.0)

            gain = tf.minimum(model_height / image_height, model_width / image_width)
            x_pad = tf.round((model_width - image_width * gain) / 2 - 0.1)
            y_pad = tf.round((model_height - image_height * gain) / 2 - 0.1)

            x1 = (x1 - x_pad) / gain
            x2 = (x2 - x_pad) / gain
            y1 = (y1 - y_pad) / gain
            y2 = (y2 - y_pad) / gain

            result = tf.convert_to_tensor([x1, x2, y1, y2, max_class_idx, max_class_prob])
            result = tf.reshape(result, [1, 6])
            boxes = tf.concat([boxes, result], 0)

        if tf.shape(boxes)[0] > 1:
            # Apply NMS
            applied_indices = tf.image.non_max_suppression(
                boxes[:, :4],
                boxes[:, 5] * boxes[:, 4],
                max_boxes,
                iou_threshold=iou_threshold,
            )

            boxes = tf.gather(boxes, applied_indices)
        return {"predictions": boxes}

    # Create new signature, to read bytes images
    custom_signature = bytes_to_prediction.get_concrete_function(
        image_bytes=tf.TensorSpec([None], dtype=tf.string, name="image_bytes")
    )

    # Adjust serving default to be better usage
    tf.saved_model.save(model, output_directory, signatures=custom_signature)
    print(f"Finished, extended model path: {output_directory}")


if __name__ == "__main__":
    main()
