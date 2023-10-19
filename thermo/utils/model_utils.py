import time
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from data_generator.thermal_data_generator import ThermalDataset


def check_model_prediction(model: tf.keras.Model, config: dict) -> Tuple[float, np.ndarray]:
    raw_frame = np.array([
        26.30, 25.40, 24.62, 24.32, 23.96, 24.13, 24.02, 23.67, 23.98, 24.12, 23.89, 23.96, 24.26, 23.89, 24.10, 24.15, 24.16, 23.91, 24.30,
        24.62, 24.57, 24.73, 24.86, 24.35, 24.54, 24.76, 25.17, 24.54, 24.62, 24.49, 24.64, 24.72, 24.94, 25.28, 24.57, 25.03, 24.21, 24.33,
        23.74, 24.34, 24.12, 24.20, 24.01, 24.31, 24.15, 24.42, 24.14, 24.20, 24.32, 24.28, 24.25, 24.69, 24.66, 25.09, 25.00, 25.21, 24.49,
        24.95, 24.66, 24.98, 25.36, 25.47, 24.80, 25.06, 25.55, 25.09, 26.00, 26.18, 24.42, 23.54, 23.89, 23.94, 24.17, 23.78, 24.33, 23.90,
        24.26, 24.18, 24.32, 24.03, 24.32, 23.96, 24.38, 24.50, 26.70, 27.40, 27.41, 27.27, 25.65, 25.27, 25.30, 24.98, 25.42, 24.89, 25.21,
        25.87, 25.52, 25.90, 26.28, 26.04, 24.06, 24.37, 24.00, 24.23, 23.92, 24.04, 24.41, 24.46, 24.30, 24.51, 24.62, 24.35, 24.12, 24.29,
        24.15, 24.53, 28.01, 28.55, 28.44, 28.22, 25.69, 25.75, 25.12, 25.37, 25.12, 25.68, 25.49, 24.66, 26.08, 26.11, 27.20, 25.16, 24.86,
        24.53, 24.05, 24.05, 24.25, 24.31, 24.10, 24.37, 24.56, 24.24, 24.22, 24.13, 24.48, 24.38, 24.39, 24.77, 28.24, 28.71, 28.54, 28.01,
        26.94, 26.82, 25.83, 25.53, 25.25, 25.18, 25.11, 25.25, 25.66, 26.08, 25.53, 25.00, 24.79, 25.03, 24.30, 24.13, 24.43, 24.37, 24.14,
        24.36, 24.16, 24.19, 24.25, 24.44, 24.24, 24.25, 24.21, 24.85, 28.01, 28.34, 28.12, 27.89, 26.98, 28.14, 26.41, 25.86, 25.07, 25.78,
        24.80, 25.27, 26.23, 25.72, 24.81, 24.51, 25.44, 24.68, 24.74, 24.50, 24.68, 24.29, 24.09, 24.39, 24.23, 24.17, 24.30, 24.32, 24.29,
        24.60, 24.40, 24.49, 26.57, 27.82, 28.14, 27.34, 26.90, 27.69, 26.14, 25.33, 25.38, 25.06, 24.48, 24.77, 26.18, 26.82, 24.87, 24.96,
        25.49, 25.17, 24.40, 25.13, 24.25, 25.18, 24.17, 24.11, 24.49, 24.18, 24.35, 24.49, 24.33, 24.53, 24.51, 24.74, 26.17, 27.09, 27.33,
        26.04, 26.09, 26.34, 25.47, 25.21, 24.75, 24.56, 24.49, 25.30, 26.54, 26.63, 24.94, 25.41, 25.32, 24.74, 25.46, 25.40, 27.16, 25.78,
        26.40, 24.17, 24.26, 24.28, 24.38, 24.29, 24.43, 24.62, 24.58, 24.62, 25.09, 25.15, 25.16, 24.91, 24.54, 24.51, 24.49, 24.58, 24.63,
        24.68, 24.95, 24.84, 26.81, 27.32, 25.08, 25.07, 25.40, 25.34, 26.05, 27.11, 27.12, 28.91, 25.48, 26.05, 24.26, 24.44, 24.60, 24.49,
        24.58, 24.66, 24.62, 24.76, 24.23, 24.54, 24.13, 24.28, 24.50, 24.55, 24.67, 24.48, 24.67, 24.53, 24.29, 25.18, 26.62, 27.28, 25.30,
        25.43, 25.79, 25.55, 26.41, 27.44, 28.50, 29.07, 27.70, 25.46, 24.76, 24.37, 24.78, 24.71, 24.71, 24.68, 24.82, 24.42, 24.18, 23.88,
        24.17, 24.42, 24.78, 24.77, 24.51, 24.43, 24.75, 24.35, 25.17, 24.62, 26.31, 27.40, 25.72, 25.93, 25.43, 25.42, 27.39, 27.29, 28.52,
        28.17, 26.92, 26.39, 24.81, 24.85, 24.49, 24.57, 24.76, 24.69, 24.43, 24.32, 23.96, 24.09, 24.06, 24.61, 24.59, 24.72, 24.72, 24.28,
        24.38, 24.40, 24.65, 25.14, 25.81, 25.71, 25.59, 25.74, 25.27, 25.78, 26.65, 27.84, 27.82, 27.98, 25.83, 25.94, 24.86, 24.90, 24.45,
        24.64, 24.83, 24.55, 24.52, 24.44, 24.35, 24.32, 24.44, 24.51, 24.78, 24.32, 24.63, 24.85, 24.75, 24.55, 24.79, 24.73, 25.85, 25.57,
        25.36, 25.60, 25.39, 25.37, 26.59, 25.84, 28.31, 27.04, 25.86, 24.87, 24.95, 24.89, 24.59, 24.96, 24.70, 24.90, 24.66, 24.62, 24.29,
        24.46, 24.55, 24.51, 24.52, 24.57, 24.42, 24.81, 24.63, 24.63, 24.67, 24.87, 25.46, 25.05, 25.19, 25.09, 25.14, 25.17, 25.19, 25.73,
        26.17, 25.72, 25.22, 25.33, 25.07, 24.83, 25.12, 24.92, 25.15, 24.74, 24.93, 24.85, 25.22, 24.73, 24.94, 24.65, 24.97, 24.63, 24.85,
        24.70, 24.67, 24.77, 24.69, 25.17, 24.88, 25.76, 25.40, 25.35, 25.39, 25.33, 25.68, 25.47, 25.82, 25.54, 25.23, 25.40, 24.99, 25.17,
        25.04, 25.26, 25.05, 25.13, 25.17, 24.91, 24.99, 25.36, 24.96, 25.15, 24.90, 24.81, 24.62, 24.97, 24.47, 25.01, 24.54, 25.29, 26.57,
        25.80, 26.06, 25.54, 27.46, 26.74, 25.75, 25.52, 25.11, 25.13, 25.28, 25.50, 25.31, 25.43, 25.15, 25.49, 25.30, 25.14, 24.98, 25.01,
        25.28, 25.03, 25.04, 24.97, 25.05, 24.93, 25.05, 24.65, 24.69, 24.56, 25.15, 25.40, 26.88, 26.72, 25.88, 26.11, 27.62, 27.41, 25.54,
        25.57, 25.48, 25.37, 25.45, 25.44, 25.64, 25.62, 25.62, 25.55, 25.12, 25.32, 24.53, 24.95, 25.11, 25.23, 25.20, 25.19, 24.97, 24.99,
        24.98, 24.88, 24.79, 25.04, 24.98, 25.18, 27.10, 26.77, 26.94, 26.84, 29.03, 28.50, 26.17, 25.22, 25.72, 25.36, 25.69, 25.70, 25.62,
        25.62, 26.09, 25.74, 25.48, 25.22, 24.93, 24.73, 24.96, 25.17, 25.48, 25.34, 25.40, 25.25, 24.78, 24.93, 25.20, 24.62, 25.33, 25.04,
        26.88, 27.34, 27.59, 28.39, 29.27, 28.78, 25.72, 25.85, 25.38, 25.67, 25.49, 25.41, 25.82, 25.75, 26.16, 25.85, 25.45, 25.30, 24.62,
        24.80, 25.18, 25.45, 25.04, 25.55, 24.91, 25.39, 25.01, 24.99, 25.17, 25.25, 24.58, 24.80, 27.64, 27.09, 27.18, 27.96, 27.94, 27.12,
        25.31, 25.52, 26.18, 25.46, 25.52, 25.65, 25.70, 25.92, 25.98, 25.79, 25.72, 25.17, 24.68, 25.21, 25.81, 25.36, 25.26, 25.17, 25.26,
        24.71, 24.90, 24.65, 25.05, 24.69, 24.88, 24.96, 26.91, 27.62, 27.37, 27.54, 26.78, 26.63, 25.92, 26.13, 25.69, 26.03, 25.89, 25.60,
        25.45, 25.69, 25.87, 25.65, 25.48, 25.41, 24.58, 25.04, 25.57, 25.62, 25.32, 24.97, 24.99, 25.05, 24.94, 25.18, 24.94, 25.19, 25.00,
        25.65, 27.27, 25.84, 26.61, 26.67, 27.08, 25.93, 25.68, 25.48, 25.85, 25.53, 26.04, 25.74, 26.05, 25.46, 25.65, 25.39, 25.72, 25.60,
        25.83, 25.77, 26.02, 26.23, 25.12, 25.31, 25.49, 25.31, 25.32, 25.28, 25.29, 25.52, 25.61, 25.92, 27.54, 27.43, 26.99, 27.30, 26.67,
        26.24, 26.08, 26.16, 25.60, 25.32, 25.40, 25.58, 25.31, 25.52, 25.60, 25.67, 25.09, 25.38, 25.73, 26.24, 25.78, 25.92, 25.74, 25.94,
        25.08, 25.43, 24.95, 25.63, 25.23, 25.43, 25.33, 25.69
    ])

    frame_2d = np.reshape(raw_frame, config['dataset']["IR_camera_resolution"])
    frame_normalized = (frame_2d - config['dataset']["temperature_normalization_min"]) * \
        (1 / (config['dataset']["temperature_normalization_max"] -
         config['dataset']["temperature_normalization_min"]))
    input_frame = np.expand_dims(frame_normalized, axis=(0, 3))

    output_frame = model.predict(input_frame, verbose=0)

    count = np.sum(output_frame) / \
        config['dataset']["sum_of_values_for_one_person"]

    return count, output_frame


def tf_keras_inference(model: tf.keras.Model, inputs: np.ndarray) -> np.ndarray:
    inference_start = time.time()
    outputs = model.predict(inputs, verbose=0)
    inference_time = time.time() - inference_start

    return outputs, inference_time


def tflite_inference(interpreter: Union[tf.lite.Interpreter, str], inputs: np.ndarray) -> np.ndarray:
    if type(interpreter) == str:
        interpreter = tf.lite.Interpreter(interpreter)
    # TFLite allocate tensors.
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale = input_details[0]['quantization'][0]
    in_zero_point = input_details[0]['quantization'][1]
    in_dtype = input_details[0]['dtype']

    out_scale = output_details[0]['quantization'][0]
    out_zero_point = output_details[0]['quantization'][1]

    outputs = []
    inference_time = 0

    for input_data in inputs:
        if (in_scale, in_zero_point) != (0.0, 0):
            input_data = input_data / in_scale + in_zero_point

        interpreter.set_tensor(
            input_details[0]['index'], [input_data.astype(in_dtype)])

        inference_start = time.time()
        interpreter.invoke()
        inference_time += time.time() - inference_start

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])

        if (out_scale, out_zero_point) != (0.0, 0):
            output_data = (output_data - out_zero_point) * out_scale

        outputs.append(output_data)

    return np.vstack(outputs), inference_time


def evaluate(
    model_path: str,
    model_type: str,
    loader: ThermalDataset,
    config: dict,
    skip_confusion_matrix: bool = False,
) -> Tuple[float, float, np.ndarray]:
    """ Validate the model on data from the loader, calculate and print the results and metrics """
    correct_count = 0
    tested_frames = 0
    number_of_frames_with_n_persons = {}
    number_of_frames_with_n_persons_predicted_correctly = {}

    confusion_matrix = np.zeros(shape=(
        config['dataset']["max_people_count"]+1, config['dataset']["max_people_count"]+1), dtype=int)

    mae_sum = 0
    mse_sum = 0
    mae_rounded_sum = 0
    mse_rounded_sum = 0
    mrae_sum = 0

    vec_real_number_of_persons = []
    vec_predicted_number_of_persons = []

    if model_type == 'keras':
        inference_func = tf_keras_inference
        model = tf.keras.models.load_model(model_path, compile=False)
    elif model_type == 'tflite':
        inference_func = tflite_inference
        model = tf.lite.Interpreter(model_path)

    inference_time = 0
    for frames, labels in loader:
        outputs, batch_inference_time = inference_func(model, frames)
        inference_time += batch_inference_time

        for i in range(len(labels)):
            predicted_img = np.array(outputs[i])
            pred_people = np.sum(predicted_img) / \
                config['dataset']["sum_of_values_for_one_person"]
            pred_label = round(pred_people)

            true_label = round(
                np.sum(labels[i]) / config['dataset']["sum_of_values_for_one_person"])

            if not skip_confusion_matrix:
                confusion_matrix[true_label][pred_label] += 1

            error = abs(pred_people - true_label)
            mae_sum += error
            mse_sum += error*error

            rounded_error = abs(pred_label - true_label)
            mae_rounded_sum += rounded_error
            mse_rounded_sum += rounded_error*rounded_error
            if rounded_error != 0:
                mrae_sum += (rounded_error + 1) / (true_label + 1) if true_label == 0 else rounded_error / true_label

            number_of_frames_with_n_persons[pred_label] = \
                number_of_frames_with_n_persons.get(pred_label, 0) + 1

            if true_label == pred_label:
                correct_count += 1
                number_of_frames_with_n_persons_predicted_correctly[pred_label] = \
                    number_of_frames_with_n_persons_predicted_correctly.get(
                        pred_label, 0) + 1

            vec_real_number_of_persons.append(true_label)
            vec_predicted_number_of_persons.append(pred_people)
            tested_frames += 1

    mae = mae_sum / tested_frames
    mse = mse_sum / tested_frames
    mae_rounded = mae_rounded_sum / tested_frames
    mse_rounded = mse_rounded_sum / tested_frames
    mrae = mrae_sum / tested_frames

    model_accuracy = correct_count / tested_frames
    model_f1_score = f1_score(vec_real_number_of_persons, np.round(
        vec_predicted_number_of_persons).astype(int), average='weighted')

    average_inference_time = inference_time / tested_frames
    average_fps = 1 / average_inference_time

    print(f"Number of tested frames: {tested_frames}")
    print(f"Average inference time: {round(average_inference_time, 6)}")
    print(f"Average FPS: {round(average_fps,4)}")
    print(f"Model Accuracy = {model_accuracy}")
    print(f"Model F1 score = {model_f1_score}")
    print('Predicted:\n' + '\n'.join([f'   {count} frames with {no} persons' for no,
          count in sorted(number_of_frames_with_n_persons.items())]))
    print('Predicted correctly:\n' + '\n'.join([f'   {count} frames with {no} persons' for no, count in sorted(
        number_of_frames_with_n_persons_predicted_correctly.items())]))
    print(f'mae: {mae}')
    print(f'mse: {mse}')
    print(f'mae_rounded: {mae_rounded}')
    print(f'mse_rounded: {mse_rounded}')
    print(f'mrae: {mrae}')

    return model_accuracy, model_f1_score, confusion_matrix
