import os
import time
from functools import partial
from pathlib import Path
from typing import Tuple

import click
import numpy as np
from tqdm import tqdm


def tflite_inference(input_data: np.ndarray, interpreter) -> Tuple[np.ndarray, float]:
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale = input_details[0]['quantization'][0]
    in_zero_point = input_details[0]['quantization'][1]
    in_dtype = input_details[0]['dtype']

    out_scale = output_details[0]['quantization'][0]
    out_zero_point = output_details[0]['quantization'][1]

    if (in_scale, in_zero_point) != (0.0, 0):
        input_data = input_data / in_scale + in_zero_point

    interpreter.set_tensor(input_details[0]['index'], [input_data.astype(in_dtype)])

    inference_start = time.time()
    interpreter.invoke()
    inference_time = time.time() - inference_start

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if (out_scale, out_zero_point) != (0.0, 0):
        output_data = (output_data - out_zero_point) * out_scale

    return output_data, inference_time


def myriad_inference(input_data: np.ndarray, interpreter, input_blob) -> Tuple[np.ndarray, float]:
    inference_start = time.time()
    output = interpreter.infer(inputs={input_blob: input_data})
    inference_time = time.time() - inference_start

    output_data = np.array(list(output.values())[0])

    return output_data, inference_time


@click.command()
@click.option('--inference-type', help='Inference framework (device)', type=click.Choice(['tflite', 'edgetpu', 'myriad'], case_sensitive=True))
@click.option('--model-path', help='Path to model', type=click.Path(exists=True, file_okay=True))
@click.option('--test-input-path', help='Path to npy file with test input', type=click.Path(exists=True, file_okay=True))
@click.option('--output-path', help='Path for output file with results', type=click.Path())
def main(inference_type, model_path, test_input_path, output_path):
    test_input_arr = np.load(Path(test_input_path))
    print(f'Test array shape: {test_input_arr.shape}')

    time_outputs = []
    result_outputs = []

    if inference_type == 'tflite':
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path)
        interpreter.allocate_tensors()

        inference_func = partial(tflite_inference, interpreter=interpreter)
    elif inference_type == 'edgetpu':
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path)
        interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        interpreter.allocate_tensors()

        inference_func = partial(tflite_inference, interpreter=interpreter)
    elif inference_type == 'myriad':
        import ngraph as ng
        from openvino.inference_engine import IECore

        ie = IECore()

        net = ie.read_network(model_path, os.path.splitext(model_path)[0] + ".bin")
        input_blob = next(iter(net.input_info))
        interpreter = ie.load_network(network=net, device_name='MYRIAD', num_requests=1)

        inference_func = partial(myriad_inference, interpreter=interpreter, input_blob=input_blob)
    
    for arr_num in tqdm(range(test_input_arr.shape[0])):
        arr = test_input_arr[arr_num]

        test_output_mask, test_infer_time = inference_func(arr)

        time_outputs.append(test_infer_time)
        result_outputs.append(test_output_mask)

    result_outputs = np.vstack(result_outputs)
    time_outputs = np.vstack(time_outputs)
    np.save(Path(output_path), result_outputs)
    np.save(Path(output_path.replace('.npy', '_inference_time.npy')), time_outputs)
    print(f'Results saved in {output_path}\nEvaluation process finished successfully!')


if __name__ == '__main__':
    main()
