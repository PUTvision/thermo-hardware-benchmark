# Arduino

This document contains configuration steps, source code, and evaluation of Arduino Nano 33 BLE Sense and Arduino Portenta H7 platforms to investigate the performance of the proposed neural network model on the Thermo Presence dataset.

## Configuration

### Requirements

1. [Arduino IDE](https://www.arduino.cc/en/software)
2. [tflite-micro-arduino-examples](https://github.com/tensorflow/tflite-micro-arduino-examples)

### Setup environment

Add *tflite-micro-arduino-examples* repository as *Arduino_TensorFlowLite* library in Arduino IDE accordingly to [How to Install](https://github.com/tensorflow/tflite-micro-arduino-examples/blob/main/README.md#how-to-install) of *tflite-micro-arduino-examples* repository.

## Source Code

The source code for Arduino devices, which use [Arduino IDE](https://www.arduino.cc/en/software), is in the [thermo](./thermo) directory. Hardware implementation consists of:
- `thermo.ino` - main functions (setup and loop) definition for the evaluation process
- `unet_model.cpp` - with proposed neural network model in flatbuffer format
- `model_utils.cpp` - utilities functions used to debug or process inference of the model 
- `image_provider.cpp` - debug module to check inference with simple test image provided in this file

## Evaluation

Connect the Arduino device using a serial port to the computer. Run the `serial_evaluator.py` script with the below configuration (adjust the baud rate or serial port when required): 

```console
python serial_evaluator.py --test-input-path ./data/test_input_2347.npy --output-path ./arduino_results.npy 
```
