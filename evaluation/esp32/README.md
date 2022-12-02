# ESP32

This document contains configuration steps, source code, and evaluation of LOLIN32 with ESP32-WROOM-32 module platform to investigate the performance of the proposed neural network model on the Thermo Presence dataset.

## Configuration

### Requirements

1. [esp-idf](https://github.com/espressif/esp-idf) in version 4.4 (branch `release/v4.4`)
2. [tflite-micro](https://github.com/tensorflow/tflite-micro)
3. [tflite-micro-esp-examples ](https://github.com/espressif/tflite-micro-esp-examples)
4. [esp-nn](https://github.com/espressif/esp-nn)
5. [esp-dl](https://www.espressif.com/en/news/ESP-DL) **NOTE:** to check!

### Setup environment

1. Set target device
```console
idf.py set-target esp32
```

2. Build project
```console
idf.py build
```

3. Flash build into device and monitor serial
```console
idf.py --port /dev/ttyUSB0 -b 115200 flash monitor
```

**NOTE:** Use `Ctrl+]` to exit.

## Source Code

The ESP32-related files are inside [thermo](./thermo) directory. Whereas the source code is in [thermo/main](./thermo/main) directory. Hardware implementation consists of:
- `main.cc` - project main application definition, program setup and loop call
- `unet_model.cc` - with proposed neural network model in flatbuffer format
- `model_utils.cc` - utilities functions used to debug or process inference of the model 
- `main_functions.cc` - main functions (setup and loop) definition for evaluation process

## Evaluation

Connect the UART pins accordingly: `RX -> 17`, `TX -> 16`. Run `serial_evaluator.py` script with below configuration:

```console
python serial_evaluator.py --test-input-path ./data/test_input_2347.npy --test-output-path ./data/test_output_2347.npy --output-path ./esp32_results.npy
```
