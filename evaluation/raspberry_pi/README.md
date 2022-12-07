# Raspberry Pi 4B

This document contains configuration steps, source code, and evaluation of the Raspberry Pi 4B platform, with and without co-processors in the form of Coral USB Accelerator and Intel Neural Compute Stick 2, to investigate the performance of the proposed neural network model on the Thermo Presence dataset.

## Configuration

### Raspberry Pi 4B

1. [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) - hardware specification
2. [Raspberry Pi OS](https://www.raspberrypi.com/software/operating-systems/)
3. [tflite_runtime](https://www.tensorflow.org/lite/guide/python) package

### Coral USB Accelerator

1. [Coral USB Accelerator](https://coral.ai/products/accelerator/) - hardware overview
2. [USB Accelerator datasheet](https://coral.ai/docs/accelerator/datasheet/) - hardware specification
3. [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler) - NN model compilation to platform-supported format
4. [Run inference on the Edge TPU with Python](https://coral.ai/docs/edgetpu/tflite-python/#overview) - Coral TPU usage with tflite_runtime package

### Intel Neural Compute Stick 2

1. [Intel Neural Compute Stick 2](https://www.intel.com/content/www/us/en/products/sku/140109/intel-neural-compute-stick-2/specifications.html) - hardware specification
2. [Intel Distribution of OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
3. [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) - module for NN model optimization for Intel hardware

## Source Code

The source code for Raspberry Pi 4B and accelerators evaluation consists of two elements:
- `evaluate_rpi4b.py` - evaluation script that should be called directly on the Raspberry Pi device
- `models` directory that includes neural networks models optimized and quantized to use with aforementioned hardware:
  - `model_int8.tflite` - TF Lite model quantized to INT8 data format, inference with *tflite* option
  - `model_int8_edgetpu.tflite` - INT8 data format model, compiled using Edge TPU Compiler to format supported by Coral devices
  - `model_FP*.{bin, xml, mapping}` - FP32 or FP16 data type models, optimized to achieve high performance on Intel hardware

## Evaluation

To evaluate the algorithm on Raspberry Pi 4B, with or without accelerators, call directly on the device below the script and select the proper inference framework (type).

```shell
python evaluate_rpi4b.py --inference_type tflite --model-path ./models/model_int8.tflite --test-input-path ../data/test_input_2347.npy --output-path ./raspberry_pi_4b_tflite_results.npy
```

```console
$ python evaluate_rpi4b.py --help

Usage: evaluate_rpi4b.py [OPTIONS]

Options:
  --inference-type [tflite|edgetpu|myriad]
                                  Inference framework (device)
  --model-path PATH               Path to model
  --test-input-path PATH          Path to npy file with test input
  --output-path PATH              Path for output file with results
  --help                          Show this message and exit.
```
