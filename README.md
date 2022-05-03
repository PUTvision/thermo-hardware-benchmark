# Hardware Benchmark of Presence Monitoring in Thermal Images

This is an official repository for *Efficient People Counting in Thermal Images: The Benchmark of Resource-Constrained Hardware* paper.

The repository includes code for reproducing obtained results in the aforementioned publication and is organized as follows:
- [thermo](./thermo/) directory contains code and scripts related to model training, optimization, and quantization phases,
- [evaluation](./evaluation/) folder consists of hardware deployment, inference and evaluation tools. 


## Table of contents
* [Citation](#citation)
* [Dataset](#dataset)
* [Benchmark Hardware](#benchmark-hardware)
* [Neural Network Architecture](#neural-network-architecture)
* [Results](#results)


## Citation


## Dataset

The benchmark utilizes [Thermo Presence](https://github.com/PUTvision/thermo-presence) dataset that consists of 13 644 low-resolution thermal images recorded in office spaces. The dataset distribution is shown in the table below. The training, validation, and test split was chosen according to the authors' choice which is described in their [project repository](https://github.com/PUTvision/thermo-presence/blob/master/dataset/README.md).

<div align="center">

|            |  0  |  1  |   2  |   3  |   4  |  5  | Total |
|:----------:|:---:|:---:|:----:|:----:|:----:|:---:|:-----:|
|  Training  |  99 | 105 | 2984 | 3217 | 1953 | 114 |  8472 |
| Validation |  0  | 139 |  631 | 1691 |  225 | 139 |  2825 |
|    Test    | 162 |  83 |  211 |  341 | 1235 | 315 |  2347 |

</div>


## Benchmark Hardware

<div align="center">

| Device                       | Target Hardware      | Max Clock Frequency     | FLASH | SRAM   | Evaluation Framework |
|------------------------------|:--------------------:|:-----------------------:|:-----:|:------:|:--------------------:|
| [Arduino Nano 33 BLE Sense](https://docs.arduino.cc/hardware/nano-33-ble-sense) | nRF52840 | 64 MHz | 1 MB | 256 KB | TFLite Micro |
| [Arduino Portenta H7](http://store.arduino.cc/products/portenta-h7) | STM32H747 | 480 MHz | 2 MB | 1 MB | TFLite Micro |
| [STM32 F429ZI Nucleo-144](https://www.st.com/en/evaluation-tools/nucleo-f429zi.html) | STM32F429 | 180 MHz | 2 MB | 256 KB | STM32 Cube AI |
| [STM32 H745ZI Nucleo-144](https://www.st.com/en/evaluation-tools/nucleo-h745zi-q.html) | STM32H745 | 480 MHz | 2 MB | 1 MB | STM32 Cube AI |
| [SiPEED MAiXDUINO](https://www.seeedstudio.com/Sipeed-Maixduino-Kit-for-RISC-V-AI-IoT-p-4047.html) | ESP32-WROOM-32 | 240 MHz | 4 MB | 520 KB | TFLite Micro |
| [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) | Quad Core Cortex-A72 | 1.5 GHz | * | 2 GB | TensorFlow Lite |
| [Coral USB Accelerator](https://coral.ai/products/accelerator/) | Google Edge TPU | 500 MHz | * | 8 MB | TensorFlow Lite |
| [Intel Neural Compute Stick 2](https://ark.intel.com/content/www/us/en/ark/products/140109/intel-neural-compute-stick-2.html) | Intel Movidius Myriad X Vision Processing Unit | 700 MHz | * | 4 GB | OpenVINO |

</div>


## Neural Network Architecture

The neural network is based on U-Net architecture with shallow structure, single input, and output. The proposed model has only 46 577 parameters.

<p align="center">
  <img width='800px' src="./README/nn_architecture.png" />
</p>


## Results

<div align="center">

| Metric Name | Results |
|-------------|:-------:|
| MAE | 0.1057 |
| MSE | 0.0332 |
| Counting MAE | 0.0226 |
| Counting MSE | 0.0234 |
| Counting MRAE | 0.0081 |
| Accuracy | 0.9778 |
| F1 Score | 0.9782 |
| No. of parameters | 46 577 |

</div>

## Hardware benchmark

<div align="center">

| Device | Data Type | Avg. Inference Time [ms] | MAE | MSE |
|--------|:---------:|:------------------------:|:---:|:---:|
| Arduino Nano 33 BLE Sense | INT8 | 1430.1253 ±1.1427 | 0.1070 | 0.0335 |
| Arduino Portenta H7 | INT8 | 137.494 ±0.5 | 0.1070 | 0.0335 |
| STM32 F429ZI Nucleo-144 | INT8 | 230.939 ±0.1 | 0.1195 | 0.0428 |
| STM32 H745ZI Nucleo-144 | FP32 | 165.983 ±0.1 | 0.1057 | 0.0332 |
| | INT8 | 53.2600 ±0.1 | 0.1195 | 0.0428 |
| SiPEED MAiXDUINO | INT8 |  |  |  |
| Raspberry Pi 4B | FP32 | 7.7065 ±0.591 | 0.1057 | 0.0332 |
| Raspberry Pi 4B | FP16 | 7.6900 ±0.5584 | 0.1058 | 0.0332 |
| Raspberry Pi 4B | INT8 | 4.1939 ±0.052 | 0.1201 | 0.0429 |
| Raspberry Pi 4B + Coral USB Accelerator (std) | INT8 | 0.6051 ±0.0443 | 0.1201 | 0.0431 |
| Raspberry Pi 4B + Coral USB Accelerator (max) | INT8 | 0.5699 ±0.0601 | 0.1201 | 0.0431 |
| Raspberry Pi 4B + Intel Neural Compute Stick 2  | FP32 | 2.6297 ±0.159 | 0.1338 | 0.0430 |
| | FP16 | 2.3001 ±0.1 | 0.1338 | 0.0430 |

</div>

The figure below consists of current consumption characteristics measured on benchmarked hardware. The graphs are limited to 180 seconds in order to highlight peaks related to the prediction stage. 

<p align="center">
  <img width='800px' src="./README/device_inference_current_consumption.png" />
</p>
