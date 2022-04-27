# Hardware Benchmark of Presence Monitoring in Thermal Images


## Table of contents
* [Citation](#citation)
* [Dataset](#dataset)
* [Benchmark Hardware](#benchmark-hardware)
* [Neural Network Architecture](#neural-network-architecture)
* [Results](#results)


## Citation


## Dataset

The benchmark utitilzes Thermo Presence dataset that consists of 13 644 low resolution thermal images recorded in office spaces. The dataset distribution is shown in table below. The training, validation and test split was chosen according to authors choice which is described in their [project repository](https://github.com/PUTvision/thermo-presence/blob/master/dataset/README.md).

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
| [Arduino Nano 33 BLE Sense](https://docs.arduino.cc/hardware/nano-33-ble-sense)    | nRF52840             | 64 MHz                  | 1 MB  | 256 KB | TFLite Micro         |
| [Arduino Portenta H7](http://store.arduino.cc/products/portenta-h7)          | STM32H747            | 480 MHz                 | 2 MB  | 1 MB   | TFLite Micro         |
| [STM32 F429ZI Nucleo-144](https://www.st.com/en/evaluation-tools/nucleo-f429zi.html)      | STM32F429            | 180 MHz                 | 2 MB  | 256 KB | STM32 Cube AI        |
| [STM32 H745ZI Nucleo-144](https://www.st.com/en/evaluation-tools/nucleo-h745zi-q.html)      | STM32H745            | 480 MHz                 | 2 MB  | 1 MB   | STM32 Cube AI        |
| [SiPEED MAiXDUINO](https://www.seeedstudio.com/Sipeed-Maixduino-Kit-for-RISC-V-AI-IoT-p-4047.html)             | ESP32-WROOM-32       | 240 MHz                 | 4 MB  | 520 KB | TFLite Micro         |
| [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)              | Quad Core Cortex-A72 | 1.5 GHz                 | *     | 2GB    | TensorFlow Lite      |
| [Coral USB Accelerator](https://coral.ai/products/accelerator/)        | Google Edge TPU      | 500 MHz                 | *     | 8 MB   | TensorFlow Lite      |
| [Intel Neural Compute Stick 2](https://ark.intel.com/content/www/us/en/ark/products/140109/intel-neural-compute-stick-2.html) | Intel Movidius Myriad X Vision Processing Unit | 700 MHz | * | 4GB | OpenVINO          |

</div>


## Neural Network Architecture

<p align="center">
  <img width='800px' src="./README/nn_architecture.png" />
</p>


## Results

<p align="center">
  <img width='800px' src="./README/device_inference_current_consumption.png" />
</p>
