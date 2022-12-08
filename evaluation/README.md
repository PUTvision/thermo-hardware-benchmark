# Hardware Evaluation

The evaluation directory is split into several folders that describe platform setup and performance evaluation and contains the source code utilized during the benchmark.

- [Arduino Nano 33 BLE Sense and Arduino Portenta H7](./arduino)
- [LOLIN32](./esp32)
- [STM32 F429ZI Nucleo-144 and STM32 H745ZI Nucleo-144](./stm32)
- [Raspberry Pi 4B](./raspberry_pi)

## Serial Communication

Utilized microcontrollers were configured to communicate using UART (serial) - receive input data array and send output estimation, as well as, inference time. For this task was implemented [`serial_evaluator.py`](./serial_evaluator.py) script that reads test set data (stored in [data](./data) directory as `.npy` files), transmit them to the platform and store outputs in user-defined `.npy` file.

## Results

<div align="center">

| Device                                          | Data Type | Avg. Inference Time [ms]  | Counting MAE | Counting MSE | Counting MRAPE [%]  |
|-------------------------------------------------|:---------:|:-------------------------:|:------------:|:------------:|:-------------------:|
| Arduino Nano 33 BLE Sense                       | INT8      | 1430.125 ±1.143           | 0.023        | 0.024        | 0.82                |
| Arduino Portenta H7                             | INT8      |  137.494 ±0.500           | 0.023        | 0.024        | 0.82                |
| LOLIN32                                         | INT8      |  840.442 ±0.021           | 0.023        | 0.023        | 0.81                |
| STM32 F429ZI Nucleo-144                         | INT8      |  230.939 ±0.100           | 0.036        | 0.038        | 1.14                |
| STM32 H745ZI Nucleo-144                         | FP32      |  165.983 ±0.100           | 0.023        | 0.023        | 0.81                |
|                                                 | INT8      |   53.260 ±0.100           | 0.036        | 0.038        | 1.14                |
| Raspberry Pi 4B                                 | FP32      |    7.707 ±0.591           | 0.023        | 0.023        | 0.81                |
|                                                 | FP16      |    7.690 ±0.558           | 0.023        | 0.023        | 0.81                |
|                                                 | INT8      |    4.194 ±0.052           | 0.038        | 0.039        | 1.18                |
| Raspberry Pi 4B + Coral USB Accelerator (std)   | INT8      |    0.605 ±0.044           | 0.037        | 0.038        | 1.16                |
| Raspberry Pi 4B + Coral USB Accelerator (max)   | INT8      |    0.570 ±0.060           | 0.037        | 0.038        | 1.16                |
| Raspberry Pi 4B + Intel Neural Compute Stick 2  | FP32      |    2.630 ±0.159           | 0.028        | 0.029        | 0.92                |
|                                                 | FP16      |    2.300 ±0.100           | 0.028        | 0.029        | 0.91                |

</div>
