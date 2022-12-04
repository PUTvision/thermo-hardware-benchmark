# Hardware Evaluation

The evaluation directory is split into several folders that describes platforms setup, performance evaluation and contains the source code utilized during benchmark.

- [Arduino Nano 33 BLE Sense and Arduino Portenta H7](./arduino)
- [LOLIN32](./esp32)
- [STM32 F429ZI Nucleo-144 and STM32 H745ZI Nucleo-144](./stm32)
- [Raspberry Pi 4B](./raspberry_pi)

## Serial Communication

Utilized microcontrollers were configured to communicate using UART (serial) - receive input data array and send output estimation, as well as, inference time. For this task was implemented [`serial_evaluator.py`](./serial_evaluator.py) script that reads test set data (stored in [data](./data) directory as `.npy` files), transmit them to the platform and store outputs in user-defined `.npy` file.
