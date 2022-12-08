# STM32

This document contains configuration steps, source code, and evaluation of STM32 F429ZI Nucleo-144 and STM32 H745ZI Nucleo-144 platforms to investigate the performance of the proposed neural network model on the Thermo Presence dataset.

## Configuration

1. [STM32CubeMX](https://www.st.com/en/development-tools/stm32cubemx.html) - STM32Cube initialization code generator
2. [X-CUBE-AI](https://www.st.com/en/embedded-software/x-cube-ai.html) - AI expansion pack for STM32CubeMX

<p align="center">
  <img width='800px' src="../../README/x_cube_ai_package_configuration.png" />
  </br>
  <b>Fig. 1.</b> X-CUBE-AI package configuration with project
</p>

## Configured CubeMX Project Files

- `thermo_f429zi.ioc` - project scheme for STM32 F429ZI Nucleo-144
- `thermo_h745zi.ioc` - project scheme for STM32 H745ZI Nucleo-144

## Evaluation

The neural network model evaluation on STM32 boards was performed using the X-CUBE-AI package and its inference module.

<p align="center">
  <img width='800px' src="../../README/x_cube_ai_model_configuration.png" />
  </br>
  <b>Fig. 2.</b> Model adding in X-CUBE-AI package
</p>

<p align="center">
  <img width='800px' src="../../README/x_cube_ai_model_evaluation.png" />
  </br>
  <b>Fig. 3.</b> Configuration of model evaluation in X-CUBE-AI package:</br>1. Set TFLite framework with STM32Cube AI runtime</br>2. Select TFLite model</br>3. Select validation data (from [data](../data/) folder)</br>4. Select evaluation module, available: analyze, validation on desktop, validation on device
</p>

<p align="center">
  <img width='800px' src="../../README/x_cube_ai_model_evaluation_report.png" />
  </br>
  <b>Fig. 4.</b> Model evaluation report
</p>
