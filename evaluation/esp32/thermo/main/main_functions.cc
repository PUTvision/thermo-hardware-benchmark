/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <unistd.h>
#include <stdio.h>

#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_log.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "unet_model.h"
#include "model_utils.h"
#include "main_functions.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;

  const unsigned char *g_model = g_model_unet;

  constexpr int kTensorArenaSize = 24 * 32 * 32 * 4;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// The name of this function is important for Arduino compatibility.
void setup()
{
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  // static tflite::AllOpsResolver resolver;
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddRelu();
  resolver.AddMaxPool2D();
  resolver.AddConcatenation();
  resolver.AddResizeNearestNeighbor();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

// The name of this function is important for Arduino compatibility.
void loop()
{
  const int buff_size = 768;
  uint8_t data[buff_size];
  uint8_t output_data[buff_size];
  int length;

  struct timespec tstart = {0, 0}, tend = {0, 0};
  float dt;

  const uart_port_t uart_num = UART_NUM_2;

  uart_config_t uart_config = {
      .baud_rate = 115200,
      .data_bits = UART_DATA_8_BITS,
      .parity = UART_PARITY_DISABLE,
      .stop_bits = UART_STOP_BITS_1,
      .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
      .rx_flow_ctrl_thresh = 122,
  };
  // Configure UART parameters
  ESP_ERROR_CHECK(uart_param_config(uart_num, &uart_config));

  // Set UART pins(TX, RX, RTS, CTS) for LOLIN32
  ESP_ERROR_CHECK(uart_set_pin(uart_num, 17, 16, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

  QueueHandle_t uart_queue;
  // Install UART driver using an event queue here
  ESP_ERROR_CHECK(uart_driver_install(uart_num, buff_size, 0, 10, &uart_queue, 0));

  while (true)
  {
    length = 0;
    uart_get_buffered_data_len(uart_num, (size_t *)&length);
    length = uart_read_bytes(uart_num, data, buff_size, 5000);

    if (length > 0)
    {
      for (int i = 0; i < buff_size; i++)
        input->data.int8[i] = (int)data[i] - 128;

      clock_gettime(CLOCK_MONOTONIC, &tstart);
      // Run the model on this input and make sure it succeeds.
      if (kTfLiteOk != interpreter->Invoke())
      {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
      }
      clock_gettime(CLOCK_MONOTONIC, &tend);
      dt = ((double)tend.tv_sec + 1.0e-9 * tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9 * tstart.tv_nsec);
      dt *= 1000;

      for (int i = 0; i < buff_size; i++)
        output_data[i] = output->data.int8[i] + 127;

      // send inference time
      uart_write_bytes(uart_num, &dt, sizeof(float));
      // wait 1000 ms
      vTaskDelay(1000 / portTICK_RATE_MS);
      // send inference results
      uart_write_bytes(uart_num, output_data, buff_size);
    }
  }
}
