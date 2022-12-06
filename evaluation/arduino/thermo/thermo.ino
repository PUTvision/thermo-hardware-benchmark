/*
  People counting in thermal images
  --------------------------------

  Uses image from thermal camera as an input for Neural Network to count people in image.

  Hardware: Arduino Nano 33 BLE Sense board.

  Created by Mateusz Piechocki
*/

#include <TensorFlowLite.h>

#include "image_provider.h"
#include "model_utils.h"
#include "unet_model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;

  const unsigned char *g_model = g_model_unet;

  // In order to use optimized tensorflow lite kernels, a signed int8_t quantized
  // model is preferred over the legacy unsigned model format. This means that
  // throughout this project, input images must be converted from unisgned to
  // signed format. The easiest and quickest way to convert from unsigned to
  // signed 8-bit integers is to subtract 128 from the unsigned value to get a
  // signed value.

  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = kNumRows * kNumCols * kNumChannels * 32 * 5;
  static uint8_t tensor_arena[kTensorArenaSize];

  float count = 0;
} // namespace

void GetSerialData(TfLiteTensor *input);

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

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  // NOLINTNEXTLINE(runtime-global-variables)
  // static tflite::AllOpsResolver op_resolver;
  static tflite::MicroMutableOpResolver<5> op_resolver;
  op_resolver.AddConv2D();
  op_resolver.AddRelu();
  op_resolver.AddMaxPool2D();
  op_resolver.AddConcatenation();
  op_resolver.AddResizeNearestNeighbor();
  // op_resolver.AddTransposeConv();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.begin(9600);
}

// The name of this function is important for Arduino compatibility.
void loop()
{
  // GetModelInfo(error_reporter, input, output);

  unsigned long loopStartTime = millis();
  //  // Get image from provider.
  //   if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels, input)) {
  //     TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  //   }
  GetSerialData(input);

  unsigned long inferenceStartTime = millis();
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke())
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }
  int inferenceTime = millis() - inferenceStartTime;

  // Process the inference results.
  count = PeopleCount(output, countFactor);
  int loopTime = millis() - loopStartTime;

  // Serial.println();
  // Serial.print("People count: ");
  Serial.println(count, 4);
  // Serial.print("Inference time [ms]: ");
  Serial.println(inferenceTime);
  // Serial.print("Loop time [ms]: ");
  // Serial.println(loopTime);
  // Serial.println();

  // PrintOutput(error_reporter, output);
  for (size_t i = 0; i < kMaxImageSize; i++)
  {
    Serial.print((output->data.int8[i] - output->params.zero_point) * output->params.scale, 8);
    Serial.print(" ");
  }
  Serial.println();
}

void GetSerialData(TfLiteTensor *input)
{
  // char msg[768];
  String msg;
  String subMsg;
  char *pch;
  float float_value = 0.0;
  int step = 0;
  int index;
  bool dataCapturing = true;

  const int BUFFER_SIZE = 768;

  while (dataCapturing)
  {
    if (Serial.available() > 0)
    {
      msg = Serial.readString();
      // Serial.println(msg);

      if (sizeof(msg) / sizeof(char) > 2)
      {
        for (step = 0; step < BUFFER_SIZE; step++)
        {
          index = msg.indexOf(",");
          subMsg = msg.substring(0, index).c_str();
          msg = msg.substring(index + 1);

          float_value = subMsg.toFloat();

          // pch = strtok(msg, ",");
          // float_value = (float)atof(pch);

          input->data.int8[step] = float_value / input->params.scale + input->params.zero_point;
        }
        dataCapturing = false;
      }
    }
  }
}
