/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_THERMO_PRESENCE_DETECTION_MODEL_UTILS_H_
#define TENSORFLOW_LITE_MICRO_THERMO_PRESENCE_DETECTION_MODEL_UTILS_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Keeping these as constant expressions allow us to allocate fixed-sized arrays
// on the stack for our working memory.

// All of these values are derived from the values used during model training,
// if you change your model you'll need to update these constants.
constexpr int kNumCols = 32;
constexpr int kNumRows = 24;
constexpr int kNumChannels = 1;
constexpr float countFactor = 51.35;

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

void GetModelInfo(tflite::ErrorReporter* error_reporter, TfLiteTensor* input, TfLiteTensor* output);

float PeopleCount(TfLiteTensor* output, float count_factor);

void PrintOutput(tflite::ErrorReporter* error_reporter, TfLiteTensor* output);

#endif  // TENSORFLOW_LITE_MICRO_THERMO_PRESENCE_DETECTION_MODEL_UTILS_H_
