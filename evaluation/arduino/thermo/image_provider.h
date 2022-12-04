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

#ifndef TENSORFLOW_LITE_MICRO_THERMO_PRESENCE_DETECTION_IMAGE_PROVIDER_H_
#define TENSORFLOW_LITE_MICRO_THERMO_PRESENCE_DETECTION_IMAGE_PROVIDER_H_

#include "model_utils.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

extern const float x[kMaxImageSize];

TfLiteStatus GetImage(tflite::ErrorReporter *error_reporter, int image_width,
                      int image_height, int channels, TfLiteTensor *input);

#endif // TENSORFLOW_LITE_MICRO_THERMO_PRESENCE_DETECTION_IMAGE_PROVIDER_H_
