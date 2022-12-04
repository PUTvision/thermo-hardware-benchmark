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

#include "model_utils.h"


void GetModelInfo(tflite::ErrorReporter* error_reporter, TfLiteTensor* input, TfLiteTensor* output){
    TF_LITE_REPORT_ERROR(error_reporter, "Dims: %d", input->dims->size);
    TF_LITE_REPORT_ERROR(error_reporter, "Shape: (%d %d %d %d)", input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);
    TF_LITE_REPORT_ERROR(error_reporter, "Type: %s", input->type);
    TF_LITE_REPORT_ERROR(error_reporter, "Input scale: %f", input->params.scale );
    TF_LITE_REPORT_ERROR(error_reporter, "Input zero point: %f", input->params.zero_point);
    TF_LITE_REPORT_ERROR(error_reporter, "Output scale: %f", output->params.scale );
    TF_LITE_REPORT_ERROR(error_reporter, "Output zero point: %f", output->params.zero_point);
}


float PeopleCount(TfLiteTensor* output, float count_factor){
    float output_num_float = 0;
    if (output->params.zero_point == 0 && output->params.scale == 0.0){
        for (size_t i = 0; i < kMaxImageSize; i++){
            output_num_float += output->data.f[i];
        }
    }
    else {
        for (int i = 0; i < kMaxImageSize; i++){
            output_num_float += (output->data.int8[i] - output->params.zero_point) * output->params.scale;
        }
    }

    return output_num_float / count_factor;
}


void PrintOutput(tflite::ErrorReporter* error_reporter, TfLiteTensor* output){
    if (output->params.zero_point == 0 && output->params.scale == 0.0){
        for (size_t i = 0; i < kMaxImageSize; i++){
            TF_LITE_REPORT_ERROR(error_reporter, "%f", output->data.f[i]);
        }
    }
    else {
        for (size_t i = 0; i < kMaxImageSize; i++){
            TF_LITE_REPORT_ERROR(error_reporter, "%f", (output->data.int8[i] - output->params.zero_point) * output->params.scale);
        }
    }
    TF_LITE_REPORT_ERROR(error_reporter, "\n\n");
}
