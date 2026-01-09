#include <Arduino.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#undef abs

#include "model_data.h"
#include "register_operators.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr int kTensorArenaSize = 200 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

void LogTensorPreview(const TfLiteTensor* tensor, const char* label) {
  MicroPrintf("%s preview:", label);
  constexpr int kMaxItems = 10;
  int count = 0;
  switch (tensor->type) {
    case kTfLiteFloat32:
      count = std::min(kMaxItems,
                       static_cast<int>(tensor->bytes / sizeof(float)));
      for (int i = 0; i < count; ++i) {
        MicroPrintf("  [%d] %f", i, tensor->data.f[i]);
      }
      break;
    case kTfLiteInt8:
      count = std::min(kMaxItems, static_cast<int>(tensor->bytes));
      for (int i = 0; i < count; ++i) {
        MicroPrintf("  [%d] %d", i, tensor->data.int8[i]);
      }
      break;
    case kTfLiteUInt8:
      count = std::min(kMaxItems, static_cast<int>(tensor->bytes));
      for (int i = 0; i < count; ++i) {
        MicroPrintf("  [%d] %u", i, tensor->data.uint8[i]);
      }
      break;
    default:
      MicroPrintf("  (unsupported tensor type %d)", tensor->type);
      break;
  }
}

TfLiteStatus LoadQuantModelAndPerformInference() {
  const tflite::Model* model = tflite::GetModel(g_model_tflm_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  tinyfm::GeneratedOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(tinyfm::RegisterGeneratedOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);
  std::memset(input->data.raw, 0, input->bytes);
  LogTensorPreview(input, "Input");

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  const TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);
  LogTensorPreview(output, "Output");
  return kTfLiteOk;
}
}  // namespace

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000) {
  }

  tflite::InitializeTarget();
  MicroPrintf("Starting TinyFM-HAR TFLM runtime...");

  if (LoadQuantModelAndPerformInference() != kTfLiteOk) {
    MicroPrintf("Model invocation failed.");
    return;
  }

  MicroPrintf("Model invocation succeeded.");
}

void loop() {
  delay(1000);
}
