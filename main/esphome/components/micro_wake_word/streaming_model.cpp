#include "streaming_model.h"

#include "esp_log.h"
#include "helpers.h"

static const char *const TAG = "micro_wake_word";

namespace esphome
{
    namespace micro_wake_word
    {

        void WakeWordModel::log_model_config()
        {
            ESP_LOGI(TAG, "    - Wake Word: %s", this->wake_word_.c_str());
            ESP_LOGI(TAG, "      Probability cutoff: %.3f", this->probability_cutoff_);
            ESP_LOGI(TAG, "      Sliding window size: %d", this->sliding_window_size_);
        }

        void VADModel::log_model_config()
        {
            ESP_LOGI(TAG, "    - VAD Model");
            ESP_LOGI(TAG, "      Probability cutoff: %.3f", this->probability_cutoff_);
            ESP_LOGI(TAG, "      Sliding window size: %d", this->sliding_window_size_);
        }

        bool StreamingModel::load_model(tflite::MicroMutableOpResolver<20> &op_resolver)
        {
            ExternalRAMAllocator<uint8_t> arena_allocator(ExternalRAMAllocator<uint8_t>::ALLOW_FAILURE);

            if (this->tensor_arena_ == nullptr) {
                this->tensor_arena_ = arena_allocator.allocate(this->tensor_arena_size_);
                if (this->tensor_arena_ == nullptr) {
                    ESP_LOGE(TAG, "Could not allocate the streaming model's tensor arena.");
                    return false;
                }
            }

            if (this->var_arena_ == nullptr) {
                this->var_arena_ = arena_allocator.allocate(STREAMING_MODEL_VARIABLE_ARENA_SIZE);
                if (this->var_arena_ == nullptr) {
                    ESP_LOGE(TAG, "Could not allocate the streaming model's variable tensor arena.");
                    return false;
                }
                this->ma_ = tflite::MicroAllocator::Create(this->var_arena_, STREAMING_MODEL_VARIABLE_ARENA_SIZE);
                this->mrv_ = tflite::MicroResourceVariables::Create(this->ma_, 20);
            }

            const tflite::Model *model = tflite::GetModel(this->model_start_);
            if (model->version() != TFLITE_SCHEMA_VERSION) {
                ESP_LOGE(TAG, "Streaming model's schema is not supported");
                return false;
            }

            if (this->interpreter_ == nullptr) {
                this->interpreter_ =
                    make_unique<tflite::MicroInterpreter>(tflite::GetModel(this->model_start_), op_resolver,
                                                          this->tensor_arena_, this->tensor_arena_size_, this->mrv_);
                if (this->interpreter_->AllocateTensors() != kTfLiteOk) {
                    ESP_LOGE(TAG, "Failed to allocate tensors for the streaming model");
                    return false;
                }

                // Verify input tensor matches expected values
                // Dimension 3 will represent the first layer stride, so skip it may vary
                TfLiteTensor *input = this->interpreter_->input(0);
                
                // 调试：记录输入张量详细信息
                ESP_LOGI(TAG, "=== ESPHOME INPUT TENSOR DEBUG ===");
                ESP_LOGI(TAG, "Input tensor info:");
                ESP_LOGI(TAG, "  Type: %d", input->type);
                ESP_LOGI(TAG, "  Dims: %d", input->dims->size);
                for (int i = 0; i < input->dims->size; i++) {
                    ESP_LOGI(TAG, "  Dim[%d]: %d", i, input->dims->data[i]);
                }
                ESP_LOGI(TAG, "  Bytes: %d", input->bytes);
                ESP_LOGI(TAG, "  Expected feature size: %d", PREPROCESSOR_FEATURE_SIZE);
                
                if ((input->dims->size != 3) || (input->dims->data[0] != 1) ||
                    (input->dims->data[2] != PREPROCESSOR_FEATURE_SIZE)) {
                    ESP_LOGE(TAG, "Streaming model tensor input dimensions has improper dimensions.");
                    return false;
                }

                if (input->type != kTfLiteInt8) {
                    ESP_LOGE(TAG, "Streaming model tensor input is not int8.");
                    return false;
                }

                // Verify output tensor matches expected values
                TfLiteTensor *output = this->interpreter_->output(0);
                if ((output->dims->size != 2) || (output->dims->data[0] != 1) || (output->dims->data[1] != 1)) {
                    ESP_LOGE(TAG, "Streaming model tensor output dimension is not 1x1.");
                }

                if (output->type != kTfLiteUInt8) {
                    ESP_LOGE(TAG, "Streaming model tensor output is not uint8.");
                    return false;
                }
            }

            ESP_LOGI(TAG, "Actual tensor arena size is %d", this->interpreter_->arena_used_bytes());
            
            // === 调试：详细的张量竞技场信息 ===
            ESP_LOGI(TAG, "=== ESPHOME TENSOR ARENA DEBUG ===");
            ESP_LOGI(TAG, "Allocated tensor arena size: %d bytes", (int)this->tensor_arena_size_);
            ESP_LOGI(TAG, "Used tensor arena size: %d bytes", this->interpreter_->arena_used_bytes());
            ESP_LOGI(TAG, "Variable arena size: %d bytes", STREAMING_MODEL_VARIABLE_ARENA_SIZE);

            return true;
        }

        void StreamingModel::unload_model()
        {
            this->interpreter_.reset();

            ExternalRAMAllocator<uint8_t> arena_allocator(ExternalRAMAllocator<uint8_t>::ALLOW_FAILURE);

            arena_allocator.deallocate(this->tensor_arena_, this->tensor_arena_size_);
            this->tensor_arena_ = nullptr;
            arena_allocator.deallocate(this->var_arena_, STREAMING_MODEL_VARIABLE_ARENA_SIZE);
            this->var_arena_ = nullptr;
        }

        bool StreamingModel::perform_streaming_inference(const int8_t features[PREPROCESSOR_FEATURE_SIZE])
        {
            if (this->interpreter_ != nullptr) {
                TfLiteTensor *input = this->interpreter_->input(0);

                std::memmove((int8_t *)(tflite::GetTensorData<int8_t>(input)) +
                                 PREPROCESSOR_FEATURE_SIZE * this->current_stride_step_,
                             features, PREPROCESSOR_FEATURE_SIZE);
                ++this->current_stride_step_;

                uint8_t stride = this->interpreter_->input(0)->dims->data[1];

                if (this->current_stride_step_ >= stride) {
                    this->current_stride_step_ = 0;

                    TfLiteStatus invoke_status = this->interpreter_->Invoke();
                    if (invoke_status != kTfLiteOk) {
                        ESP_LOGW(TAG, "Streaming interpreter invoke failed");
                        return false;
                    }

                    TfLiteTensor *output = this->interpreter_->output(0);

                    // === 调试：详细分析模型输出 ===
                    static bool logged_output_info = false;
                    if (!logged_output_info) {
                        ESP_LOGI(TAG, "=== ESPHOME MODEL OUTPUT DEBUG ===");
                        ESP_LOGI(TAG, "Output tensor type: %d", output->type);
                        ESP_LOGI(TAG, "Output tensor dims: %d", output->dims->size);
                        for (int i = 0; i < output->dims->size; i++) {
                            ESP_LOGI(TAG, "  Dim[%d]: %d", i, output->dims->data[i]);
                        }
                        ESP_LOGI(TAG, "Output tensor bytes: %d", output->bytes);
                        
                        // 检查量化参数
                        if (output->type == kTfLiteUInt8) {
                            ESP_LOGI(TAG, "UInt8 quantization params: scale=%.6f, zero_point=%d", 
                                    output->params.scale, output->params.zero_point);
                        }
                        logged_output_info = true;
                    }

                    uint8_t raw_probability = output->data.uint8[0];
                    
                    // 调试：记录模型输出的详细信息
                    static int inference_count = 0;
                    inference_count++;
                    if (inference_count % 100 == 0 || raw_probability > 20) {
                        ESP_LOGI(TAG, "ESPHome inference #%d: raw_uint8=%d (%.6f), scale=%.6f, zero_point=%d", 
                                inference_count, raw_probability, raw_probability/255.0f,
                                output->params.scale, output->params.zero_point);
                        
                        if (output->params.scale != 0.0f) {
                            float dequantized = (raw_probability - output->params.zero_point) * output->params.scale;
                            ESP_LOGI(TAG, "Dequantized probability: %.6f", dequantized);
                        }
                    }

                    // === 修复：使用正确的量化参数 ===
                    // 将UInt8值转换为实际概率值，然后重新量化为0-255范围
                    float actual_probability = (raw_probability - output->params.zero_point) * output->params.scale;
                    uint8_t normalized_probability = (uint8_t)(actual_probability * 255.0f);
                    
                    // 调试：对比原始和修正后的值
                    if (inference_count % 100 == 0 || raw_probability > 20) {
                        ESP_LOGI(TAG, "Original formula: %.6f, Corrected formula: %.6f, normalized: %d", 
                                raw_probability/255.0f, actual_probability, normalized_probability);
                    }

                    ++this->last_n_index_;
                    if (this->last_n_index_ == this->sliding_window_size_) {
                        this->last_n_index_ = 0;
                    }
                    this->recent_streaming_probabilities_[this->last_n_index_] = normalized_probability;
                }
                return true;
            }
            ESP_LOGE(TAG, "Streaming interpreter is not initialized.");
            return false;
        }

        void StreamingModel::reset_probabilities()
        {
            for (auto &prob : this->recent_streaming_probabilities_) {
                prob = 0;
            }
        }

        WakeWordModel::WakeWordModel(const uint8_t *model_start, float probability_cutoff,
                                     size_t sliding_window_average_size, const std::string &wake_word,
                                     size_t tensor_arena_size)
        {
            this->model_start_ = model_start;
            this->probability_cutoff_ = probability_cutoff;
            this->sliding_window_size_ = sliding_window_average_size;
            this->recent_streaming_probabilities_.resize(sliding_window_average_size, 0);
            this->wake_word_ = wake_word;
            this->tensor_arena_size_ = tensor_arena_size;
        };

        bool WakeWordModel::determine_detected()
        {
            uint32_t sum = 0;
            for (auto &prob : this->recent_streaming_probabilities_) {
                sum += prob;
            }

            // 修正：使用正确的量化参数而不是 255
            // 假设scale=0.003906, zero_point=0，那么正确的公式是: (raw_value - 0) * 0.003906
            float scale = 0.003906f;  // 1.0f / 256.0f
            int zero_point = 0;
            
            float sliding_window_average = 0.0f;
            for (auto &prob : this->recent_streaming_probabilities_) {
                sliding_window_average += (prob - zero_point) * scale;
            }
            sliding_window_average /= this->sliding_window_size_;

            // Detect the wake word if the sliding window average is above the cutoff
            if (sliding_window_average > this->probability_cutoff_) {
                ESP_LOGI(TAG,
                         "ESPHome CORRECTED: The '%s' model sliding average probability is %.6f and most recent "
                         "probability is %.6f (raw=%d)",
                         this->wake_word_.c_str(), sliding_window_average,
                         (this->recent_streaming_probabilities_[this->last_n_index_] - zero_point) * scale,
                         this->recent_streaming_probabilities_[this->last_n_index_]);
                return true;
            } else {
                // 每100次记录一次调试信息
                static int debug_count = 0;
                debug_count++;
                if (debug_count % 100 == 0) {
                    ESP_LOGI(TAG,
                             "ESPHome CORRECTED: The '%s' model sliding average probability is %.6f and most recent "
                             "probability is %.6f (raw=%d)",
                             this->wake_word_.c_str(), sliding_window_average,
                             (this->recent_streaming_probabilities_[this->last_n_index_] - zero_point) * scale,
                             this->recent_streaming_probabilities_[this->last_n_index_]);
                }
            }
            return false;
        }

        VADModel::VADModel(const uint8_t *model_start, float probability_cutoff, size_t sliding_window_size,
                           size_t tensor_arena_size)
        {
            this->model_start_ = model_start;
            this->probability_cutoff_ = probability_cutoff;
            this->sliding_window_size_ = sliding_window_size;
            this->recent_streaming_probabilities_.resize(sliding_window_size, 0);
            this->tensor_arena_size_ = tensor_arena_size;
        };

        bool VADModel::determine_detected()
        {
            uint32_t sum = 0;
            for (auto &prob : this->recent_streaming_probabilities_) {
                sum += prob;
            }

            float sliding_window_average =
                static_cast<float>(sum) / static_cast<float>(255 * this->sliding_window_size_);

            return sliding_window_average > this->probability_cutoff_;
        }

    } // namespace micro_wake_word
} // namespace esphome
