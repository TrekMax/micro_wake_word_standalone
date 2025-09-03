#include <driver/i2s_std.h>
#include <stdio.h>

#include "i2s_audio_microphone.h"
#include "micro_wake_word.h"
#include "streaming_model.h"

// tflite model for "Hey Jarvis"
// #include "hey_jarvis.h"
extern const uint8_t model_alexa_start[] asm("_binary_alexa_tflite_start");
extern const uint8_t model_alexa_end[]   asm("_binary_alexa_tflite_end");
// extern const uint8_t model_nihaowenwen_start[] asm("_binary_nihaowenwen_tflite_start");
// extern const uint8_t model_nihaowenwen_end[]   asm("_binary_nihaowenwen_tflite_end");
// extern const uint8_t model_nihaowenwen_v2_start[] asm("_binary_nihaowenwen_v2_tflite_start");
// extern const uint8_t model_nihaowenwen_v2_end[]   asm("_binary_nihaowenwen_v2_tflite_end");
// extern const uint8_t model_xiaolingxiaoling_start[] asm("_binary_xiaolingxiaoling_tflite_start");
// extern const uint8_t model_xiaolingxiaoling_end[]   asm("_binary_xiaolingxiaoling_tflite_end");
// extern const uint8_t model_XLXL_V3_start[] asm("_binary_XLXL_V3_tflite_start");
// extern const uint8_t model_XLXL_V3_end[]   asm("_binary_XLXL_V3_tflite_end");
extern const uint8_t model_XLXL_V4_start[] asm("_binary_XLXL_V4_tflite_start");
extern const uint8_t model_XLXL_V4_end[]   asm("_binary_XLXL_V4_tflite_end");

// INMP441 microphone
#define I2S_BCK_PIN    GPIO_NUM_1 //SCK
#define I2S_SD_PIN     GPIO_NUM_2 //SD
#define I2S_WS_PIN     GPIO_NUM_4 //WS
// GND
// VCC - 3.3V

#define I2S_PORT       I2S_NUM_0
#define SAMPLE_RATE_HZ 16000

bool gDetected = false;

void wakeWordDetected(std::string detected_wake_word);
void wakeWordDetectionTask(void *params);

void wakeWordDetected(std::string detected_wake_word)
{
    gDetected = true;
}

void wakeWordDetectionTask(void *params)
{
    esphome::i2s_audio::I2SAudioMicrophone microphone;
    // setup and initialize the microphone
    microphone.set_bclk_pin(I2S_BCK_PIN);
    microphone.set_lrclk_pin(I2S_WS_PIN);
    microphone.set_din_pin(I2S_SD_PIN);
    microphone.set_channel(I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER));
    microphone.set_sample_rate(SAMPLE_RATE_HZ);
    microphone.set_bits_per_sample(I2S_DATA_BIT_WIDTH_32BIT);

    // set up the wake word detector
    esphome::micro_wake_word::MicroWakeWord wakeWord;
    // uint8_t *model = const_cast<uint8_t *>(hey_jarvis_tflite);
    uint8_t *model_alexa = const_cast<uint8_t *>(model_alexa_start);
    // uint8_t *model = const_cast<uint8_t *>(model_nihaowenwen_start);
    // uint8_t *model = const_cast<uint8_t *>(model_nihaowenwen_v2_start);
    // uint8_t *model_xiaoling = const_cast<uint8_t *>(model_xiaolingxiaoling_start);
    // uint8_t *model_xlxl_v3 = const_cast<uint8_t *>(model_XLXL_V3_start);
    uint8_t *model_xlxl_v4 = const_cast<uint8_t *>(model_XLXL_V4_start);

    wakeWord.set_microphone(&microphone);
    // wakeWord.add_wake_word_model(model, 0.97f, 5, "Hey Jarvis", 22940);
    wakeWord.add_wake_word_model(model_alexa, 0.97f, 5, "Alexa", 22348);
    // wakeWord.add_wake_word_model(model, 0.56f, 5, "你好问问", 40000); // Increased from 25000 to 40000 to support model memory needs
    // wakeWord.add_wake_word_model(model, 0.5f, 5, "你好问问", 60000); // Increased from 25000 to 40000 to support model memory needs
    // wakeWord.add_wake_word_model(model_xiaoling, 0.5f, 5, "小聆小聆", 60000);
    // wakeWord.add_wake_word_model(model_xlxl_v3, 0.9f, 5, "小聆小聆V3", 60000);
    wakeWord.add_wake_word_model(model_xlxl_v4, 0.9f, 5, "小聆小聆V4", 70*1024);
    wakeWord.set_features_step_size(10);
    wakeWord.add_detection_callback(&wakeWordDetected);

    wakeWord.setup();
    wakeWord.start();

    for (;;) {
        wakeWord.loop();
        if (gDetected) {
            printf("Wake word detected!\n");
            gDetected = false;
            // start listening again
            wakeWord.start();
        }
    }
}

extern "C" void app_main(void)
{
    xTaskCreatePinnedToCore(wakeWordDetectionTask, "wake_word_task", 4096, NULL, 2, NULL, 1);
}
