#include "test_audio.h"
#include "test_audio_data.h"
#include <cstring>
#include <cstdint>

static int current_frame_index = 0;

bool init_test_audio() {
    current_frame_index = 0;
    return true;
}

bool has_more_test_frames() {
    return current_frame_index < TEST_FRAME_COUNT;
}

const int16_t* get_test_audio_frame(int frame_index) {
    if (frame_index >= 0 && frame_index < TEST_FRAME_COUNT) {
        return test_audio_data[frame_index];
    }
    return nullptr;
}

bool get_next_test_frame(int16_t* buffer, size_t buffer_size) {
    if (current_frame_index >= TEST_FRAME_COUNT || buffer_size < TEST_FRAME_SIZE) {
        return false;
    }
    
    // Copy frame data to buffer
    memcpy(buffer, test_audio_data[current_frame_index], TEST_FRAME_SIZE * sizeof(int16_t));
    current_frame_index++;
    
    return true;
}

bool reset_test_audio() {
    current_frame_index = 0;
    return true;
}

size_t get_total_test_frames() {
    return TEST_FRAME_COUNT;
}

size_t get_current_test_frame() {
    return current_frame_index;
}
