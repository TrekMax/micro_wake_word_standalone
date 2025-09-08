#pragma once

#include <cstdint>
#include <cstddef>

// Test audio interface
bool init_test_audio();
bool get_next_test_frame(int16_t* buffer, size_t buffer_size);
bool reset_test_audio();
size_t get_total_test_frames();
size_t get_current_test_frame();
