#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Basic C API declarations
int kpu_initialize(void);
void kpu_shutdown(void);
int kpu_is_initialized(void);

#ifdef __cplusplus
}
#endif