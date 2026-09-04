#ifndef DINOV3_LOOM_H
#define DINOV3_LOOM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Increment this when the ABI changes incompatibly. Callers must compare it
// before creating a session.
#define DINOV3_ABI_VERSION 1u

enum {
    DINOV3_OK = 0,
    DINOV3_ERROR = 1,
    DINOV3_INVALID_ARGUMENT = 64,
};

typedef struct dinov3_session dinov3_session;

uint32_t dinov3_abi_version(void);

// Creates an independent resident inference session. On failure, returns a
// non-zero status, leaves *out_session null and writes a NUL-terminated message
// to error (when error_capacity is non-zero).
int dinov3_create(const char *weights_dir, const char *kernels_dir,
                  int max_batch, dinov3_session **out_session, char *error,
                  size_t error_capacity);

// Runs one contiguous, patchified f32 batch. The element counts are part of the
// ABI so an undersized caller allocation is rejected before the GPU sees it.
// Calls using the same session are serialized internally. Destruction must not
// race a run.
int dinov3_run(dinov3_session *session, const float *input,
               size_t input_elements, float *output, size_t output_elements,
               int batch, char *error, size_t error_capacity);

int dinov3_max_batch(const dinov3_session *session);
void dinov3_destroy(dinov3_session *session);

#ifdef __cplusplus
}
#endif

#endif
