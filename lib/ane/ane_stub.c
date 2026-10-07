/* Linux stub for the ANE prefill offload (lib/ane ane_bridge.m/ane_mlp.m are
 * Objective-C against the private AppleNeuralEngine framework, macOS only).
 * Every symbol answers "unavailable" so the opt-in --ane-prefill path, which
 * gates on sushi_ane_available(), never engages. */
#include <stddef.h>
#include <stdint.h>

typedef struct sushi_ane_mlp sushi_ane_mlp;
typedef struct sushi_ane_bank sushi_ane_bank;
typedef struct sushi_ane_plane sushi_ane_plane;

int sushi_ane_available(void) { return 0; }
uint64_t sushi_ane_internal_free_disk(void) { return 0; }

void sushi_ane_cache_lineage(const char *group, const char *variant) {
    (void)group; (void)variant;
}
void sushi_ane_cache_variant(const char *group, char *out, int out_len) {
    (void)group;
    if (out && out_len > 0) out[0] = '\0';
}

sushi_ane_plane *sushi_ane_plane_create(size_t bytes) { (void)bytes; return NULL; }
void sushi_ane_plane_free(sushi_ane_plane *p) { (void)p; }
__fp16 *sushi_ane_plane_base(sushi_ane_plane *p) { (void)p; return NULL; }

sushi_ane_bank *sushi_ane_bank_create(void) { return NULL; }
void sushi_ane_bank_free(sushi_ane_bank *b) { (void)b; }
uint32_t sushi_ane_bank_count(const sushi_ane_bank *b) { (void)b; return 0; }
uint64_t sushi_ane_bank_bytes(const sushi_ane_bank *b) { (void)b; return 0; }

int sushi_ane_bank_add_mlp(sushi_ane_bank *b, uint32_t hidden, uint32_t ffn,
                           uint32_t rows,
                           const int8_t *gate_q, const float *gate_s,
                           const int8_t *up_q, const float *up_s,
                           const int8_t *down_q, const float *down_s,
                           char *error, size_t error_size) {
    (void)b; (void)hidden; (void)ffn; (void)rows;
    (void)gate_q; (void)gate_s; (void)up_q; (void)up_s; (void)down_q; (void)down_s;
    if (error && error_size > 0) {
        const char *msg = "ANE unavailable on this platform";
        size_t i = 0;
        for (; msg[i] && i + 1 < error_size; i++) error[i] = msg[i];
        error[i] = '\0';
    }
    return -1;
}

int sushi_ane_bank_add_gdn(sushi_ane_bank *b, uint32_t hidden, uint32_t qkv_out,
                           uint32_t z_out, uint32_t rows,
                           const int8_t *qkv_q, const float *qkv_s,
                           const int8_t *z_q, const float *z_s,
                           char *error, size_t error_size) {
    (void)b; (void)hidden; (void)qkv_out; (void)z_out; (void)rows;
    (void)qkv_q; (void)qkv_s; (void)z_q; (void)z_s;
    if (error && error_size > 0) {
        const char *msg = "ANE unavailable on this platform";
        size_t i = 0;
        for (; msg[i] && i + 1 < error_size; i++) error[i] = msg[i];
        error[i] = '\0';
    }
    return -1;
}

sushi_ane_mlp *sushi_ane_bank_finish(sushi_ane_bank *b, const char *name,
                                     int ane_instance,
                                     sushi_ane_plane *input_plane,
                                     sushi_ane_plane *output_plane,
                                     char *error, size_t error_size) {
    (void)b; (void)name; (void)ane_instance; (void)input_plane; (void)output_plane;
    if (error && error_size > 0) {
        const char *msg = "ANE unavailable on this platform";
        size_t i = 0;
        for (; msg[i] && i + 1 < error_size; i++) error[i] = msg[i];
        error[i] = '\0';
    }
    return NULL;
}

void sushi_ane_mlp_free(sushi_ane_mlp *m) { (void)m; }
__fp16 *sushi_ane_mlp_input(sushi_ane_mlp *m) { (void)m; return NULL; }
__fp16 *sushi_ane_mlp_output(sushi_ane_mlp *m) { (void)m; return NULL; }
int sushi_ane_mlp_eval(sushi_ane_mlp *m, uint32_t procedure, char *error,
                       size_t error_size) {
    (void)m; (void)procedure;
    if (error && error_size > 0) {
        const char *msg = "ANE unavailable on this platform";
        size_t i = 0;
        for (; msg[i] && i + 1 < error_size; i++) error[i] = msg[i];
        error[i] = '\0';
    }
    return -1;
}
double sushi_ane_mlp_compile_seconds(const sushi_ane_mlp *m) { (void)m; return 0.0; }
int sushi_ane_mlp_cache_hit(const sushi_ane_mlp *m) { (void)m; return 0; }
