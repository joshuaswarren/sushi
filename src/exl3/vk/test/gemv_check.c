/* gemv_check.c — three-way cross-check for the M2 GEMV stage, row 0 slot 0
 * of oracle_moe.bin case pack-r1: C re-implementation of prepareInput +
 * innerGemv + finishOutput (straight from expert_exl3.zig), printed next to
 * the oracle's expected ag bits and the shader's dumped ag bits. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint8_t *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc(*len);
    if (fread(p, 1, *len, f) != *len) exit(1);
    fclose(f);
    return p;
}

static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static float rd32f(const uint8_t *p) { float v; memcpy(&v, p, 4); return v; }

static float f16ToF32(uint32_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) { bits = sign << 31; }
        else {
            uint32_t m = man, s = 0;
            while ((m & 0x200) == 0) { m <<= 1; s++; }
            bits = (sign << 31) | ((112 - s) << 23) | ((m & 0x1FF) << 14);
        }
    } else if (exp == 0x1F) { bits = (sign << 31) | (0xFFu << 23) | (man << 13); }
    else { bits = (sign << 31) | ((exp + 112) << 23) | (man << 13); }
    float v; memcpy(&v, &bits, 4); return v;
}

static uint32_t f32bits(float v) { uint32_t b; memcpy(&b, &v, 4); return b; }
static float bitsf32(uint32_t b) { float v; memcpy(&v, &b, 4); return v; }

static uint32_t f32ToF16(float v) {
    uint32_t b = f32bits(v);
    uint32_t sign = (b >> 16) & 0x8000, exp = (b >> 23) & 0xFF, man = b & 0x7FFFFF;
    if (exp == 0xFF) return sign | 0x7C00 | (man ? 0x200 : 0);
    int ne = (int)exp - 127 + 15;
    if (ne >= 31) return sign | 0x7C00;
    if (ne <= 0) {
        if (ne < -10) return sign;
        uint32_t m = man | 0x800000, shift = 14 - ne;
        uint32_t rem = m & ((1u << shift) - 1), h = m >> shift;
        if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (h & 1))) h++;
        return sign | h;
    }
    uint32_t h = ((uint32_t)ne << 10) | (man >> 13), rem = man & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && ((man >> 13) & 1))) h++;
    if (h >= 0x7C00) return sign | 0x7C00;
    return sign | h;
}

static uint32_t decodeMcg(uint32_t cw) {
    uint32_t mixed = cw * 0xCBAC1FEDu;
    uint32_t pair = 0x3B603B60u ^ (mixed & 0x8FFF8FFFu);
    float lo = f16ToF32(pair & 0xFFFF), hi = f16ToF32(pair >> 16);
    return f32ToF16(lo + hi);
}

static uint32_t permOf(uint32_t i) {
    uint32_t thread = i >> 3, j = i & 7;
    uint32_t row0 = (thread & 3) << 1, col0 = thread >> 2, col1 = col0 + 8;
    uint32_t rows[4] = { row0, row0 + 1, row0 + 8, row0 + 9 };
    uint32_t col = (j < 4) ? col0 : col1;
    return rows[j & 3] * 16 + col;
}

static const uint32_t *g_words;
static void decodeTile(uint32_t wordBase, uint32_t wc, uint32_t total, uint32_t mask, uint32_t *tw) {
    for (uint32_t t = 0; t < 128; t++) {
        uint32_t e0 = ((t * 2 + 1) * 32) >> 4, e1 = ((t * 2 + 2) * 32) >> 4;
        uint32_t bit0 = e0 + total - 16, bit2 = e1 + total;
        uint32_t i0 = bit0 >> 5, i1 = (bit2 - 1) >> 5;
        uint32_t shift = (i1 + 1) * 32 - bit2;
        uint32_t wa = g_words[wordBase + (i0 % wc)], wb = g_words[wordBase + (i1 % wc)];
        uint32_t funnel = (shift == 0) ? wb : (shift >= 32) ? wa
                        : ((wa << (32 - shift)) | (wb >> shift));
        uint32_t fresh = e1 - e0;
        tw[permOf(t * 2)] = decodeMcg(((funnel >> fresh) & 0xFFFF) & mask);
        tw[permOf(t * 2 + 1)] = decodeMcg((funnel & 0xFFFF) & mask);
    }
}

#define HAD_SCALE 0.08838834764831845

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s oracle_moe.bin\n", argv[0]); return 2; }
    size_t mlen; uint8_t *mb = slurp(argv[1], &mlen);
    size_t pos = 12;
    /* walk to case index 1 (pack-r1) */
    for (int cs = 0; cs < 2; cs++) {
        pos += 16 + 9 * 4;
        uint32_t rows, hidden, inter, experts, topk, n;
        memcpy(&rows, mb + pos - 36, 4);
        memcpy(&hidden, mb + pos - 32, 4);
        memcpy(&inter, mb + pos - 28, 4);
        memcpy(&experts, mb + pos - 24, 4);
        memcpy(&topk, mb + pos - 20, 4);
        memcpy(&n, mb + pos - 16, 4);
        size_t rts = rows * topk, ts = (hidden / 16) * (inter / 16) * n;
        size_t s_x = rows * hidden * 4, s_slots = rts * 4, s_scores = rts * 4;
        size_t s_t = experts * ts * 2;
        size_t s_suh = experts * hidden * 2, s_suhd = experts * inter * 2;
        size_t s_svh = experts * inter * 2, s_svhd = experts * hidden * 2;
        size_t s_mid = rts * inter * 2, s_slot = rts * hidden * 2;
        size_t s_o32 = rows * hidden * 4, s_o16 = rows * hidden * 2;
        size_t blob = s_x + s_slots + s_scores + 3 * s_t + 2 * s_suh + s_suhd
                    + 2 * s_svh + s_svhd + 3 * s_mid + 2 * s_slot
                    + s_o32 + 2 * s_o16 + s_o32 + 2 * s_o16;
        if (cs == 0) { pos += blob; continue; }
        size_t off_x = pos, off_slots = off_x + s_x;
        size_t off_tg = off_slots + s_slots + s_scores;
        size_t off_suhg = off_tg + 3 * s_t;
        size_t off_svhg = off_suhg + 2 * s_suh + s_suhd;
        size_t off_ag = off_svhg + s_svh + s_svh + s_svhd;
        size_t off_ag_want = off_ag; /* same place: R1 expected ag */

        uint32_t e = rd32f ? 0 : 0;
        (void)e;
        uint32_t slot0; memcpy(&slot0, mb + off_slots, 4);
        printf("slot0 expert = %u\n", slot0);

        /* unpack trellis words */
        size_t nwords = experts * ts / 2;
        uint32_t *words = malloc(nwords * 4);
        for (size_t i = 0; i < nwords; i++)
            words[i] = (uint32_t)rd16(mb + off_tg + 4 * i) | ((uint32_t)rd16(mb + off_tg + 4 * i + 2) << 16);
        g_words = words;

        /* suh for expert e (u32/u16) */
        size_t su = slot0 * hidden;
        uint32_t *suh = malloc(hidden * 4), *svh = malloc(inter * 4);
        for (size_t i = 0; i < hidden; i++) suh[i] = rd16(mb + off_suhg + 2 * (su + i));
        for (size_t i = 0; i < inter; i++) svh[i] = rd16(mb + off_svhg + 2 * (slot0 * inter + i));

        /* prepareInput in C */
        float *xt = malloc(hidden * 4);
        for (size_t u = 0; u < hidden; u++) {
            float x = rd32f(mb + off_x + 4 * u);
            xt[u] = f16ToF32(f32ToF16(x)) * f16ToF32(suh[u]);
        }
        for (size_t blk = 0; blk < hidden / 128; blk++) {
            float out[128];
            for (size_t r = 0; r < 128; r++) {
                float acc = 0.0f;
                for (size_t k = 0; k < 128; k++) {
                    float he = (__builtin_popcount(r & k) & 1) ? -HAD_SCALE : HAD_SCALE;
                    acc += he * xt[blk * 128 + k];
                }
                out[r] = acc;
            }
            for (size_t r = 0; r < 128; r++)
                xt[blk * 128 + r] = f16ToF32(f32ToF16(out[r]));
        }
        printf("xt[0..5] bits : %04X %04X %04X %04X %04X %04X\n",
               f32ToF16(xt[0]), f32ToF16(xt[1]), f32ToF16(xt[2]),
               f32ToF16(xt[3]), f32ToF16(xt[4]), f32ToF16(xt[5]));

        /* tile (0,0) words for comparison with the shader decode probe */
        uint32_t tw0[256];
        decodeTile((uint32_t)(slot0 * ts / 2), 16, 512, 0x7FFFu, tw0);
        printf("C tw[0..7]    : %04X %04X %04X %04X %04X %04X %04X %04X\n",
               tw0[0], tw0[1], tw0[2], tw0[3], tw0[4], tw0[5], tw0[6], tw0[7]);
        /* per-(r,c) derivation (the shader's new path) vs placed decode */
        {
            uint32_t bad = 0;
            for (uint32_t r = 0; r < 16; r++) {
                for (uint32_t c = 0; c < 16; c++) {
                    uint32_t r0, rowIdx;
                    if (r < 8) { r0 = r & ~1u; rowIdx = r & 1u; }
                    else { r0 = (r & ~1u) - 8; rowIdx = 2 + (r & 1u); }
                    uint32_t col0 = (c >= 8) ? c - 8 : c;
                    uint32_t thread = col0 * 4 + (r0 >> 1);
                    uint32_t t = thread * 8 + rowIdx + ((c >= 8) ? 4 : 0);
                    uint32_t T = t >> 1;
                    uint32_t e0 = ((T * 2 + 1) * 32) >> 4, e1 = ((T * 2 + 2) * 32) >> 4;
                    uint32_t bit0 = e0 + 512 - 16, bit2 = e1 + 512;
                    uint32_t i0 = bit0 >> 5, i1 = (bit2 - 1) >> 5;
                    uint32_t shift = (i1 + 1) * 32 - bit2;
                    uint32_t wa = words[slot0 * ts / 2 + (i0 % 16)], wb = words[slot0 * ts / 2 + (i1 % 16)];
                    uint32_t funnel = (shift == 0) ? wb : ((wa << (32 - shift)) | (wb >> shift));
                    uint32_t fresh = e1 - e0;
                    uint32_t cw = (t & 1) ? ((funnel & 0xFFFF) & 0x7FFF)
                                          : (((funnel >> fresh) & 0xFFFF) & 0x7FFF);
                    uint32_t val = decodeMcg(cw);
                    if (val != tw0[r * 16 + c] && bad < 4)
                        printf("valAt mismatch r=%u c=%u t=%u: %04X vs %04X\n",
                               r, c, t, val, tw0[r * 16 + c]);
                    if (val != tw0[r * 16 + c]) bad++;
                }
            }
            printf("valAt mismatches: %u/256\n", bad);
        }
        /* one-tile partial: acc[c] = sum_r xt[r] * tw[r*16+c] */
        printf("C acc1[0..7]  : %04X %04X %04X %04X %04X %04X %04X %04X\n",
               f32ToF16(xt[0] * f16ToF32(tw0[0]) + xt[1] * f16ToF32(tw0[16]) + xt[2] * f16ToF32(tw0[32]) + xt[3] * f16ToF32(tw0[48])
                      + xt[4] * f16ToF32(tw0[64]) + xt[5] * f16ToF32(tw0[80]) + xt[6] * f16ToF32(tw0[96]) + xt[7] * f16ToF32(tw0[112])),
               f32ToF16(xt[0] * f16ToF32(tw0[1]) + xt[1] * f16ToF32(tw0[17]) + xt[2] * f16ToF32(tw0[33]) + xt[3] * f16ToF32(tw0[49])
                      + xt[4] * f16ToF32(tw0[65]) + xt[5] * f16ToF32(tw0[81]) + xt[6] * f16ToF32(tw0[97]) + xt[7] * f16ToF32(tw0[113])),
               f32ToF16(xt[0] * f16ToF32(tw0[2]) + xt[1] * f16ToF32(tw0[18]) + xt[2] * f16ToF32(tw0[34]) + xt[3] * f16ToF32(tw0[50])
                      + xt[4] * f16ToF32(tw0[66]) + xt[5] * f16ToF32(tw0[82]) + xt[6] * f16ToF32(tw0[98]) + xt[7] * f16ToF32(tw0[114])),
               f32ToF16(xt[0] * f16ToF32(tw0[3]) + xt[1] * f16ToF32(tw0[19]) + xt[2] * f16ToF32(tw0[35]) + xt[3] * f16ToF32(tw0[51])
                      + xt[4] * f16ToF32(tw0[67]) + xt[5] * f16ToF32(tw0[83]) + xt[6] * f16ToF32(tw0[99]) + xt[7] * f16ToF32(tw0[115])),
               0x3F00u, 0x3F00u, 0x3F00u, 0x3F00u);

        /* innerGemv + finishOutput in C for the gate projection */
        uint32_t wc = 16, total = 512, tw[256];
        float *ag = calloc(inter, 4);
        for (size_t tk = 0; tk < hidden / 16; tk++) {
            for (size_t tn = 0; tn < inter / 16; tn++) {
                decodeTile((uint32_t)(slot0 * ts / 2 + (tk * (inter / 16) + tn) * 16), wc, total, 0x7FFFu, tw);
                for (size_t r = 0; r < 16; r++) {
                    float xv = xt[tk * 16 + r];
                    for (size_t c = 0; c < 16; c++)
                        ag[tn * 16 + c] += xv * f16ToF32(tw[r * 16 + c]);
                }
            }
        }
        for (size_t o = 0; o < inter; o++) ag[o] = f16ToF32(f32ToF16(ag[o]));
        for (size_t blk = 0; blk < inter / 128; blk++) {
            float out[128];
            for (size_t r = 0; r < 128; r++) {
                float acc = 0.0f;
                for (size_t k = 0; k < 128; k++) {
                    float he = (__builtin_popcount(r & k) & 1) ? -HAD_SCALE : HAD_SCALE;
                    acc += he * ag[blk * 128 + k];
                }
                out[r] = acc;
            }
            for (size_t r = 0; r < 128; r++)
                ag[blk * 128 + r] = f16ToF32(f32ToF16(out[r])) * f16ToF32(svh[blk * 128 + r]);
        }
        printf("ag C bits     : %04X %04X %04X %04X %04X %04X %04X %04X\n",
               f32ToF16(ag[0]), f32ToF16(ag[1]), f32ToF16(ag[2]), f32ToF16(ag[3]),
               f32ToF16(ag[4]), f32ToF16(ag[5]), f32ToF16(ag[6]), f32ToF16(ag[7]));
        printf("ag oracle     : %04X %04X %04X %04X %04X %04X %04X %04X\n",
               rd16(mb + off_ag_want), rd16(mb + off_ag_want + 2),
               rd16(mb + off_ag_want + 4), rd16(mb + off_ag_want + 6),
               rd16(mb + off_ag_want + 8), rd16(mb + off_ag_want + 10),
               rd16(mb + off_ag_want + 12), rd16(mb + off_ag_want + 14));
        return 0;
    }
    return 1;
}
