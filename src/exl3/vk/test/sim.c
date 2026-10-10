/* sim.c — replay the exl3_decode.comp f32 stages from oracle.bin under
 * different driver-arithmetic hypotheses and compare the mismatch signature
 * against the on-device (G13C) numbers:
 *   rowH128 exact 67659/294912, maxRel 1.5e25
 *   colH128 exact 45814/294912, maxRel 4.2e22
 *   final f16 290464/294912, final bf16 287280/294912
 * Usage: sim <oracle.bin> [mode]
 *   strict    IEEE mul+add (must reproduce the oracle bit-for-bit)
 *   fma       contract acc += he*x into fmaf(he,x,acc)
 *   ftz       flush every op result (and op inputs) that is denormal to zero
 *   ftzout    flush only op results
 *   fmaftz    fmaf + full flush
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define BLOCKS 18
#define EL 16384

typedef struct { uint32_t magic, version, n, mask, blocks; } Hdr;

static float f16ToF32(uint32_t h) {
    uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0u) {
        if (man == 0u) bits = sign << 31;
        else {
            uint32_t m = man, s = 0;
            while ((m & 0x200u) == 0u) { m <<= 1; s++; }
            bits = (sign << 31) | ((112u - s) << 23) | ((m & 0x1FFu) << 14);
        }
    } else if (exp == 0x1Fu) bits = (sign << 31) | (0xFFu << 23) | (man << 13);
    else bits = (sign << 31) | ((exp + 112u) << 23) | (man << 13);
    float v; memcpy(&v, &bits, 4); return v;
}

static uint32_t f32ToF16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint32_t sign = (b >> 16) & 0x8000u, exp = (b >> 23) & 0xFFu, man = b & 0x7FFFFFu;
    if (exp == 0xFFu) return sign | 0x7C00u | (man != 0u ? 0x0200u : 0u);
    int ne = (int)exp - 127 + 15;
    if (ne >= 31) return sign | 0x7C00u;
    if (ne <= 0) {
        if (ne < -10) return sign;
        uint32_t m = man | 0x800000u, shift = (uint32_t)(14 - ne);
        uint32_t rem = m & ((1u << shift) - 1u), h16 = m >> shift;
        uint32_t halfUlp = 1u << (shift - 1u);
        if (rem > halfUlp || (rem == halfUlp && (h16 & 1u) == 1u)) h16++;
        return sign | h16;
    }
    uint32_t h16 = ((uint32_t)ne << 10) | (man >> 13), rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && ((man >> 13) & 1u) == 1u)) h16++;
    if (h16 >= 0x7C00u) return sign | 0x7C00u;
    return sign | h16;
}

static uint32_t f32ToBf16(uint32_t b) {
    uint32_t lsb = (b >> 16) & 1u;
    return (b + 0x7FFFu + lsb) >> 16;
}

static int ftzMode = 0; /* 0 none, 1 results+inputs, 2 results only */

static float flushf(float v) {
    if (!ftzMode) return v;
    uint32_t b; memcpy(&b, &v, 4);
    if ((b & 0x7FFFFFFFu) < 0x00800000u) b &= 0x80000000u; /* zero + NaN-free */
    float r; memcpy(&r, &b, 4); return r;
}

typedef struct { size_t exact, total, zeromatch; double maxRel; } Stats;

static void acc(Stats *s, uint32_t got, uint32_t expb) {
    s->total++;
    if (got == expb) { s->exact++; return; }
    float g, e; memcpy(&g, &got, 4); memcpy(&e, &expb, 4);
    if (((got & 0x7FFFFFFFu) == 0) && ((expb & 0x7FFFFFFFu) == 0)) { s->zeromatch++; return; }
    double d = fabs((double)g - (double)e);
    double m = fmax(fabs((double)e), 1e-30);
    if (d / m > s->maxRel) s->maxRel = d / m;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s oracle.bin [strict|fma|ftz|ftzout|fmaftz]\n", argv[0]); return 2; }
    const char *mode = argc > 2 ? argv[2] : "strict";
    if (!strcmp(mode, "fma") || !strcmp(mode, "fmaftz")) ftzMode = 0;
    if (!strcmp(mode, "ftz")) ftzMode = 1;
    if (!strcmp(mode, "ftzout")) ftzMode = 2;
    int useFma = !strcmp(mode, "fma") || !strcmp(mode, "fmaftz");
    if (!strcmp(mode, "fmaftz")) ftzMode = 1;

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); size_t olen = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *o = malloc(olen);
    if (fread(o, 1, olen, f) != olen) return 1;
    fclose(f);
    Hdr h; memcpy(&h, o, sizeof h);
    if (h.magic != 0x45584C33u || h.blocks != BLOCKS) { fprintf(stderr, "bad header\n"); return 2; }
    (void)f16ToF32;

    const size_t BOFF = 4096 + 256 + 256 + 2 * EL + 4 * EL + 4 * EL + 2 * EL + 2 * EL;
    size_t base = sizeof h + 2u * 65536;
    if (base + BOFF * BLOCKS != olen) { fprintf(stderr, "size mismatch\n"); return 2; }
    const size_t O_SUH = 4096, O_SVH = O_SUH + 256, O_DEC = O_SVH + 256;
    const size_t O_TMP = O_DEC + 2 * EL, O_OUT = O_TMP + 4 * EL;
    const size_t O_W16 = O_OUT + 4 * EL, O_WBF = O_W16 + 2 * EL;
#define BLK(b, fo) (o + base + (size_t)(b) * BOFF + (fo))
#define RD16(p) ((uint32_t)(p)[0] | ((uint32_t)(p)[1] << 8))
#define RD32(p) ((uint32_t)(p)[0] | ((uint32_t)(p)[1] << 8) | ((uint32_t)(p)[2] << 16) | ((uint32_t)(p)[3] << 24))

    const float HAD_SCALE = 0.08838834764831845f;
    float *tmp = malloc(sizeof(float) * EL); /* per block scratch */

    Stats tmpS = {0}, outS = {0}, w16 = {0}, wbf = {0};
    /* denormal diagnostics */
    size_t dips = 0, tmpDenorm = 0, outDenorm = 0, suhDenorm = 0, svhDenorm = 0, decDenorm = 0;

    for (int b = 0; b < BLOCKS; b++) {
        const uint8_t *suhB = BLK(b, O_SUH), *svhB = BLK(b, O_SVH), *decB = BLK(b, O_DEC);
        /* Stage B: row H128 + suh */
        for (uint32_t e = 0; e < EL; e++) {
            uint32_t r = e >> 7, c = e & 127u;
            float accv = 0.0f;
            for (uint32_t k = 0; k < 128u; k++) {
                float he = (__builtin_popcount(r & k) & 1u) ? -HAD_SCALE : HAD_SCALE;
                float x = f16ToF32(RD16(decB + 2u * ((size_t)k * 128u + c)));
                if ((f32ToF16(x) & 0x7C00u) == 0 && (f32ToF16(x) & 0x3FFu) && 0) {}
                if (x != 0.0f && fabsf(x) < 1.17549435e-38f) decDenorm++;
                if (ftzMode == 1) x = flushf(x);
                float p = flushf(he * x);
                accv = useFma ? flushf(fmaf(he, x, accv)) : flushf(accv + p);
                if (accv != 0.0f && fabsf(accv) < 1.17549435e-38f) dips++;
            }
            float suh = f16ToF32(RD16(suhB + 2u * r));
            if (suh != 0.0f && fabsf(suh) < 1.17549435e-38f) suhDenorm++;
            if (ftzMode == 1) suh = flushf(suh);
            float t = flushf(accv * suh);
            if (t != 0.0f && fabsf(t) < 1.17549435e-38f) tmpDenorm++;
            tmp[e] = t;
            acc(&tmpS, *(uint32_t *)&t, RD32(BLK(b, O_TMP) + 4u * e));
        }
        /* Stage C: column H128 + svh, consume the SIMULATED tmp (device feeds its own stage B) */
        for (uint32_t e = 0; e < EL; e++) {
            uint32_t r = e >> 7, c = e & 127u;
            float accv = 0.0f;
            for (uint32_t k = 0; k < 128u; k++) {
                float he = (__builtin_popcount(c & k) & 1u) ? -HAD_SCALE : HAD_SCALE;
                float x = tmp[(size_t)r * 128u + k];
                if (ftzMode == 1) x = flushf(x);
                float p = flushf(he * x);
                accv = useFma ? flushf(fmaf(he, x, accv)) : flushf(accv + p);
                if (accv != 0.0f && fabsf(accv) < 1.17549435e-38f) dips++;
            }
            float svh = f16ToF32(RD16(svhB + 2u * c));
            if (svh != 0.0f && fabsf(svh) < 1.17549435e-38f) svhDenorm++;
            if (ftzMode == 1) svh = flushf(svh);
            float v = flushf(accv * svh);
            if (v != 0.0f && fabsf(v) < 1.17549435e-38f) outDenorm++;
            uint32_t vb = *(uint32_t *)&v;
            acc(&outS, vb, RD32(BLK(b, O_OUT) + 4u * e));
            uint32_t g16 = f32ToF16(v);
            uint32_t e16 = RD16(BLK(b, O_W16) + 2u * e);
            w16.total++;
            if (g16 == e16 || ((((g16 ^ e16) & 0x7C00u) == 0x7C00u) && ((g16 & 0x3FFu) && (e16 & 0x3FFu)))) w16.exact++;
            uint32_t gbf = f32ToBf16(vb);
            uint32_t ebf = RD16(BLK(b, O_WBF) + 2u * e);
            wbf.total++;
            if (gbf == ebf || ((((gbf ^ ebf) & 0x7C00u) == 0x7C00u) && ((gbf & 0x3FFu) && (ebf & 0x3FFu)))) wbf.exact++;
        }
    }

    printf("mode %s:\n", mode);
    printf("  rowH128 exact %zu/%zu (%.2f%%) + %zu signed-zero, maxRel %.3e\n",
           tmpS.exact, tmpS.total, 100.0 * tmpS.exact / tmpS.total, tmpS.zeromatch, tmpS.maxRel);
    printf("  colH128 exact %zu/%zu (%.2f%%) + %zu signed-zero, maxRel %.3e\n",
           outS.exact, outS.total, 100.0 * outS.exact / outS.total, outS.zeromatch, outS.maxRel);
    printf("  final f16  %zu/%zu, final bf16 %zu/%zu\n", w16.exact, w16.total, wbf.exact, wbf.total);
    printf("  diag: denormal-dip partial sums %zu, denormal tmp %zu, denormal out %zu, dec-subnorm-inputs %zu, suh denorm %zu, svh denorm %zu\n",
           dips, tmpDenorm, outDenorm, decDenorm, suhDenorm, svhDenorm);
    return 0;
}
