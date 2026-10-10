/* harness.c — M1 oracle runner for exl3_decode.comp on lavapipe.
 * Usage: harness <oracle.bin> <exl3_decode.spv> <exl3_mcg_test.spv>
 * Exit 0 only if every integer part is exact and every f32 part is within
 * rel 1e-6 (scope doc section 5(i)). NaN==NaN counts as a match and is
 * reported separately. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define VKC(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "VkError %d at %d: %s\n", r_, __LINE__, #x); exit(1); } } while (0)

#define BLOCKS 18
#define EL 16384
#define MCG_N 65536

typedef struct { uint32_t magic, version, n, mask, blocks; } Hdr;

static uint8_t *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc(*len);
    if (fread(p, 1, *len, f) != *len) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return p;
}

static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static float rd32f(const uint8_t *p) { float v; memcpy(&v, p, 4); return v; }

static uint32_t findMem(VkPhysicalDevice pd, VkMemoryRequirements req,
                        VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((req.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    fprintf(stderr, "no memory type\n");
    exit(1);
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *ptr; VkDeviceSize size; } Buf;

static void makeBuf(VkDevice dev, VkPhysicalDevice pd, Buf *b, VkDeviceSize size) {
    b->size = size;
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VKC(vkCreateBuffer(dev, &bi, NULL, &b->buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b->buf, &req);
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = findMem(pd, req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
    VKC(vkAllocateMemory(dev, &ai, NULL, &b->mem));
    VKC(vkBindBufferMemory(dev, b->buf, b->mem, 0));
    VKC(vkMapMemory(dev, b->mem, 0, size, 0, &b->ptr));
}

static uint32_t f16NaN(uint32_t v) { return (v & 0x7C00u) == 0x7C00u && (v & 0x3FFu) != 0u; }

typedef struct {
    size_t exact, nanMatch, total;
    double maxRel, maxAbs;
} Stats;

static void accF32(Stats *s, uint32_t got, float expf) {
    s->total++;
    uint32_t expb;
    float gotf;
    memcpy(&expb, &expf, 4);
    memcpy(&gotf, &got, 4);
    if (got == expb) { s->exact++; return; }
    if (isnan(gotf) && isnan(expf)) { s->nanMatch++; s->exact++; return; }
    double d = fabs((double)gotf - (double)expf);
    double m = fmax(fabs((double)expf), 1e-30);
    if (d / m > s->maxRel) s->maxRel = d / m;
    if (d > s->maxAbs) s->maxAbs = d;
}

/* --- M2: MoE layer stage (exl3_gemv.comp + exl3_moe.comp) -----------------
 * Per case: upload, run prep-gate -> prep-up -> gemv-gate -> gemv-up ->
 * swiglu -> prep-down -> gemv-down -> combine with barriers, then compare:
 *  - ag/au (gate/up projections) must be BIT-EXACT vs the R1 reference
 *    (NaN-aware); they contain no transcendentals.
 *  - h16/xtd/slotout/out16/outbf vs R1: every pair within 1 f16/bf16 ulp;
 *    flips (exactly 1 ulp, from exp() ULP differences between glibc and
 *    llvmpipe) must stay under 1%.
 *  - out vs R2 (plainMoeFused mirror, flags bit 0): report maxAbs/maxRel/maxUlp
 *    table; finite pairs beyond 2 bf16 ulp must stay under 0.1%. Non-finite
 *    pairs are counted and excluded (legitimate overflow divergence between
 *    dense and fused rounding). */

#define MOE_BINDINGS 27

typedef struct {
    char name[16];
    uint32_t rows, hidden, inter, experts, topk, n, mask, flags;
    float limit;
    size_t off_x, off_slots, off_scores;
    size_t off_tg, off_tu, off_td;
    size_t off_suhg, off_suhu, off_suhd, off_svhg, off_svhu, off_svhd;
    size_t off_ag, off_au, off_h16, off_xtd, off_slotout;
    size_t off_r1o32, off_r1o16, off_r1obf;
    size_t off_r2o32, off_r2o16, off_r2obf;
    size_t blob;
} MoeCase;

typedef struct {
    size_t exact, nanMatch, total;
    size_t flips, bad, nonfin;
    double maxAbs, maxRel, maxUlp;
} MStats;

static void mAcc(MStats *s, uint32_t got, uint32_t exp, int gateUlp) {
    s->total++;
    float gf, ef;
    memcpy(&gf, &got, 4);
    memcpy(&ef, &exp, 4);
    if (got == exp) { s->exact++; return; }
    if (isnan(gf) && isnan(ef)) { s->nanMatch++; return; }
    if (!isfinite(gf) || !isfinite(ef)) { s->nonfin++; return; }
    double d = fabs((double)gf - (double)ef);
    if (d == 0.0) { s->exact++; return; } /* covers +0/-0 */
    double m = fmax(fabs((double)gf), fabs((double)ef));
    if (d > s->maxAbs) s->maxAbs = d;
    if (d / fmax(m, 1e-30) > s->maxRel) s->maxRel = d / fmax(m, 1e-30);
    /* ulp of the wider magnitude, bf16 has an 8-bit significand, f16 11 */
    int ib = ilogbf((float)m);
    double ulp = ldexp(1.0, ib - gateUlp);
    double u = d / ulp;
    if (u > s->maxUlp) s->maxUlp = u;
    if (u <= 1.0) s->flips++;
    else s->bad++;
}

static void mAccExact(MStats *s, uint32_t got, uint32_t exp) {
    s->total++;
    float gf, ef;
    memcpy(&gf, &got, 4);
    memcpy(&ef, &exp, 4);
    if (got == exp) { s->exact++; return; }
    if (isnan(gf) && isnan(ef)) { s->nanMatch++; s->exact++; }
}

static void up16u32(uint32_t *dst, const uint8_t *src, size_t n16) {
    for (size_t i = 0; i < n16; i++) dst[i] = rd16(src + 2 * i);
}

static void up16packed(uint32_t *dst, const uint8_t *src, size_t n16) {
    for (size_t i = 0; i < n16 / 2; i++)
        dst[i] = (uint32_t)rd16(src + 4 * i) | ((uint32_t)rd16(src + 4 * i + 2) << 16);
}

static int runMoeStage(VkDevice dev, VkPhysicalDevice pd, VkQueue queue, uint32_t qf,
                       const uint8_t *mb, size_t mlen,
                       const uint8_t *moeSpv, size_t moeLen) {
    uint32_t magic, version, caseCount;
    memcpy(&magic, mb, 4); memcpy(&version, mb + 4, 4); memcpy(&caseCount, mb + 8, 4);
    if (magic != 0x4D334C58u || version != 1) { fprintf(stderr, "bad moe oracle header\n"); return 0; }

    VkDescriptorSetLayoutBinding rb[MOE_BINDINGS];
    for (int i = 0; i < MOE_BINDINGS; i++)
        rb[i] = (VkDescriptorSetLayoutBinding){ .binding = (uint32_t)i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = MOE_BINDINGS, .pBindings = rb };
    VkDescriptorSetLayout layout;
    VKC(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &layout));
    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 40 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout pl;
    VKC(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkShaderModule moeMod;
    smci.codeSize = moeLen; smci.pCode = (const uint32_t *)moeSpv;
    VKC(vkCreateShaderModule(dev, &smci, NULL, &moeMod));
    VkComputePipelineCreateInfo cpi[1] = {{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = moeMod, .pName = "main" },
        .layout = pl }};
    VkPipeline pipes[1];
    VKC(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, cpi, NULL, pipes));

    int maxCases = INT32_MAX;
    const char *mc = getenv("EXL3_VK_MOE_MAX_CASES");
    if (mc) maxCases = atoi(mc);
    int minCase = 0;
    const char *mn = getenv("EXL3_VK_MOE_MIN_CASE");
    if (mn) minCase = atoi(mn);

    VkDescriptorPoolSize ps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = MOE_BINDINGS * caseCount };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = caseCount, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkDescriptorPool pool;
    VKC(vkCreateDescriptorPool(dev, &pci, NULL, &pool));

    VkCommandPoolCreateInfo cpoi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = qf };
    VkCommandPool cpool;
    VKC(vkCreateCommandPool(dev, &cpoi, NULL, &cpool));
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cmd;
    VKC(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VKC(vkCreateFence(dev, &fci, NULL, &fence));

    int allOk = 1;
    size_t pos = 12;
    for (uint32_t cs = 0; cs < caseCount; cs++) {
        if (cs >= (uint32_t)maxCases) break;
        MoeCase C = {0};
        memcpy(C.name, mb + pos, 16); pos += 16;
        memcpy(&C.rows, mb + pos, 4); pos += 4;
        memcpy(&C.hidden, mb + pos, 4); pos += 4;
        memcpy(&C.inter, mb + pos, 4); pos += 4;
        memcpy(&C.experts, mb + pos, 4); pos += 4;
        memcpy(&C.topk, mb + pos, 4); pos += 4;
        memcpy(&C.n, mb + pos, 4); pos += 4;
        memcpy(&C.mask, mb + pos, 4); pos += 4;
        memcpy(&C.flags, mb + pos, 4); pos += 4;
        memcpy(&C.limit, mb + pos, 4); pos += 4;
        size_t rows = C.rows, hidden = C.hidden, inter = C.inter, E = C.experts, topk = C.topk;
        size_t rts = rows * topk;
        size_t ts = (hidden / 16) * (inter / 16) * C.n;
        size_t s_x = rows * hidden * 4, s_slots = rts * 4, s_scores = rts * 4;
        size_t s_t = E * ts * 2;
        size_t s_suh = E * hidden * 2, s_suhd = E * inter * 2;
        size_t s_svh = E * inter * 2, s_svhd = E * hidden * 2;
        size_t s_mid = rts * inter * 2, s_slot = rts * hidden * 2;
        size_t s_o32 = rows * hidden * 4, s_o16 = rows * hidden * 2;
        C.off_x = pos; C.off_slots = C.off_x + s_x; C.off_scores = C.off_slots + s_slots;
        C.off_tg = C.off_scores + s_scores; C.off_tu = C.off_tg + s_t; C.off_td = C.off_tu + s_t;
        C.off_suhg = C.off_td + s_t; C.off_suhu = C.off_suhg + s_suh; C.off_suhd = C.off_suhu + s_suh;
        C.off_svhg = C.off_suhd + s_suhd; C.off_svhu = C.off_svhg + s_svh; C.off_svhd = C.off_svhu + s_svh;
        C.off_ag = C.off_svhd + s_svhd; C.off_au = C.off_ag + s_mid; C.off_h16 = C.off_au + s_mid;
        C.off_xtd = C.off_h16 + s_mid; C.off_slotout = C.off_xtd + s_mid;
        C.off_r1o32 = C.off_slotout + s_slot; C.off_r1o16 = C.off_r1o32 + s_o32;
        C.off_r1obf = C.off_r1o16 + s_o16; C.off_r2o32 = C.off_r1obf + s_o16;
        C.off_r2o16 = C.off_r2o32 + s_o32; C.off_r2obf = C.off_r2o16 + s_o16;
        C.blob = C.off_r2obf + s_o16;
        if (C.blob > mlen) { fprintf(stderr, "moe oracle truncated at %s\n", C.name); return 0; }
        pos = C.blob; /* authoritative: offsets above carry the rest */
        if ((int)cs < minCase || cs >= (uint32_t)maxCases) continue;

        /* u16 arrays upload one u32 per element -> every such buffer is 2x
         * its file size; trellis is uploaded as packed u32 words instead. */
        Buf bufs[MOE_BINDINGS];
        makeBuf(dev, pd, &bufs[0], s_x);                        /* x f32 */
        makeBuf(dev, pd, &bufs[1], s_suh * 2);                  /* suhg u32/u16 */
        makeBuf(dev, pd, &bufs[2], s_suh * 2);
        makeBuf(dev, pd, &bufs[3], s_suhd * 2);
        makeBuf(dev, pd, &bufs[4], s_slot * 2);                 /* xtG */
        makeBuf(dev, pd, &bufs[5], s_slot * 2);                 /* xtU */
        makeBuf(dev, pd, &bufs[6], s_t);                        /* tg packed words */
        makeBuf(dev, pd, &bufs[7], s_t);
        makeBuf(dev, pd, &bufs[8], s_t);
        makeBuf(dev, pd, &bufs[9], s_svh * 2);                  /* svhg */
        makeBuf(dev, pd, &bufs[10], s_svh * 2);
        makeBuf(dev, pd, &bufs[11], s_svhd * 2);
        makeBuf(dev, pd, &bufs[12], s_mid * 2);                 /* ag */
        makeBuf(dev, pd, &bufs[13], s_mid * 2);                 /* au */
        makeBuf(dev, pd, &bufs[14], rts * inter * 4);           /* h32 f32 */
        makeBuf(dev, pd, &bufs[15], s_mid * 2);                 /* h16 */
        makeBuf(dev, pd, &bufs[16], s_slot * 2);                /* xtd */
        makeBuf(dev, pd, &bufs[17], s_slot * 2);                /* slotout */
        makeBuf(dev, pd, &bufs[18], s_slots);                   /* slots */
        makeBuf(dev, pd, &bufs[19], s_scores);                  /* scores f32 */
        makeBuf(dev, pd, &bufs[20], s_o32);                     /* out32 */
        makeBuf(dev, pd, &bufs[21], s_o16 * 2);                 /* out16 */
        makeBuf(dev, pd, &bufs[22], s_o16 * 2);                 /* outbf */
        makeBuf(dev, pd, &bufs[23], rts * inter * 4);           /* accG f32 */
        makeBuf(dev, pd, &bufs[24], rts * inter * 4);           /* accU f32 */
        makeBuf(dev, pd, &bufs[25], rts * hidden * 4);          /* accD f32 */
        makeBuf(dev, pd, &bufs[26], (size_t)C.hidden / 16 * inter * 512 * 2); /* Wstr u32/u16 */

        up16u32((uint32_t *)bufs[1].ptr, mb + C.off_suhg, E * hidden);
        up16u32((uint32_t *)bufs[2].ptr, mb + C.off_suhu, E * hidden);
        up16u32((uint32_t *)bufs[3].ptr, mb + C.off_suhd, E * inter);
        up16packed((uint32_t *)bufs[6].ptr, mb + C.off_tg, E * ts);
        up16packed((uint32_t *)bufs[7].ptr, mb + C.off_tu, E * ts);
        up16packed((uint32_t *)bufs[8].ptr, mb + C.off_td, E * ts);
        up16u32((uint32_t *)bufs[9].ptr, mb + C.off_svhg, E * inter);
        up16u32((uint32_t *)bufs[10].ptr, mb + C.off_svhu, E * inter);
        up16u32((uint32_t *)bufs[11].ptr, mb + C.off_svhd, E * hidden);
        memcpy(bufs[0].ptr, mb + C.off_x, s_x);
        memcpy(bufs[18].ptr, mb + C.off_slots, s_slots);
        memcpy(bufs[19].ptr, mb + C.off_scores, s_scores);

        VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout };
        VkDescriptorSet set;
        VKC(vkAllocateDescriptorSets(dev, &dai, &set));
        for (int i = 0; i < MOE_BINDINGS; i++) {
            VkWriteDescriptorSet wr = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = set, .dstBinding = (uint32_t)i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &(VkDescriptorBufferInfo){ .buffer = bufs[i].buf, .offset = 0, .range = VK_WHOLE_SIZE } };
            vkUpdateDescriptorSets(dev, 1, &wr, 0, NULL);
        }

        float limitBits; memcpy(&limitBits, &C.limit, 4);
        uint32_t pc8[10] = { 0, C.hidden, C.inter, C.topk, C.n, C.mask, 0, 0, 0, 16 };
        memcpy(&pc8[6], &limitBits, 4);
        /* One submit per dispatch: this llvmpipe build does not honor
         * vkCmdPipelineBarrier between compute dispatches (kernel B reads
         * zeros from a buffer kernel A just wrote), so submission boundaries
         * are the visibility mechanism. Per-slot GEMV pairs: for every
         * (projection, gx): A decodes the slot tile row into Wstr, B writes
         * the f32 accumulator; finish applies f16 round + H128 + svh. */
        typedef struct { uint32_t mode, gx, grid; } MStep;
        uint32_t inTiles = (uint32_t)(hidden / 16), tTiles = (uint32_t)(inter / 16);
        size_t cap = 16 + 40 * (size_t)rts; /* slabbed B */
        MStep *steps = malloc(cap * sizeof *steps);
        size_t n = 0;
        steps[n++] = (MStep){ 0, 0, (uint32_t)(rts * (hidden / 128)) };
        steps[n++] = (MStep){ 1, 0, (uint32_t)(rts * (hidden / 128)) };
        /* finish-gate/up run before SwiGLU (it consumes the finished f16
         * gate/up); finish-down runs before the combine. */
        /* stage B in tile slabs: this llvmpipe build stops the in-kernel tk
         * loop after ~6-8 iterations (acc[393+] diverges, acc[512+] never
         * written); slabbing keeps the exact accumulation order (k ascending)
         * via read-modify-write across dispatches. */
        /* EXL3_SLAB = stage-B in-tiles per dispatch (default 16, the lavapipe workaround). 0 = one dispatch per row,
         * the exact reference accumulation order (use on a driver that runs the whole tk loop). */
        const char *slabEnv = getenv("EXL3_SLAB");
        uint32_t slab = slabEnv ? (uint32_t)strtoul(slabEnv, NULL, 10) : 16u;
        uint32_t gTiles = (uint32_t)(hidden / 16), dTiles = (uint32_t)(inter / 16);
        if (slab == 0) slab = gTiles > dTiles ? gTiles : dTiles;
        pc8[9] = slab;
        uint32_t gSlabs = (gTiles + slab - 1) / slab, dSlabs = (dTiles + slab - 1) / slab;
        for (size_t g = 0; g < rts; g++) {
            steps[n++] = (MStep){ 2, (uint32_t)g, inTiles };
            for (uint32_t sl2 = 0; sl2 < gSlabs; sl2++)
                steps[n++] = (MStep){ 3, (uint32_t)g | ((sl2 * slab) << 16), 1 };
        }
        steps[n++] = (MStep){ 10, 0, (uint32_t)(rts * (inter / 128)) };
        for (size_t g = 0; g < rts; g++) {
            steps[n++] = (MStep){ 4, (uint32_t)g, inTiles };
            for (uint32_t sl2 = 0; sl2 < gSlabs; sl2++)
                steps[n++] = (MStep){ 5, (uint32_t)g | ((sl2 * slab) << 16), 1 };
        }
        steps[n++] = (MStep){ 11, 0, (uint32_t)(rts * (inter / 128)) };
        steps[n++] = (MStep){ 8, 0, (uint32_t)rts };
        for (size_t g = 0; g < rts; g++) {
            steps[n++] = (MStep){ 9, (uint32_t)g, (uint32_t)(inter / 128) }; /* gx via push */
            steps[n++] = (MStep){ 6, (uint32_t)g, tTiles };
            for (uint32_t sl2 = 0; sl2 < dSlabs; sl2++)
                steps[n++] = (MStep){ 7, (uint32_t)g | ((sl2 * slab) << 16), 1 };
        }
        steps[n++] = (MStep){ 12, 0, (uint32_t)(rts * (hidden / 128)) };
        steps[n++] = (MStep){ 13, 0, (uint32_t)(rows * ((hidden + 63) / 64)) };

        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        for (size_t st = 0; st < n; st++) {
            VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
            VKC(vkBeginCommandBuffer(cmd, &bbi));
            pc8[0] = steps[st].mode;
            pc8[7] = steps[st].gx & 0xFFFFu;
            pc8[8] = steps[st].gx >> 16;  /* stage-B tile slab */
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[0]);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
            vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 40, pc8);
            vkCmdDispatch(cmd, steps[st].grid, 1, 1);
            VKC(vkEndCommandBuffer(cmd));
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1, .pCommandBuffers = &cmd };
            VKC(vkQueueSubmit(queue, 1, &si, fence));
            { /* EXL3_FENCE_S caps one dispatch (default 1200 s). A timeout is fatal: VKC exits on VK_TIMEOUT. */
              const char *fs = getenv("EXL3_FENCE_S");
              uint64_t sec = fs ? strtoull(fs, NULL, 10) : 1200ull;
              VKC(vkWaitForFences(dev, 1, &fence, VK_TRUE, sec * 1000ull * 1000ull * 1000ull)); }
            VKC(vkResetFences(dev, 1, &fence));
            vkResetCommandPool(dev, cpool, 0);
        }
        free(steps);
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double ms = (ts1.tv_sec - ts0.tv_sec) * 1e3 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;

        /* --- compare --- */
        MStats ag = {0}, au = {0}, h16 = {0}, xtd = {0}, so = {0}, o16 = {0}, obf = {0},
               r2o = {0};
        uint32_t *agG = (uint32_t *)bufs[12].ptr, *auG = (uint32_t *)bufs[13].ptr;
        uint32_t *hG = (uint32_t *)bufs[15].ptr, *xtdG = (uint32_t *)bufs[16].ptr;
        uint32_t *soG = (uint32_t *)bufs[17].ptr;
        uint32_t *o16G = (uint32_t *)bufs[21].ptr, *obfG = (uint32_t *)bufs[22].ptr;
        size_t nMid = rts * inter, nOut = rows * hidden;
        for (size_t i = 0; i < nMid; i++) {
            mAcc(&ag, agG[i], rd16(mb + C.off_ag + 2 * i), 10);
            mAcc(&au, auG[i], rd16(mb + C.off_au + 2 * i), 10);
            mAcc(&h16, hG[i], rd16(mb + C.off_h16 + 2 * i), 10);
            mAcc(&xtd, xtdG[i], rd16(mb + C.off_xtd + 2 * i), 10);
        }
        for (size_t i = 0; i < nOut; i++) {
            mAcc(&so, soG[i], rd16(mb + C.off_slotout + 2 * i), 10);
            mAcc(&o16, o16G[i], rd16(mb + C.off_r1o16 + 2 * i), 10);
            mAcc(&obf, obfG[i], rd16(mb + C.off_r1obf + 2 * i), 7);
        }
        int r2ok = 1;
        if (C.flags & 1) {
            for (size_t i = 0; i < nOut; i++) {
                float ev; memcpy(&ev, mb + C.off_r2o32 + 4 * i, 4);
                uint32_t eb; memcpy(&eb, &ev, 4);
                mAcc(&r2o, ((uint32_t)obfG[i]) << 16, eb, 7); /* bf16 bits -> f32 */
            }
            r2ok = r2o.bad <= (size_t)((r2o.total - r2o.nonfin) / 1000 + 1); /* <= ~0.1% */
        }
        int caseOk = ag.bad == 0 && au.bad == 0 &&
                     ag.flips * 20 <= ag.total && au.flips * 20 <= au.total &&
                     h16.bad == 0 && xtd.bad == 0 && so.bad == 0 &&
                     h16.flips * 100 <= nMid && xtd.flips * 100 <= nMid && so.flips * 100 <= nOut &&
                     o16.bad == 0 && obf.bad == 0 &&
                     o16.flips * 100 <= nOut && obf.flips * 100 <= nOut && r2ok;
        allOk = allOk && caseOk;
        if (getenv("EXL3_VK_MOE_DEBUG")) {
            uint32_t *xtgG = (uint32_t *)bufs[4].ptr;
            float *o32G2 = (float *)bufs[20].ptr;
            printf("  debug xtg0..3   : %04X %04X %04X %04X\n",
                   xtgG[0], xtgG[1], xtgG[2], xtgG[3]);
            printf("  debug xtg 64    : %04X  xtg 63: %04X xtg 65: %04X\n",
                   xtgG[64], xtgG[63], xtgG[65]);
            printf("  debug h16 0..5  : %04X %04X %04X %04X %04X %04X\n",
                   hG[0], hG[1], hG[2], hG[3], hG[4], hG[5]);
            printf("  debug h16 want  : %04X %04X %04X %04X %04X %04X\n",
                   rd16(mb + C.off_h16), rd16(mb + C.off_h16 + 2),
                   rd16(mb + C.off_h16 + 4), rd16(mb + C.off_h16 + 6),
                   rd16(mb + C.off_h16 + 8), rd16(mb + C.off_h16 + 10));
            printf("  debug ag0..2    : %04X %04X %04X  au0..2: %04X %04X %04X\n",
                   agG[0], agG[1], agG[2], auG[0], auG[1], auG[2]);
            {
                float *accp = (float *)bufs[23].ptr;
                printf("  dev acc[380..392]:");
                for (int o = 380; o <= 392; o++) printf(" %.6g", (double)accp[o]);
                printf("\n");
                printf("  r1 out[0..2]  : %.6g %.6g %.6g\n",
                   (double)rd32f(mb + C.off_r1o32), (double)rd32f(mb + C.off_r1o32 + 4),
                   (double)rd32f(mb + C.off_r1o32 + 8));
            printf("  r2 out[0..2]  : %.6g %.6g %.6g\n",
                   (double)rd32f(mb + C.off_r2o32), (double)rd32f(mb + C.off_r2o32 + 4),
                   (double)rd32f(mb + C.off_r2o32 + 8));
            printf("  dev acc[508..516]:");
                for (int o = 508; o <= 516; o++) printf(" %.6g", (double)accp[o]);
                printf("\n");
            }
            uint32_t *auG2 = (uint32_t *)bufs[13].ptr;
            printf("  debug ag got    : %04X %04X %04X %04X\n", agG[0], agG[1], agG[2], agG[3]);
            printf("  debug au got    : %04X %04X %04X %04X\n", auG2[0], auG2[1], auG2[2], auG2[3]);
            printf("  debug ag want   : %04X %04X %04X %04X\n",
                   rd16(mb + C.off_ag), rd16(mb + C.off_ag + 2),
                   rd16(mb + C.off_ag + 4), rd16(mb + C.off_ag + 6));
            printf("  debug so got    : %04X %04X %04X %04X\n", soG[0], soG[1], soG[2], soG[3]);
            printf("  debug so want   : %04X %04X %04X %04X\n",
                   rd16(mb + C.off_slotout), rd16(mb + C.off_slotout + 2),
                   rd16(mb + C.off_slotout + 4), rd16(mb + C.off_slotout + 6));
            printf("  debug o32 got   : %.6g %.6g %.6g\n", (double)o32G2[0], (double)o32G2[1], (double)o32G2[2]);
            printf("  debug o32 want  : %.6g %.6g %.6g\n",
                   (double)rd32f(mb + C.off_r1o32), (double)rd32f(mb + C.off_r1o32 + 4),
                   (double)rd32f(mb + C.off_r1o32 + 8));
            printf("  debug slots     : %u %u %u %u\n",
               ((uint32_t *)bufs[18].ptr)[0], ((uint32_t *)bufs[18].ptr)[1],
               ((uint32_t *)bufs[18].ptr)[2], ((uint32_t *)bufs[18].ptr)[3]);
        }
        if (getenv("EXL3_VK_MOE_DEBUG")) {
            for (size_t sl = 0; sl < rts; sl++) {
                size_t ex = 0, first = inter, second = inter;
                for (size_t o = 0; o < inter; o++) {
                    if (agG[sl * inter + o] == rd16(mb + C.off_ag + 2 * (sl * inter + o))) ex++;
                    else { if (first == inter) first = o; else if (second == inter) second = o; }
                }
                printf("  slot %zu (e=%u): ag exact %zu/%zu firstBad %zu second %zu\n",
                       sl, ((uint32_t *)bufs[18].ptr)[sl], ex, inter, first, second);
                if (sl == 0) {
                    printf("    per-tile exact:");
                    for (size_t tn = 0; tn < inter / 16; tn++) {
                        size_t te = 0;
                        for (size_t c = 0; c < 16; c++)
                            if (agG[tn * 16 + c] == rd16(mb + C.off_ag + 2 * (tn * 16 + c))) te++;
                        printf(" %zu:%zu", (size_t)tn, te);
                    }
                    printf("\n");
                }
            }
        }
        printf("moe %-13s rows=%zu hid=%zu inter=%zu E=%zu k=%zu lim=%.1f | %.1f ms | "
               "ag %zu/%zu au %zu/%zu | h16 f%zu b%zu nf%zu xtd f%zu b%zu so f%zu b%zu | "
               "o16 f%zu b%zu nf%zu obf f%zu b%zu nf%zu maxUlp %.2f | "
               "R2 maxAbs %.3e maxRel %.3e maxUlp %.2f bad %zu nf %zu | %s\n",
               C.name, rows, hidden, inter, E, topk, (double)C.limit, ms,
               ag.exact, ag.total, au.exact, au.total,
               h16.flips, h16.bad, h16.nonfin, xtd.flips, xtd.bad, so.flips, so.bad,
               o16.flips, o16.bad, o16.nonfin, obf.flips, obf.bad, obf.nonfin, obf.maxUlp,
               r2o.maxAbs, r2o.maxRel, r2o.maxUlp, r2o.bad, r2o.nonfin,
               caseOk ? "PASS" : "FAIL");
        fflush(stdout);

        for (int i = 0; i < MOE_BINDINGS; i++) {
            vkFreeMemory(dev, bufs[i].mem, NULL);
            vkDestroyBuffer(dev, bufs[i].buf, NULL);
        }
        vkResetCommandPool(dev, cpool, 0);
    }

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cpool, NULL);
    vkDestroyDescriptorPool(dev, pool, NULL);
    vkDestroyPipeline(dev, pipes[0], NULL);
    vkDestroyShaderModule(dev, moeMod, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyDescriptorSetLayout(dev, layout, NULL);
    return allOk;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 6) {
        fprintf(stderr, "usage: %s oracle.bin decode.spv mcg.spv [moe.bin moe.spv]\n", argv[0]);
        return 2;
    }
    setbuf(stdout, NULL);
    size_t olen;
    uint8_t *o = slurp(argv[1], &olen);
    Hdr h;
    memcpy(&h, o, sizeof h);
    if (h.magic != 0x45584C33u || h.version != 1 || h.blocks != BLOCKS || h.n != 32) {
        fprintf(stderr, "bad oracle header\n"); return 2;
    }
    const uint8_t *mcgExp = o + sizeof h;
    size_t per = (size_t)BLOCKS * EL;
    // Generator writes per block, interleaved: trellis(4096B) suh(256) svh(256)
    // dec16(32768) tmp(65536) out(65536) w16(32768) wbf(32768) = 233984 B.
    const size_t BOFF = 4096 + 256 + 256 + 2 * EL + 4 * EL + 4 * EL + 2 * EL + 2 * EL;
    size_t base = sizeof h + 2u * MCG_N;
    if (base + BOFF * BLOCKS != olen) {
        fprintf(stderr, "oracle size mismatch %zu != %zu\n", base + BOFF * BLOCKS, olen);
        return 2;
    }
#define BLK(b, fo) (o + base + (size_t)(b) * BOFF + (fo))
    const size_t O_SUH = 4096, O_SVH = O_SUH + 256, O_DEC = O_SVH + 256;
    const size_t O_TMP = O_DEC + 2 * EL, O_OUT = O_TMP + 4 * EL;
    const size_t O_W16 = O_OUT + 4 * EL, O_WBF = O_W16 + 2 * EL;

    // --- Vulkan setup ---
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app };
    VkInstance inst;
    VKC(vkCreateInstance(&ici, NULL, &inst));
    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice *pds = malloc(nd * sizeof *pds);
    vkEnumeratePhysicalDevices(inst, &nd, pds);
    VkPhysicalDevice pd = pds[0];
    VkPhysicalDeviceProperties pp;
    /* EXL3_VK_DEVICE=<substring> picks another device (for example the Honeykrisp ICD on an Apple Silicon Mac).
     * Default: the software device (lavapipe/llvmpipe), so the oracle runs without a GPU. */
    const char *want = getenv("EXL3_VK_DEVICE");
    for (uint32_t i = 0; i < nd; i++) {
        vkGetPhysicalDeviceProperties(pds[i], &pp);
        int hit = want ? strstr(pp.deviceName, want) != NULL
                       : (strstr(pp.deviceName, "llvmpipe") || strstr(pp.deviceName, "lavapipe"));
        if (hit) { pd = pds[i]; break; }
    }
    vkGetPhysicalDeviceProperties(pd, &pp);
    printf("device            : %s (vulkan %u.%u)\n", pp.deviceName,
           VK_VERSION_MAJOR(pp.apiVersion), VK_VERSION_MINOR(pp.apiVersion));

    uint32_t qf = UINT32_MAX;
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qp = malloc(nq * sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    for (uint32_t i = 0; i < nq; i++)
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    if (qf == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 1; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkPhysicalDeviceFeatures feat = { .shaderInt64 = VK_TRUE };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
        .pEnabledFeatures = &feat };
    VkDevice dev;
    VKC(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, qf, 0, &queue);

    // Buffers: [0]=trellis u32 [1]=suh [2]=svh [3]=dec16 [4]=tmp [5]=out32
    // [6]=w16 [7]=wbf [8]=mcg out
    Buf bufs[9];
    makeBuf(dev, pd, &bufs[0], (VkDeviceSize)BLOCKS * 64 * 16 * 4);
    makeBuf(dev, pd, &bufs[1], (VkDeviceSize)BLOCKS * 128 * 4);
    makeBuf(dev, pd, &bufs[2], (VkDeviceSize)BLOCKS * 128 * 4);
    for (int i = 3; i < 9; i++)
        makeBuf(dev, pd, &bufs[i], i < 8 ? (VkDeviceSize)per * 4 : (VkDeviceSize)MCG_N * 4);

    // Upload: pack u16 pairs into u32 words (wordU32 order).
    uint32_t *tw = (uint32_t *)bufs[0].ptr;
    uint32_t *suhw = (uint32_t *)bufs[1].ptr;
    uint32_t *svhw = (uint32_t *)bufs[2].ptr;
    for (int b = 0; b < BLOCKS; b++) {
        const uint8_t *tb = BLK(b, 0);
        for (size_t i = 0; i < 1024; i++)
            tw[(size_t)b * 1024 + i] = (uint32_t)rd16(tb + 4 * i) | ((uint32_t)rd16(tb + 4 * i + 2) << 16);
        const uint8_t *sb = BLK(b, O_SUH), *vb = BLK(b, O_SVH);
        for (size_t i = 0; i < 128; i++) {
            suhw[(size_t)b * 128 + i] = rd16(sb + 2 * i);
            svhw[(size_t)b * 128 + i] = rd16(vb + 2 * i);
        }
    }

    // Descriptor layouts + pool + sets
    VkDescriptorSetLayoutBinding rb[8];
    for (int i = 0; i < 8; i++)
        rb[i] = (VkDescriptorSetLayoutBinding){ .binding = (uint32_t)i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 8, .pBindings = rb };
    VkDescriptorSetLayout mainLayout;
    VKC(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &mainLayout));
    VkDescriptorSetLayoutBinding mrb = { .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    dlci.bindingCount = 1;
    dlci.pBindings = &mrb;
    VkDescriptorSetLayout mcgLayout;
    VKC(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &mcgLayout));

    VkDescriptorPoolSize ps[2] = {
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 10 } };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = ps };
    VkDescriptorPool pool;
    VKC(vkCreateDescriptorPool(dev, &pci, NULL, &pool));
    VkDescriptorSetLayout lays[2] = { mainLayout, mcgLayout };
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool, .descriptorSetCount = 2, .pSetLayouts = lays };
    VkDescriptorSet sets[2];
    VKC(vkAllocateDescriptorSets(dev, &dai, sets));
    for (int i = 0; i < 8; i++) {
        VkWriteDescriptorSet wr = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sets[0], .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &(VkDescriptorBufferInfo){ .buffer = bufs[i].buf, .offset = 0, .range = VK_WHOLE_SIZE } };
        vkUpdateDescriptorSets(dev, 1, &wr, 0, NULL);
    }
    VkWriteDescriptorSet wr = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = sets[1], .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &(VkDescriptorBufferInfo){ .buffer = bufs[8].buf, .offset = 0, .range = VK_WHOLE_SIZE } };
    vkUpdateDescriptorSets(dev, 1, &wr, 0, NULL);

    // Pipelines
    size_t sl;
    uint8_t *mainSpv = slurp(argv[2], &sl);
    size_t sl2;
    uint8_t *mcgSpv = slurp(argv[3], &sl2);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkShaderModule mainMod, mcgMod;
    smci.codeSize = sl; smci.pCode = (const uint32_t *)mainSpv;
    VKC(vkCreateShaderModule(dev, &smci, NULL, &mainMod));
    smci.codeSize = sl2; smci.pCode = (const uint32_t *)mcgSpv;
    VKC(vkCreateShaderModule(dev, &smci, NULL, &mcgMod));

    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0, .size = 8 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &mainLayout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout mainPl;
    VKC(vkCreatePipelineLayout(dev, &plci, NULL, &mainPl));
    plci.pPushConstantRanges = NULL;
    plci.pushConstantRangeCount = 0;
    plci.pSetLayouts = &mcgLayout;
    VkPipelineLayout mcgPl;
    VKC(vkCreatePipelineLayout(dev, &plci, NULL, &mcgPl));

    VkComputePipelineCreateInfo cpi[2] = {{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mainMod, .pName = "main" },
        .layout = mainPl }, {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mcgMod, .pName = "main" },
        .layout = mcgPl }};
    VkPipeline pipes[2];
    VKC(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 2, cpi, NULL, pipes));

    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = VK_NULL_HANDLE, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1 };
    VkCommandPool cpool;
    VkCommandPoolCreateInfo cpoi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = qf };
    VKC(vkCreateCommandPool(dev, &cpoi, NULL, &cpool));
    cai.commandPool = cpool;
    VkCommandBuffer cmd;
    VKC(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VKC(vkBeginCommandBuffer(cmd, &bbi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[1]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mcgPl, 0, 1, &sets[1], 0, NULL);
    vkCmdDispatch(cmd, MCG_N / 256, 1, 1);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mainPl, 0, 1, &sets[0], 0, NULL);
    uint32_t push[2] = { h.n, h.mask };
    vkCmdPushConstants(cmd, mainPl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
    vkCmdDispatch(cmd, BLOCKS, 1, 1);
    VKC(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    VKC(vkCreateFence(dev, &fci, NULL, &fence));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmd };
    VKC(vkQueueSubmit(queue, 1, &si, fence));
    VKC(vkWaitForFences(dev, 1, &fence, VK_TRUE, 60ull * 1000ull * 1000ull * 1000ull));

    // --- Compare ---
    Stats mcg = {0}, dec = {0}, tmp = {0}, out = {0}, w16 = {0}, wbf = {0};
    uint32_t *mcgGot = (uint32_t *)bufs[8].ptr;
    for (size_t i = 0; i < MCG_N; i++) {
        mcg.total++;
        if (mcgGot[i] == rd16(mcgExp + 2 * i)) mcg.exact++;
    }
    uint32_t *decGot = (uint32_t *)bufs[3].ptr;
    for (int b = 0; b < BLOCKS; b++) {
        const uint8_t *db = BLK(b, O_DEC);
        for (size_t i = 0; i < EL; i++) {
            size_t g = (size_t)b * EL + i;
            dec.total++;
            if (decGot[g] == rd16(db + 2 * i)) dec.exact++;
            else if (dec.exact + (dec.total - dec.exact - 1) < 12 && i < 8)
                printf("  dec mismatch blk%d el%zu: got 0x%04X exp 0x%04X\n",
                       b, i, decGot[g], rd16(db + 2 * i));
        }
    }
    float *tmpGot = (float *)bufs[4].ptr;
    float *outGot = (float *)bufs[5].ptr;
    for (int b = 0; b < BLOCKS; b++) {
        const uint8_t *tb = BLK(b, O_TMP), *ob = BLK(b, O_OUT);
        for (size_t i = 0; i < EL; i++) {
            uint32_t gb, gob;
            memcpy(&gb, &tmpGot[(size_t)b * EL + i], 4);
            memcpy(&gob, &outGot[(size_t)b * EL + i], 4);
            accF32(&tmp, gb, rd32f(tb + 4 * i));
            accF32(&out, gob, rd32f(ob + 4 * i));
        }
    }
    uint32_t *w16Got = (uint32_t *)bufs[6].ptr;
    uint32_t *wbfGot = (uint32_t *)bufs[7].ptr;
    for (int b = 0; b < BLOCKS; b++) {
        const uint8_t *w16b = BLK(b, O_W16), *wbfb = BLK(b, O_WBF);
        for (size_t i = 0; i < EL; i++) {
            size_t g = (size_t)b * EL + i;
            uint32_t e16 = rd16(w16b + 2 * i), eb = rd16(wbfb + 2 * i);
            w16.total++; wbf.total++;
            if (w16Got[g] == e16 || (f16NaN(w16Got[g]) && f16NaN(e16))) w16.exact++;
            if (wbfGot[g] == eb) wbf.exact++;
            else if (f16NaN(wbfGot[g]) && f16NaN(eb)) { wbf.nanMatch++; wbf.exact++; }
        }
    }

    int ok = mcg.exact == MCG_N && dec.exact == dec.total &&
             tmp.maxRel <= 1e-6 && out.maxRel <= 1e-6 &&
             w16.exact == w16.total && wbf.exact == wbf.total;
    printf("mcg decodeMcg     : %zu/%zu exact, maxAbsErr 0 (integer table)\n", mcg.exact, mcg.total);
    printf("tile decode (f16) : %zu/%zu exact across %d blocks (%d tiles)\n",
           dec.exact, dec.total, BLOCKS, BLOCKS * 64);
    printf("rowH128+suh (f32) : %zu exact + %zu NaN-pairs, maxRel %.3e, maxAbs %.3e\n",
           tmp.exact, tmp.nanMatch, tmp.maxRel, tmp.maxAbs);
    printf("colH128+svh (f32) : %zu exact + %zu NaN-pairs, maxRel %.3e, maxAbs %.3e\n",
           out.exact, out.nanMatch, out.maxRel, out.maxAbs);
    printf("final f16         : %zu/%zu exact (incl %zu NaN-pairs)\n", w16.exact, w16.total, w16.nanMatch);
    printf("final bf16        : %zu/%zu exact (incl %zu NaN-pairs)\n", wbf.exact, wbf.total, wbf.nanMatch);
    printf("RESULT            : %s (f32 tolerance rel 1e-6)\n", ok ? "PASS" : "FAIL");

    int moeOk = 1;
    if (argc == 6) {
        size_t mlen; uint8_t *mb = slurp(argv[4], &mlen);
        size_t moelen; uint8_t *moeSpv = slurp(argv[5], &moelen);
        moeOk = runMoeStage(dev, pd, queue, qf, mb, mlen, moeSpv, moelen);
        printf("RESULT moe        : %s\n", moeOk ? "PASS" : "FAIL");
        free(mb); free(moeSpv);
    }
    ok = ok && moeOk;

    for (int i = 0; i < 9; i++) { vkFreeMemory(dev, bufs[i].mem, NULL); vkDestroyBuffer(dev, bufs[i].buf, NULL); }
    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cpool, NULL);
    vkDestroyPipeline(dev, pipes[0], NULL);
    vkDestroyPipeline(dev, pipes[1], NULL);
    vkDestroyPipelineLayout(dev, mainPl, NULL);
    vkDestroyPipelineLayout(dev, mcgPl, NULL);
    vkDestroyShaderModule(dev, mainMod, NULL);
    vkDestroyShaderModule(dev, mcgMod, NULL);
    vkDestroyDescriptorPool(dev, pool, NULL);
    vkDestroyDescriptorSetLayout(dev, mainLayout, NULL);
    vkDestroyDescriptorSetLayout(dev, mcgLayout, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    return ok ? 0 : 1;
}
