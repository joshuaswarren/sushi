/* harness.c — M1 oracle runner for exl3_decode.comp on lavapipe.
 * Usage: harness <oracle.bin> <exl3_decode.spv> <exl3_mcg_test.spv>
 * Exit 0 only if every integer part is exact and every f32 part is within
 * rel 1e-6 (scope doc section 5(i)). NaN==NaN counts as a match and is
 * reported separately. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s oracle.bin decode.spv mcg.spv\n", argv[0]); return 2; }
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
