/* probe.c — CPU reference + lavapipe run for probe.comp.
 * Usage: probe <probe.spv> [oracle.bin]
 * Prints each case's GPU bits, CPU IEEE reference bits, and match. Cases
 * 19-22 are the real-data Hadamard column (block/row from probe_col.h):
 * separate mul+add must equal the strict reference; if the device returns
 * the fma() bits there, the driver contracts `acc + he*x`.
 * With oracle.bin, also cross-checks case 19/21 against the oracle's tmp.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#include "probe_col.h"

#define VKC(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "VkError %d at %d: %s\n", r_, __LINE__, #x); exit(1); } } while (0)

#define NCASE 23
static float PB_SUH;

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

static const char *NAMES[NCASE] = {
    "passthrough min denormal", "min_den + min_den", "min_den * 2.0",
    "fma(min_den, 2, 0)", "ladder add 2^-120", "ladder mul 0.25 a",
    "ladder mul 0.25 b", "ladder mul 0.25 c", "sep (a*b)+c [contract probe]",
    "fma(a, b, c)", "(x-x)*-1 signed zero", "f16 RNE rem==half lsb0",
    "f16 RNE rem>half", "f16 2^-24 -> min subnorm", "f16 just below 2^-24",
    "bf16 max<1 -> round up", "bf16 maxfinite -> inf", "bf16 mid boundary",
    "HAD_SCALE * f16 minsub", "COL separate sum", "COL fma sum",
    "COL separate * suh", "COL fma * suh",
};

static void computeRef(uint32_t *ref) {
    float minDen, tiny, a, b3, p4;
    uint32_t mb = 0x00000001u, tb = 0x00400000u, ab = 0x3F800001u, bb = 0x3F800003u, p4b = 0x3F800004u;
    memcpy(&minDen, &mb, 4); memcpy(&tiny, &tb, 4);
    memcpy(&a, &ab, 4); memcpy(&b3, &bb, 4); memcpy(&p4, &p4b, 4);
    float c = -p4;
    uint16_t col[128] = { COL_RAW };
    memcpy(&ref[0], &minDen, 4);
    float t1 = minDen + minDen, t2 = minDen * 2.0f, t3 = fmaf(minDen, 2.0f, 0.0f);
    memcpy(&ref[1], &t1, 4); memcpy(&ref[2], &t2, 4); memcpy(&ref[3], &t3, 4);
    float l = tiny;
    l = l + tiny; memcpy(&ref[4], &l, 4);
    l = l * 0.25f; memcpy(&ref[5], &l, 4);
    l = l * 0.25f; memcpy(&ref[6], &l, 4);
    l = l * 0.25f; memcpy(&ref[7], &l, 4);
    float t8 = a * b3 + c, t9 = fmaf(a, b3, c);
    memcpy(&ref[8], &t8, 4); memcpy(&ref[9], &t9, 4);
    float z = 1.0f;
    float t10 = (z - z) * -1.0f;
    memcpy(&ref[10], &t10, 4);
    float f11, f12, f13, f14;
    uint32_t b11 = 0x3F801000u, b12 = 0x3F801001u, b13 = 0x33800000u, b14 = 0x337FFFFFu;
    memcpy(&f11, &b11, 4); memcpy(&f12, &b12, 4); memcpy(&f13, &b13, 4); memcpy(&f14, &b14, 4);
    ref[11] = f32ToF16(f11);
    ref[12] = f32ToF16(f12);
    ref[13] = f32ToF16(f13);
    ref[14] = f32ToF16(f14);
    ref[15] = f32ToBf16(0x3F7FFFFFu);
    ref[16] = f32ToBf16(0x7F7FFFFFu);
    ref[17] = f32ToBf16(0x3F807FFFu);
    float hsp = 0.08838834764831845f * f16ToF32(0x0001u);
    memcpy(&ref[18], &hsp, 4);
    float accS = 0.0f, accF = 0.0f;
    for (int k = 0; k < 128; k++) {
        float he = (__builtin_popcount((int)PB_ROW & k) & 1) ? -0.08838834764831845f : 0.08838834764831845f;
        float x = f16ToF32(col[k]);
        accS = accS + he * x;
        accF = fmaf(he, x, accF);
    }
    memcpy(&ref[19], &accS, 4);
    memcpy(&ref[20], &accF, 4);
    float s21 = PB_SUH * accS, s22 = PB_SUH * accF;
    memcpy(&ref[21], &s21, 4);
    memcpy(&ref[22], &s22, 4);
}

static uint8_t *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *len = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc(*len);
    if (fread(p, 1, *len, f) != *len) exit(1);
    fclose(f); return p;
}

static uint32_t findMem(VkPhysicalDevice pd, VkMemoryRequirements req, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    fprintf(stderr, "no memory type\n"); exit(1);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s probe.spv [oracle.bin]\n", argv[0]); return 2; }
    PB_SUH = 0.0f;
    memcpy(&PB_SUH, &(uint32_t){PB_SUH_BITS}, 4);
    uint32_t ref[NCASE];
    computeRef(ref);

    uint32_t oracleTmp = UINT32_MAX, oracleTmpFma = UINT32_MAX;
    if (argc > 2) {
        size_t olen; uint8_t *o = slurp(argv[2], &olen);
        const size_t EL = 16384, BOFF = 4096 + 256 + 256 + 2 * EL + 4 * EL + 4 * EL + 2 * EL + 2 * EL;
        size_t base = 20u + 131072u;
        uint32_t b = PB_BLOCK, r = PB_ROW;
        memcpy(&oracleTmp, o + base + (size_t)b * BOFF + 4096u + 512u + 2u * EL + 4u * ((size_t)r * 128u + r), 4);
        (void)oracleTmpFma;
    }

    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    VkInstance inst; VKC(vkCreateInstance(&ici, NULL, &inst));
    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice *pds = malloc(nd * sizeof *pds);
    vkEnumeratePhysicalDevices(inst, &nd, pds);
    VkPhysicalDevice pd = pds[0];
    const char *want = getenv("EXL3_VK_DEVICE");
    VkPhysicalDeviceProperties pp;
    for (uint32_t i = 0; i < nd; i++) {
        vkGetPhysicalDeviceProperties(pds[i], &pp);
        int hit = want ? strstr(pp.deviceName, want) != NULL
                       : (strstr(pp.deviceName, "llvmpipe") || strstr(pp.deviceName, "lavapipe"));
        if (hit) { pd = pds[i]; break; }
    }
    vkGetPhysicalDeviceProperties(pd, &pp);
    printf("device: %s\n", pp.deviceName);

    uint32_t qf = UINT32_MAX, nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qp = malloc(nq * sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi };
    VkDevice dev; VKC(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qf, 0, &queue);

    VkBuffer buf; VkDeviceMemory mem; void *ptr;
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = NCASE * 4, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
    VKC(vkCreateBuffer(dev, &bi, NULL, &buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, buf, &req);
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = findMem(pd, req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
    VKC(vkAllocateMemory(dev, &ai, NULL, &mem));
    VKC(vkBindBufferMemory(dev, buf, mem, 0));
    VKC(vkMapMemory(dev, mem, 0, NCASE * 4, 0, &ptr));

    VkDescriptorSetLayoutBinding rb = { .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &rb };
    VkDescriptorSetLayout layout; VKC(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &layout));
    VkDescriptorPoolSize ps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkDescriptorPool pool; VKC(vkCreateDescriptorPool(dev, &pci, NULL, &pool));
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout };
    VkDescriptorSet set; VKC(vkAllocateDescriptorSets(dev, &dai, &set));
    VkWriteDescriptorSet wr = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
        .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &(VkDescriptorBufferInfo){ .buffer = buf, .offset = 0, .range = VK_WHOLE_SIZE } };
    vkUpdateDescriptorSets(dev, 1, &wr, 0, NULL);

    size_t sl; uint8_t *spv = slurp(argv[1], &sl);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sl, .pCode = (const uint32_t *)spv };
    VkShaderModule mod; VKC(vkCreateShaderModule(dev, &smci, NULL, &mod));
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &layout };
    VkPipelineLayout pl; VKC(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main" }, .layout = pl };
    VkPipeline pipe; VKC(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

    VkCommandPoolCreateInfo cpoi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf };
    VkCommandPool cpool; VKC(vkCreateCommandPool(dev, &cpoi, NULL, &cpool));
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cmd; VKC(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo bbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VKC(vkBeginCommandBuffer(cmd, &bbi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
    vkCmdDispatch(cmd, 1, 1, 1);
    VKC(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; VKC(vkCreateFence(dev, &fci, NULL, &fence));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
    VKC(vkQueueSubmit(queue, 1, &si, fence));
    VKC(vkWaitForFences(dev, 1, &fence, VK_TRUE, 60ull * 1000ull * 1000ull * 1000ull));

    uint32_t *got = (uint32_t *)ptr;
    int sepIsFma = 0;
    printf("%-28s %-11s %-11s %s\n", "case", "GPU", "CPU-ref", "match");
    for (int i = 0; i < NCASE; i++) {
        int match = got[i] == ref[i];
        printf("%-28s 0x%08X 0x%08X %s\n", NAMES[i], got[i], ref[i], match ? "OK" : "DIFF");
    }
    /* contraction verdict on this device */
    if (got[19] != ref[19] && got[19] == ref[20]) {
        sepIsFma = 1;
        printf("CONTRACTION: separate-op column returned the fma() bits -> driver fuses acc + he*x\n");
    } else if (got[19] == ref[19]) {
        printf("NO-CONTRACTION: separate-op column matched strict IEEE rounding\n");
    }
    if (got[1] == 0 && ref[1] != 0) printf("FTZ: denormal add flushed to zero (CPU keeps 0x%08X)\n", ref[1]);
    else if (got[1] == ref[1]) printf("NO-FTZ: denormal add kept denormal bits\n");
    if (argc > 2 && oracleTmp != UINT32_MAX)
        printf("oracle tmp[%u][%u][%u] = 0x%08X | GPU sep*suh 0x%08X | CPU fma*suh 0x%08X\n",
               PB_BLOCK, PB_ROW, PB_ROW, oracleTmp, got[21], ref[22]);
    return sepIsFma ? 3 : 0;
}
