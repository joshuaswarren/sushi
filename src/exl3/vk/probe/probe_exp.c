/* probe_exp.c - device exp() ulp profile probe (M3dIntegrate).
 * Usage: probe_exp <probe_exp.spv> [arm]
 *   arm "default" (or no arg): plain GLSL exp() profile, informational.
 *   arm "accurate": the EXL3_ACCURATE_EXP build, rc-gated at max|ulp| <= 1
 *     against correctly rounded.
 * The spirv file name selects nothing; compile probe_exp.comp with or
 * without -DEXL3_ACCURATE_EXP=1 and pass the matching arm.
 * Exit 0 iff all sweeps ran (and, for the accurate arm, the gate held).
 * Env:
 *   EXL3_VK_DEVICE=<substring>  device pick (default "lavapipe", falls
 *     back to the first compute device)
 *   EXL3_VK_PROBE=1             enumerate devices and exit
 *   EXL3_PROBE_UCAP=<n>         ulp histogram bucket cap (default 8)
 *
 * Sweeps (identical inputs for both arms):
 *   f16domain - ALL 65536 f16 bit patterns widened to f32. The fused
 *     kernel only ever evaluates exp at f16-rounded points (the finished
 *     gate values are f16 bits), so this is the complete reachable domain.
 *   dense     - 2^24 f32 bit patterns at a fixed stride over the 2^32
 *     word space; nonfinite samples are dropped on the host.
 *   edges     - hand-built boundary set: overflow/underflow cutoffs,
 *     k = -127/-126/+127/+128 boundary lattices, +-ln2/2 points,
 *     denormals, min/max subnormal, maxfloat, inf, nan, +-0.
 * Correctly rounded = expl (64-bit mantissa) rounded to float; glibc
 * expf = the host libm single-precision reference.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#define VKC(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "VkError %d at %d: %s\n", r_, __LINE__, #x); exit(1); } } while (0)

#define PROBE_BINDINGS 2
#define PC_U32 1

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

static VkDevice g_dev;
static VkQueue g_queue;
static VkPhysicalDevice g_pd;

static void makeBuf(Buf *b, VkDeviceSize size) {
    b->size = size;
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    VKC(vkCreateBuffer(g_dev, &bi, NULL, &b->buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_dev, b->buf, &req);
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = findMem(g_pd, req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
    VKC(vkAllocateMemory(g_dev, &ai, NULL, &b->mem));
    VKC(vkBindBufferMemory(g_dev, b->buf, b->mem, 0));
    VKC(vkMapMemory(g_dev, b->mem, 0, size, 0, &b->ptr));
    memset(b->ptr, 0, (size_t)size);
}

static void freeBuf(Buf *b) {
    vkFreeMemory(g_dev, b->mem, NULL);
    vkDestroyBuffer(g_dev, b->buf, NULL);
}

static void runShader(const uint8_t *spv, size_t splen, Buf *in, Buf *out, uint32_t n) {
    VkDescriptorSetLayoutBinding rb[PROBE_BINDINGS];
    for (int i = 0; i < PROBE_BINDINGS; i++)
        rb[i] = (VkDescriptorSetLayoutBinding){ .binding = (uint32_t)i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = PROBE_BINDINGS, .pBindings = rb };
    VkDescriptorSetLayout layout;
    VKC(vkCreateDescriptorSetLayout(g_dev, &dlci, NULL, &layout));
    VkDescriptorPoolSize ps = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = PROBE_BINDINGS };
    VkDescriptorPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkDescriptorPool pool;
    VKC(vkCreateDescriptorPool(g_dev, &pci, NULL, &pool));
    VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout };
    VkDescriptorSet set;
    VKC(vkAllocateDescriptorSets(g_dev, &dai, &set));
    VkDescriptorBufferInfo dbs[PROBE_BINDINGS] = {
        { .buffer = in->buf, .offset = 0, .range = in->size },
        { .buffer = out->buf, .offset = 0, .range = out->size } };
    VkWriteDescriptorSet ws[PROBE_BINDINGS];
    for (int i = 0; i < PROBE_BINDINGS; i++)
        ws[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set, .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbs[i] };
    vkUpdateDescriptorSets(g_dev, PROBE_BINDINGS, ws, 0, NULL);

    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0, .size = PC_U32 * 4 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout pl;
    VKC(vkCreatePipelineLayout(g_dev, &plci, NULL, &pl));
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = splen, .pCode = (const uint32_t *)spv };
    VkShaderModule mod;
    VKC(vkCreateShaderModule(g_dev, &smci, NULL, &mod));
    VkComputePipelineCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main" },
        .layout = pl };
    VkPipeline pipe;
    VKC(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

    VkCommandPoolCreateInfo cpi2 = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT };
    VkCommandPool pool2;
    VKC(vkCreateCommandPool(g_dev, &cpi2, NULL, &pool2));
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandPool = pool2,
        .commandBufferCount = 1 };
    VkCommandBuffer cb;
    VKC(vkAllocateCommandBuffers(g_dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VKC(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, PC_U32 * 4, &n);
    vkCmdDispatch(cb, (n + 127) / 128, 1, 1);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cb };
    VKC(vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE));
    VKC(vkQueueWaitIdle(g_queue));

    vkDestroyCommandPool(g_dev, pool2, NULL);
    vkDestroyPipeline(g_dev, pipe, NULL);
    vkDestroyShaderModule(g_dev, mod, NULL);
    vkDestroyPipelineLayout(g_dev, pl, NULL);
    vkDestroyDescriptorPool(g_dev, pool, NULL);
    vkDestroyDescriptorSetLayout(g_dev, layout, NULL);
}

/* ordered int64 key of an f32 bit pattern (total order) */
static int64_t ordered(uint32_t b) {
    return (b >> 31) ? (int64_t)0xFFFFFFFFu - (int64_t)b : (int64_t)b;
}

static int64_t ulp_diff(uint32_t got, uint32_t ref) {
    float gf, rf;
    memcpy(&gf, &got, 4);
    memcpy(&rf, &ref, 4);
    if (got == ref) return 0;
    if (isnan(gf) && isnan(rf)) return 0;
    if (isinf(gf) || isinf(rf))
        return (isinf(gf) && isinf(rf) && signbit(gf) == signbit(rf)) ? 0 : 1000000;
    int64_t d = ordered(got) - ordered(ref);
    return d < 0 ? -d : d;
}

static void sweep_run(const uint8_t *spv, size_t splen, const char *name,
                      uint32_t *xs, size_t n, int gated, int ucap) {
    Buf in, out;
    makeBuf(&in, (VkDeviceSize)n * 4);
    makeBuf(&out, (VkDeviceSize)n * 4);
    memcpy(in.ptr, xs, (size_t)n * 4);
    runShader(spv, splen, &in, &out, (uint32_t)n);

    size_t ndiff_f = 0, ndiff_cr = 0;
    int64_t max_abs = 0;
    uint32_t worst_in = 0, worst_got = 0, worst_ref = 0;
    long *hist = calloc((size_t)(2 * ucap + 2), sizeof(long));
    uint32_t *got = (uint32_t *)out.ptr;
    for (size_t i = 0; i < n; i++) {
        float xf, gf;
        memcpy(&xf, &xs[i], 4);
        memcpy(&gf, &got[i], 4);
        (void)gf;
        float crf_l = (float)expl((long double)xf);
        uint32_t crb;
        memcpy(&crb, &crf_l, 4);
        float ef = expf(xf);
        uint32_t eb;
        memcpy(&eb, &ef, 4);
        if (ulp_diff(got[i], eb) != 0) ndiff_f++;
        int64_t d = ulp_diff(got[i], crb);
        if (d != 0) ndiff_cr++;
        int64_t a = d < 0 ? -d : d;
        if (a > max_abs) {
            max_abs = a;
            worst_in = xs[i];
            worst_got = got[i];
            worst_ref = crb;
        }
        if (a <= (int64_t)ucap) hist[d + ucap]++;
        else hist[2 * ucap + 1]++;
    }
    printf("sweep %-10s n=%zu ndiff_vs_expf=%zu ndiff_vs_cr=%zu maxAbsUlp=%lld hist[",
           name, n, ndiff_f, ndiff_cr, (long long)max_abs);
    for (int i = 0; i <= 2 * ucap + 1; i++)
        printf("%ld%s", hist[i], i <= 2 * ucap ? "," : "");
    printf("]");
    if (max_abs > 0) {
        float wi, wg, wr;
        memcpy(&wi, &worst_in, 4);
        memcpy(&wg, &worst_got, 4);
        memcpy(&wr, &worst_ref, 4);
        printf(" worst x=%.9g got=%.9g cr=%.9g", (double)wi, (double)wg, (double)wr);
    }
    printf("\n");
    freeBuf(&in);
    freeBuf(&out);
    free(hist);
    if (gated && max_abs > 1) {
        printf("GATE FAIL: %s maxAbsUlp %lld > 1\n", name, (long long)max_abs);
        exit(1);
    }
}

static size_t build_f16domain(uint32_t **out) {
    uint32_t *v = malloc(65536 * 4);
    size_t n = 0;
    for (uint32_t h = 0; h < 0x10000u; h++)
        v[n++] = h << 16; /* f16 bit pattern widened to f32 */
    *out = v;
    return n;
}

static size_t build_dense(uint32_t **out) {
    const size_t n = 1u << 24;
    uint32_t *v = malloc(n * 4);
    for (size_t i = 0; i < n; i++) {
        /* fixed stride over the 2^32 word space, phi-spread */
        v[i] = (uint32_t)((i * 2654435769ull) >> 8);
    }
    *out = v;
    return n;
}

static size_t build_edges(uint32_t **out) {
    static uint32_t v[4096];
    size_t n = 0;
    v[n++] = 0x00000000u; /* +0 */
    v[n++] = 0x80000000u; /* -0 */
    v[n++] = 0x00000001u; /* min denormal */
    v[n++] = 0x80000001u;
    v[n++] = 0x007FFFFFu; /* max denormal */
    v[n++] = 0x807FFFFFu;
    v[n++] = 0x00800000u; /* min normal */
    v[n++] = 0x80800000u;
    v[n++] = 0x7F7FFFFFu; /* maxfloat */
    v[n++] = 0xFF7FFFFFu;
    v[n++] = 0x7F800000u; /* +inf */
    v[n++] = 0xFF800000u; /* -inf */
    v[n++] = 0x7FC00000u; /* nan */
    v[n++] = 0xFFC00000u;
    /* overflow boundary: 88.72283935546875f and its two f32 neighbors */
    v[n++] = 0x42B17217u;
    v[n++] = 0x42B17216u;
    v[n++] = 0x42B17218u;
    v[n++] = 0xC2B17217u;
    v[n++] = 0xC2B17218u;
    /* underflow-to-zero boundary: -103.972084f and neighbors */
    v[n++] = 0xC2CFFA93u; /* -103.97208 */
    v[n++] = 0xC2CFFA92u;
    v[n++] = 0xC2CFFA94u;
    /* k boundary lattices: x*ln2 near integers, k = 126,127,128, -127,-126 */
    for (int k = -130; k <= 130; k++) {
        for (int t = -1; t <= 1; t++) {
            double x = ((double)k - 0.5 * t) * 0.6931471805599453;
            float xf = (float)x;
            uint32_t b;
            memcpy(&b, &xf, 4);
            v[n++] = b;
            uint32_t b1 = b + 1;
            v[n++] = b1;
        }
    }
    /* the exact reduction r-extremes: +-ln2/2 */
    {
        float p = 0.34657359027997264f, m = -0.34657359027997264f;
        uint32_t b;
        memcpy(&b, &p, 4); v[n++] = b;
        memcpy(&b, &m, 4); v[n++] = b;
    }
    *out = v;
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <probe_exp.spv> [default|accurate]\n", argv[0]);
        return 2;
    }
    const char *arm = (argc > 2) ? argv[2] : "default";
    int gated = strcmp(arm, "accurate") == 0;
    int ucap = 8;
    if (getenv("EXL3_PROBE_UCAP")) ucap = atoi(getenv("EXL3_PROBE_UCAP"));
    size_t splen;
    uint8_t *spv = slurp(argv[1], &splen);

    VkInstance inst;
    VKC(vkCreateInstance(&(VkInstanceCreateInfo){
                             .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO},
                         NULL, &inst));
    uint32_t nd = 0;
    VKC(vkEnumeratePhysicalDevices(inst, &nd, NULL));
    if (nd == 0) { fprintf(stderr, "no vulkan devices\n"); return 1; }
    VkPhysicalDevice *pds = malloc(nd * sizeof(VkPhysicalDevice));
    VKC(vkEnumeratePhysicalDevices(inst, &nd, pds));
    const char *want = getenv("EXL3_VK_DEVICE");
    if (want == NULL) want = "lavapipe";
    g_pd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(pds[i], &pp);
        printf("device %u           : %s (vulkan %u.%u)\n", i, pp.deviceName,
               VK_VERSION_MAJOR(pp.apiVersion), VK_VERSION_MINOR(pp.apiVersion));
        if (strstr(pp.deviceName, want) != NULL && g_pd == VK_NULL_HANDLE) g_pd = pds[i];
    }
    if (getenv("EXL3_VK_PROBE")) return 0;
    if (g_pd == VK_NULL_HANDLE) {
        fprintf(stderr, "no device matching '%s'\n", want);
        return 1;
    }
    uint32_t qf = UINT32_MAX, nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL);
    VkQueueFamilyProperties *qp = malloc(nq * sizeof(*qp));
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
    for (uint32_t i = 0; i < nq; i++)
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    if (qf == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 1; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi };
    VKC(vkCreateDevice(g_pd, &dci, NULL, &g_dev));
    vkGetDeviceQueue(g_dev, qf, 0, &g_queue);

    uint32_t *a, *b, *e;
    size_t na = build_f16domain(&a);
    size_t nb = build_dense(&b);
    size_t ne = build_edges(&e);
    printf("=== exp probe arm=%s ucap=%d ===\n", arm, ucap);
    sweep_run(spv, splen, "f16domain", a, na, gated, ucap);
    sweep_run(spv, splen, "dense", b, nb, gated, ucap);
    sweep_run(spv, splen, "edges", e, ne, gated, ucap);
    printf("PROBE_%s_OK\n", gated ? "ACCURATE" : "DEFAULT");
    return 0;
}
