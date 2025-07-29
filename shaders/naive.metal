#include <metal_stdlib>

using namespace metal;

struct matrixdimsuniforms {
    uint m;
    uint k;
    uint n;
};

// mxk * kxn = mxn
kernel void gemm_tensor_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant matrixdimsuniforms& sizes [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if(gid.x >= sizes.n || gid.y >= sizes.m) return;

    float sum = 0.0;
    for(uint i = 0; i < sizes.k; ++i) {
        sum += a[gid.y * sizes.k + i] * b[sizes.n * i + gid.x];
    }
    c[gid.y * sizes.n + gid.x] = sum;
}
