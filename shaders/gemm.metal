#include <metal_stdlib>

using namespace metal;

struct MatrixDimsUniforms {
    uint m;
    uint k;
    uint n;
};

// MxK * KxN = MxN
kernel void gemm_tensor_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatrixDimsUniforms& sizes [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if(gid.x >= sizes.n || gid.y >= sizes.m) return;

    float sum = 0.0;
    for(uint i = 0; i < sizes.k; ++i) {
        sum += A[gid.y * sizes.k + i] * B[sizes.n * i + gid.x];
    }
    C[gid.y * 10 + gid.x] = sum;
}