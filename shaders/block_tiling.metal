#include <metal_stdlib>

using namespace metal;
#define TILE_SIZE 16
struct matrixdimsuniforms {
    uint m;
    uint k;
    uint n;
};

// mxk * kxn = mxn
kernel void gemm_tensor_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant matrixdimsuniforms& sizes [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint K = sizes.k;
    float sum=0.0;

    for(uint shift = 0; shift < K; shift +=TILE_SIZE) {
        As[lid.y][lid.x] = (gid.x < sizes.k && gid.y < sizes.m) ? A[gid.y * K + (gid.x + shift)] : 0.0;
        Bs[lid.y][lid.x] = (gid.x < sizes.k && gid.y < sizes.n) ? B[(gid.y + shift) * sizes.n + gid.x] : 0.0;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=0; i<TILE_SIZE; ++i) {
            sum += As[gid.y][i] * Bs[i][gid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[gid.y * sizes.n + gid.x] = sum;
}
