#include <metal_stdlib>

using namespace metal;

// MxK * KxN = MxN
kernel void add_array(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    C[index] = A[index] + B[index];
}