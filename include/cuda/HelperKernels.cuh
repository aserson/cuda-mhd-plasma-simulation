#pragma once
#include "cuda_runtime.h"

#include <limits>

#include <cufft.h>
#include <curand_kernel.h>

#include "Buffers.cuh"

#if defined(_MSC_VER)
#define M_PI 3.141592653589793238462643
#endif

namespace mhd {
// Shared memory in template kernels is declared through a raw byte array:
// extern __shared__ arrays of different element types would collide between
// instantiations
template <typename T>
__device__ inline T* sharedBuffer() {
    extern __shared__ unsigned char sharedRaw[];
    return reinterpret_cast<T*>(sharedRaw);
}

// Multiplication Kernels
template <typename T>
__global__ void MultReal_kernel(T* input, unsigned int gridLength, T value,
                                T* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = gridLength * x + y;

    output[idx] = input[idx] * value;
}

template <typename T>
__global__ void MultComplex_kernel(const Complex_t<T>* input,
                                   unsigned int gridLength, T value,
                                   Complex_t<T>* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    output[idx].x = input[idx].x * value;
    output[idx].y = input[idx].y * value;
}

// Differentiation Kernels
template <typename T>
__global__ void DiffByX_kernel(const Complex_t<T>* input,
                               unsigned int gridLength, Complex_t<T>* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }

    output[idx].x = -(T)x * input[idx].y;
    output[idx].y = (T)x * input[idx].x;

    // The launch grid covers y in [0, N/2): zero the Nyquist column too,
    // the consumers inverse-transform the whole buffer
    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        output[idx + 1].x = T(0.0);
        output[idx + 1].y = T(0.0);
    }
}

template <typename T>
__global__ void DiffByY_kernel(const Complex_t<T>* input,
                               unsigned int gridLength, Complex_t<T>* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    output[idx].x = -(T)y * input[idx].y;
    output[idx].y = (T)y * input[idx].x;

    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        output[idx + 1].x = T(0.0);
        output[idx + 1].y = T(0.0);
    }
}

template <typename T>
__global__ void LaplasOperator_kernel(const Complex_t<T>* input,
                                      unsigned int gridLength,
                                      Complex_t<T>* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    T value = -(T)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

template <typename T>
__global__ void MinusLaplasOperator_kernel(const Complex_t<T>* input,
                                           unsigned int gridLength,
                                           Complex_t<T>* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    T value = (T)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

template <typename T>
__global__ void InverseLaplasOperator_kernel(const Complex_t<T>* input,
                                             unsigned int gridLength,
                                             Complex_t<T>* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    T value = (idx == 0) ? T(0.0) : T(-1.0) / (T)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

template <typename T>
__global__ void MinusInverseLaplasOperator_kernel(const Complex_t<T>* input,
                                                  unsigned int gridLength,
                                                  Complex_t<T>* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    T value = (idx == 0) ? T(0.0) : T(1.0) / (T)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

// Fused updateStream + updateCurrent: stream = vorticity / k^2,
// current = -k^2 * potential
template <typename T>
__global__ void StreamCurrent_kernel(const Complex_t<T>* vorticity,
                                     const Complex_t<T>* potential,
                                     Complex_t<T>* stream,
                                     Complex_t<T>* current,
                                     unsigned int gridLength) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    T value = (T)(x * x + y * y);
    T inverseValue = (idx == 0) ? T(0.0) : T(1.0) / value;

    stream[idx].x = inverseValue * vorticity[idx].x;
    stream[idx].y = inverseValue * vorticity[idx].y;

    current[idx].x = -value * potential[idx].x;
    current[idx].y = -value * potential[idx].y;
}

// Shared Memory Kernels
template <typename T>
__global__ void Max_kernel(const T* input, T* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;

    T* shared = sharedBuffer<T>();
    shared[tidx] = fabs(input[idx]);

    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) {
            shared[tidx] = fmax(shared[tidx], shared[tidx + i]);
        }
        __syncthreads();
    }

    if (tidx == 0)
        output[blockIdx.x] = shared[0];
}

template <typename T>
__global__ void EnergyTransform_kernel(T* velocityX, T* velocityY, T* energy,
                                       unsigned int gridLength, T lambda) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = gridLength * x + y;

    velocityX[idx] *= lambda;
    velocityY[idx] *= lambda;

    energy[idx] =
        (velocityX[idx] * velocityX[idx] + velocityY[idx] * velocityY[idx]) /
        T(2.0);
}

// Single-block kernel: reduces the per-block partial maxima to output[0].
// Assumes non-negative input (partials of fabs).
template <typename T>
__global__ void MaxFinal_kernel(const T* input, unsigned int length,
                                T* output) {
    int tidx = threadIdx.x;

    T* shared = sharedBuffer<T>();

    T value = T(0.0);
    for (unsigned int i = tidx; i < length; i += blockDim.x) {
        value = fmax(value, input[i]);
    }
    shared[tidx] = value;

    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) {
            shared[tidx] = fmax(shared[tidx], shared[tidx + i]);
        }
        __syncthreads();
    }

    if (tidx == 0)
        output[0] = shared[0];
}

// Single-block kernel: reduces the per-block partial sums to output[0]
template <typename T>
__global__ void SumFinal_kernel(const T* input, unsigned int length,
                                T* output) {
    int tidx = threadIdx.x;

    T* shared = sharedBuffer<T>();

    T value = T(0.0);
    for (unsigned int i = tidx; i < length; i += blockDim.x) {
        value += input[i];
    }
    shared[tidx] = value;

    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) {
            shared[tidx] = shared[tidx] + shared[tidx + i];
        }
        __syncthreads();
    }

    if (tidx == 0)
        output[0] = shared[0];
}

template <typename T>
__global__ void EnergyIntegrate_kernel(T* field, T* sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;

    T* shared = sharedBuffer<T>();
    shared[tidx] = field[idx];

    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) {
            shared[tidx] = shared[tidx] + shared[tidx + i];
        }
        __syncthreads();
    }

    if (tidx == 0)
        sum[blockIdx.x] = shared[0];
}

// Initial Conditions
__global__ static void FillStates(curandState* state, unsigned int gridLength,
                                  unsigned long seed) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    curand_init(seed, idx, 0, &state[idx]);
}

// The spectrum shape is computed in double regardless of the solver type,
// so both precisions start from the same initial conditions
template <typename T>
__global__ void FillNormally_kernel(Complex_t<T>* f, curandState* state,
                                    unsigned int gridLength,
                                    unsigned int averageWN) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    double k = sqrt((double)(x * x + y * y));

    double value =
        (k > 0) ? (double)(gridLength * gridLength) *
                      exp(-(k * k) / (2.f * (double)(averageWN * averageWN))) /
                      sqrt(k)
                : 0.0;

    double phase = 2.f * M_PI * curand_uniform(&state[idx]);

    f[idx].x = (T)(cos(phase) * value);
    f[idx].y = (T)(sin(phase) * value);
}
}  // namespace mhd
