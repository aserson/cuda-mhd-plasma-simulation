#pragma once
#include "cuda_runtime.h"

#include <limits>

#include <cufft.h>
#include <curand_kernel.h>

#if defined(_MSC_VER)
#define M_PI 3.141592653589793238462643
#endif

namespace mhd {
// Multiplication Kernels
__global__ static void MultDouble_kernel(double* input, unsigned int gridLength,
                                         double value, double* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = gridLength * x + y;

    output[idx] = input[idx] * value;
}

__global__ static void MultComplex_kernel(const cufftDoubleComplex* input,
                                          unsigned int gridLength, double value,
                                          cufftDoubleComplex* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    output[idx].x = input[idx].x * value;
    output[idx].y = input[idx].y * value;
}

// Differentiation Kernels
__global__ static void DiffByX_kernel(const cufftDoubleComplex* input,
                                      unsigned int gridLength,
                                      cufftDoubleComplex* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }

    output[idx].x = -(double)x * input[idx].y;
    output[idx].y = (double)x * input[idx].x;

    // The launch grid covers y in [0, N/2): zero the Nyquist column too,
    // the consumers inverse-transform the whole buffer
    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        output[idx + 1].x = 0.0;
        output[idx + 1].y = 0.0;
    }
}

__global__ static void DiffByY_kernel(const cufftDoubleComplex* input,
                                      unsigned int gridLength,
                                      cufftDoubleComplex* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    output[idx].x = -(double)y * input[idx].y;
    output[idx].y = (double)y * input[idx].x;

    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        output[idx + 1].x = 0.0;
        output[idx + 1].y = 0.0;
    }
}

__global__ static void LaplasOperator_kernel(const cufftDoubleComplex* input,
                                             unsigned int gridLength,
                                             cufftDoubleComplex* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    double value = -(double)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

__global__ static void MinusLaplasOperator_kernel(
    const cufftDoubleComplex* input, unsigned int gridLength,
    cufftDoubleComplex* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    double value = (double)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

__global__ static void InverseLaplasOperator_kernel(
    const cufftDoubleComplex* input, unsigned int gridLength,
    cufftDoubleComplex* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    double value = (idx == 0) ? 0.0 : (-1.) / (double)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

__global__ static void MinusInverseLaplasOperator_kernel(
    const cufftDoubleComplex* input, unsigned int gridLength,
    cufftDoubleComplex* output) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    double value = (idx == 0) ? 0.0 : 1. / (double)(x * x + y * y);

    output[idx].x = value * input[idx].x;
    output[idx].y = value * input[idx].y;
}

// Fused updateStream + updateCurrent: stream = vorticity / k^2,
// current = -k^2 * potential
__global__ static void StreamCurrent_kernel(
    const cufftDoubleComplex* vorticity, const cufftDoubleComplex* potential,
    cufftDoubleComplex* stream, cufftDoubleComplex* current,
    unsigned int gridLength) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2) {
        x = x - gridLength;
    }
    double value = (double)(x * x + y * y);
    double inverseValue = (idx == 0) ? 0.0 : 1. / value;

    stream[idx].x = inverseValue * vorticity[idx].x;
    stream[idx].y = inverseValue * vorticity[idx].y;

    current[idx].x = -value * potential[idx].x;
    current[idx].y = -value * potential[idx].y;
}

// Shared Memory Kernels
__global__ static void Max_kernel(const double* input, double* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;

    extern __shared__ double sharedBuffer[];
    sharedBuffer[tidx] = fabs(input[idx]);

    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) {
            sharedBuffer[tidx] =
                fmax(sharedBuffer[tidx], sharedBuffer[tidx + i]);
        }
        __syncthreads();
    }

    if (tidx == 0)
        output[blockIdx.x] = sharedBuffer[0];
}

__global__ static void EnergyTransform_kernel(double* velocityX,
                                              double* velocityY, double* energy,
                                              unsigned int gridLength,
                                              double lambda) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = gridLength * x + y;

    velocityX[idx] *= lambda;
    velocityY[idx] *= lambda;

    energy[idx] =
        (velocityX[idx] * velocityX[idx] + velocityY[idx] * velocityY[idx]) /
        2.;
}

// Single-block kernel: reduces the per-block partial maxima to output[0].
// Assumes non-negative input (partials of fabs).
__global__ static void MaxFinal_kernel(const double* input, unsigned int length,
                                       double* output) {
    int tidx = threadIdx.x;

    extern __shared__ double sharedBuffer[];

    double value = 0.0;
    for (unsigned int i = tidx; i < length; i += blockDim.x) {
        value = fmax(value, input[i]);
    }
    sharedBuffer[tidx] = value;

    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) {
            sharedBuffer[tidx] =
                fmax(sharedBuffer[tidx], sharedBuffer[tidx + i]);
        }
        __syncthreads();
    }

    if (tidx == 0)
        output[0] = sharedBuffer[0];
}

// Single-block kernel: reduces the per-block partial sums to output[0]
__global__ static void SumFinal_kernel(const double* input, unsigned int length,
                                       double* output) {
    int tidx = threadIdx.x;

    extern __shared__ double sharedBuffer[];

    double value = 0.0;
    for (unsigned int i = tidx; i < length; i += blockDim.x) {
        value += input[i];
    }
    sharedBuffer[tidx] = value;

    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) {
            sharedBuffer[tidx] = sharedBuffer[tidx] + sharedBuffer[tidx + i];
        }
        __syncthreads();
    }

    if (tidx == 0)
        output[0] = sharedBuffer[0];
}

__global__ static void EnergyIntegrate_kernel(double* field, double* sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;

    extern __shared__ double sharedBuffer[];
    sharedBuffer[tidx] = field[idx];

    __syncthreads();

    for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tidx < i) {
            sharedBuffer[tidx] = sharedBuffer[tidx] + sharedBuffer[tidx + i];
        }
        __syncthreads();
    }

    if (tidx == 0)
        sum[blockIdx.x] = sharedBuffer[0];
}

// Initial Conditions
__global__ static void FillStates(curandState* state, unsigned int gridLength,
                                  unsigned long seed) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    curand_init(seed, idx, 0, &state[idx]);
}

__global__ static void FillNormally_kernel(cufftDoubleComplex* f,
                                           curandState* state,
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

    f[idx].x = cos(phase) * value;
    f[idx].y = sin(phase) * value;
}
}  // namespace mhd
