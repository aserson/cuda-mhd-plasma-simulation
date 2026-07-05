#pragma once
#include "cuda_runtime.h"

#include <cufft.h>

#include "Buffers.cuh"

namespace mhd {

// Jacobian Kernels
template <typename T>
__global__ void DealaliasingDiffXY_kernel(const Complex_t<T>* input,
                                          Complex_t<T>* outputX,
                                          Complex_t<T>* outputY,
                                          unsigned int gridLength,
                                          unsigned int dealWN) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;

    if ((abs(x) < dealWN) && (abs(y) < dealWN)) {
        outputX[idx].x = -(T)x * input[idx].y;
        outputX[idx].y = (T)x * input[idx].x;
        outputY[idx].x = -(T)y * input[idx].y;
        outputY[idx].y = (T)y * input[idx].x;
    } else {
        outputX[idx].x = T(0.0);
        outputX[idx].y = T(0.0);
        outputY[idx].x = T(0.0);
        outputY[idx].y = T(0.0);
    }

    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        outputX[idx + 1].x = T(0.0);
        outputX[idx + 1].y = T(0.0);
        outputY[idx + 1].x = T(0.0);
        outputY[idx + 1].y = T(0.0);
    }
}

// output may alias one of the inputs: each thread only writes the element
// it has already read
template <typename T>
__global__ void Jacobian_kernel(const T* leftX, const T* rightY, const T* leftY,
                                const T* rightX, T* output,
                                unsigned int gridLength, T lambda) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = gridLength * x + y;

    output[idx] = (leftX[idx] * rightY[idx] - leftY[idx] * rightX[idx]) *
                  lambda * lambda;
}

// Equation Kernels
// Dealiasing of the Jacobian spectra is folded in via the dealWN window:
// out-of-window Jacobian modes are never read anywhere else
template <typename T>
__global__ void KineticRigthPart_kernel(const Complex_t<T>* w,
                                        const Complex_t<T>* jacobianFirst,
                                        const Complex_t<T>* jacobianSecond,
                                        Complex_t<T>* rightPart,
                                        unsigned int gridLength, T nu,
                                        unsigned int dealWN) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;
    T value = (T)(x * x + y * y);

    if ((abs(x) < dealWN) && (abs(y) < dealWN)) {
        rightPart[idx].x = jacobianFirst[idx].x + jacobianSecond[idx].x -
                           nu * value * w[idx].x;
        rightPart[idx].y = jacobianFirst[idx].y + jacobianSecond[idx].y -
                           nu * value * w[idx].y;
    } else {
        rightPart[idx].x = -nu * value * w[idx].x;
        rightPart[idx].y = -nu * value * w[idx].y;
    }
}

template <typename T>
__global__ void ThirdRigthPart_kernel(const Complex_t<T>* a,
                                      const Complex_t<T>* jacobian,
                                      Complex_t<T>* rightPart,
                                      unsigned int gridLength, T eta,
                                      unsigned int dealWN) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;
    T value = (T)(x * x + y * y);

    if ((abs(x) < dealWN) && (abs(y) < dealWN)) {
        rightPart[idx].x = jacobian[idx].x - eta * value * a[idx].x;
        rightPart[idx].y = jacobian[idx].y - eta * value * a[idx].y;
    } else {
        rightPart[idx].x = -eta * value * a[idx].x;
        rightPart[idx].y = -eta * value * a[idx].y;
    }
}

// Time Scheme Kernels
// The time step is read from device memory so that the kernels can be
// captured into a CUDA graph while dt changes between launches
template <typename T>
__global__ void TimeScheme_kernel(Complex_t<T>* field,
                                  const Complex_t<T>* oldField,
                                  const Complex_t<T>* rightPart,
                                  unsigned int gridLength, const T* dt,
                                  T weight) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    field[idx].x = oldField[idx].x + weight * rightPart[idx].x * dt[0];
    field[idx].y = oldField[idx].y + weight * rightPart[idx].y * dt[0];
}

// Final integration stage: also stores the new value as the old field for
// the next time step, replacing a separate device-to-device copy
template <typename T>
__global__ void TimeSchemeFinal_kernel(Complex_t<T>* field,
                                       Complex_t<T>* oldField,
                                       const Complex_t<T>* rightPart,
                                       unsigned int gridLength, const T* dt,
                                       T weight) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    Complex_t<T> value;
    value.x = oldField[idx].x + weight * rightPart[idx].x * dt[0];
    value.y = oldField[idx].y + weight * rightPart[idx].y * dt[0];

    field[idx] = value;
    oldField[idx] = value;
}
}  // namespace mhd
