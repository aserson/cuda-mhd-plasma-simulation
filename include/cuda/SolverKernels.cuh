#pragma once
#include "cuda_runtime.h"

#include <cufft.h>

namespace mhd {

// Jacobian Kernels
__global__ void DealaliasingDiffXY_kernel(const cufftDoubleComplex* input,
                                          cufftDoubleComplex* outputX,
                                          cufftDoubleComplex* outputY,
                                          unsigned int gridLength,
                                          unsigned int dealWN) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;

    if ((abs(x) < dealWN) && (abs(y) < dealWN)) {
        outputX[idx].x = -(double)x * input[idx].y;
        outputX[idx].y = (double)x * input[idx].x;
        outputY[idx].x = -(double)y * input[idx].y;
        outputY[idx].y = (double)y * input[idx].x;
    } else {
        outputX[idx].x = 0.0;
        outputX[idx].y = 0.0;
        outputY[idx].x = 0.0;
        outputY[idx].y = 0.0;
    }

    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        outputX[idx + 1].x = 0.0;
        outputX[idx + 1].y = 0.0;
        outputY[idx + 1].x = 0.0;
        outputY[idx + 1].y = 0.0;
    }
}

// output may alias one of the inputs: each thread only writes the element
// it has already read
__global__ void Jacobian_kernel(const double* leftX, const double* rightY,
                                const double* leftY, const double* rightX,
                                double* output, unsigned int gridLength,
                                double lambda) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = gridLength * x + y;

    output[idx] = (leftX[idx] * rightY[idx] - leftY[idx] * rightX[idx]) *
                  lambda * lambda;
}

// Equation Kernels
// Dealiasing of the Jacobian spectra is folded in via the dealWN window:
// out-of-window Jacobian modes are never read anywhere else
__global__ void KineticRigthPart_kernel(
    const cufftDoubleComplex* w, const cufftDoubleComplex* jacobianFirst,
    const cufftDoubleComplex* jacobianSecond, cufftDoubleComplex* rightPart,
    unsigned int gridLength, double nu, unsigned int dealWN) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;
    double value = (double)(x * x + y * y);

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

__global__ void ThirdRigthPart_kernel(const cufftDoubleComplex* a,
                                      const cufftDoubleComplex* jacobian,
                                      cufftDoubleComplex* rightPart,
                                      unsigned int gridLength, double eta,
                                      unsigned int dealWN) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;
    double value = (double)(x * x + y * y);

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
__global__ void TimeScheme_kernel(cufftDoubleComplex* field,
                                  const cufftDoubleComplex* oldField,
                                  const cufftDoubleComplex* rightPart,
                                  unsigned int gridLength, const double* dt,
                                  double weight = 1.0) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    field[idx].x = oldField[idx].x + weight * rightPart[idx].x * dt[0];
    field[idx].y = oldField[idx].y + weight * rightPart[idx].y * dt[0];
}

// Final integration stage: also stores the new value as the old field for
// the next time step, replacing a separate device-to-device copy
__global__ void TimeSchemeFinal_kernel(cufftDoubleComplex* field,
                                       cufftDoubleComplex* oldField,
                                       const cufftDoubleComplex* rightPart,
                                       unsigned int gridLength,
                                       const double* dt, double weight = 1.0) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    cufftDoubleComplex value;
    value.x = oldField[idx].x + weight * rightPart[idx].x * dt[0];
    value.y = oldField[idx].y + weight * rightPart[idx].y * dt[0];

    field[idx] = value;
    oldField[idx] = value;
}
}  // namespace mhd
