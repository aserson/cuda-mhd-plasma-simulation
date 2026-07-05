#pragma once
#include "cuda_runtime.h"

#include <cufft.h>

namespace mhd {

// Jacobian Kernels
__global__ void DealaliasingDiffByX_kernel(const cufftDoubleComplex* input,
                                           cufftDoubleComplex* output,
                                           unsigned int gridLength,
                                           unsigned int dealWN) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;

    if ((abs(x) < dealWN) && (abs(y) < dealWN)) {
        output[idx].x = -(double)x * input[idx].y;
        output[idx].y = (double)x * input[idx].x;
    } else {
        output[idx].x = 0.0;
        output[idx].y = 0.0;
    }

    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        output[idx + 1].x = 0.0;
        output[idx + 1].y = 0.0;
    }
}

__global__ void DealaliasingDiffByY_kernel(const cufftDoubleComplex* input,
                                           cufftDoubleComplex* output,
                                           unsigned int gridLength,
                                           unsigned int dealWN) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;

    if ((abs(x) < dealWN) && (abs(y) < dealWN)) {
        output[idx].x = -(double)y * input[idx].y;
        output[idx].y = (double)y * input[idx].x;
    } else {
        output[idx].x = 0.0;
        output[idx].y = 0.0;
    }

    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        output[idx + 1].x = 0.0;
        output[idx + 1].y = 0.0;
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

__global__ void Dealaliasing_kernel(cufftDoubleComplex* output,
                                    unsigned int gridLength,
                                    unsigned int dealWN) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;

    if ((abs(x) >= dealWN) || (abs(y) >= dealWN)) {
        output[idx].x = 0.0;
        output[idx].y = 0.0;
    }

    if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == blockDim.x - 1)) {
        output[idx + 1].x = 0.0;
        output[idx + 1].y = 0.0;
    }
}

// Equation Kernels
__global__ void KineticRigthPart_kernel(
    const cufftDoubleComplex* w, const cufftDoubleComplex* jacobianFirst,
    const cufftDoubleComplex* jacobianSecond, cufftDoubleComplex* rightPart,
    unsigned int gridLength, double nu) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;
    double value = (double)(x * x + y * y);

    rightPart[idx].x =
        jacobianFirst[idx].x + jacobianSecond[idx].x - nu * value * w[idx].x;
    rightPart[idx].y =
        jacobianFirst[idx].y + jacobianSecond[idx].y - nu * value * w[idx].y;
}

__global__ void ThirdRigthPart_kernel(cufftDoubleComplex* a,
                                      cufftDoubleComplex* jacobian,
                                      cufftDoubleComplex* rightPart,
                                      unsigned int gridLength, double eta) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    if (x > gridLength / 2)
        x = x - gridLength;
    double value = (double)(x * x + y * y);

    rightPart[idx].x = jacobian[idx].x - eta * value * a[idx].x;
    rightPart[idx].y = jacobian[idx].y - eta * value * a[idx].y;
}

// Time Scheme Kernels
__global__ void TimeScheme_kernel(cufftDoubleComplex* field,
                                  const cufftDoubleComplex* oldField,
                                  const cufftDoubleComplex* rightPart,
                                  unsigned int gridLength, double dt,
                                  double weight = 1.0) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    field[idx].x = oldField[idx].x + weight * rightPart[idx].x * dt;
    field[idx].y = oldField[idx].y + weight * rightPart[idx].y * dt;
}

// Final integration stage: also stores the new value as the old field for
// the next time step, replacing a separate device-to-device copy
__global__ void TimeSchemeFinal_kernel(cufftDoubleComplex* field,
                                       cufftDoubleComplex* oldField,
                                       const cufftDoubleComplex* rightPart,
                                       unsigned int gridLength, double dt,
                                       double weight = 1.0) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (gridLength / 2 + 1) * x + y;

    cufftDoubleComplex value;
    value.x = oldField[idx].x + weight * rightPart[idx].x * dt;
    value.y = oldField[idx].y + weight * rightPart[idx].y * dt;

    field[idx] = value;
    oldField[idx] = value;
}
}  // namespace mhd
