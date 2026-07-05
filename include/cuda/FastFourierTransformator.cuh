#pragma once
#include <cufft.h>

#include "Buffers.cuh"

namespace mhd {
class FastFourierTransformator {
private:
    cufftHandle planD2Z, planZ2D;

public:
    FastFourierTransformator(unsigned int gridLength);

    ~FastFourierTransformator();

    void setStream(cudaStream_t stream);

    void forwardFFT(double* input, cufftDoubleComplex* output) const;
    void inverseFFT(cufftDoubleComplex* input, double* output) const;

    void forward(GpuDoubleBuffer2D& input, GpuComplexBuffer2D& output) const;
    void inverse(GpuComplexBuffer2D& input, GpuDoubleBuffer2D& output) const;
};
}  // namespace mhd
