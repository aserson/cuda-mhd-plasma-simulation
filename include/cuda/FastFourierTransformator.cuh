#pragma once
#include <cufft.h>

#include "Buffers.cuh"

namespace mhd {
template <typename T>
class FastFourierTransformator {
public:
    using Complex = Complex_t<T>;

private:
    cufftHandle planForward, planInverse;

public:
    FastFourierTransformator(unsigned int gridLength);

    ~FastFourierTransformator();

    void setStream(cudaStream_t stream);

    void forwardFFT(T* input, Complex* output) const;
    void inverseFFT(Complex* input, T* output) const;

    void forward(GpuBuffer2D<T>& input, GpuComplexBuffer2D<T>& output) const;
    void inverse(GpuComplexBuffer2D<T>& input, GpuBuffer2D<T>& output) const;
};
}  // namespace mhd
