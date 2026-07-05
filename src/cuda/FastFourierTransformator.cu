#include "cuda/FastFourierTransformator.cuh"

#include <iostream>

#define CUFFT_CALL(result) \
    checkCufftResult(result, __FUNCTION__, __FILE__, __LINE__)

void checkCufftResult(cufftResult_t result, const std::string& functionName,
                      const std::string& fileName, int lineNumber) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "CUFFT Error " << result << " in " << functionName
                  << " at " << fileName << ":" << lineNumber << std::endl;
    }
}

namespace mhd {

namespace {
template <typename T>
struct FftPlans;

template <>
struct FftPlans<double> {
    static constexpr cufftType forward = CUFFT_D2Z;
    static constexpr cufftType inverse = CUFFT_Z2D;

    static cufftResult execForward(cufftHandle plan, double* input,
                                   cufftDoubleComplex* output) {
        return cufftExecD2Z(plan, input, output);
    }
    static cufftResult execInverse(cufftHandle plan, cufftDoubleComplex* input,
                                   double* output) {
        return cufftExecZ2D(plan, input, output);
    }
};

template <>
struct FftPlans<float> {
    static constexpr cufftType forward = CUFFT_R2C;
    static constexpr cufftType inverse = CUFFT_C2R;

    static cufftResult execForward(cufftHandle plan, float* input,
                                   cufftComplex* output) {
        return cufftExecR2C(plan, input, output);
    }
    static cufftResult execInverse(cufftHandle plan, cufftComplex* input,
                                   float* output) {
        return cufftExecC2R(plan, input, output);
    }
};
}  // namespace

template <typename T>
FastFourierTransformator<T>::FastFourierTransformator(unsigned int gridLength) {
    CUFFT_CALL(
        cufftPlan2d(&planForward, gridLength, gridLength, FftPlans<T>::forward));
    CUFFT_CALL(
        cufftPlan2d(&planInverse, gridLength, gridLength, FftPlans<T>::inverse));
}

template <typename T>
FastFourierTransformator<T>::~FastFourierTransformator() {
    CUFFT_CALL(cufftDestroy(planForward));
    CUFFT_CALL(cufftDestroy(planInverse));
}

template <typename T>
void FastFourierTransformator<T>::setStream(cudaStream_t stream) {
    CUFFT_CALL(cufftSetStream(planForward, stream));
    CUFFT_CALL(cufftSetStream(planInverse, stream));
}

template <typename T>
void FastFourierTransformator<T>::forwardFFT(T* input, Complex* output) const {
    CUFFT_CALL(FftPlans<T>::execForward(planForward, input, output));
}

template <typename T>
void FastFourierTransformator<T>::inverseFFT(Complex* input, T* output) const {
    CUFFT_CALL(FftPlans<T>::execInverse(planInverse, input, output));
}

template <typename T>
void FastFourierTransformator<T>::forward(GpuBuffer2D<T>& input,
                                          GpuComplexBuffer2D<T>& output) const {
    CUFFT_CALL(
        FftPlans<T>::execForward(planForward, input.data(), output.data()));
}

template <typename T>
void FastFourierTransformator<T>::inverse(GpuComplexBuffer2D<T>& input,
                                          GpuBuffer2D<T>& output) const {
    CUFFT_CALL(
        FftPlans<T>::execInverse(planInverse, input.data(), output.data()));
}

template class FastFourierTransformator<double>;
template class FastFourierTransformator<float>;

}  // namespace mhd
