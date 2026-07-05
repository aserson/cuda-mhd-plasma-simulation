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
FastFourierTransformator::FastFourierTransformator(unsigned int gridLength) {
    CUFFT_CALL(cufftPlan2d(&planD2Z, gridLength, gridLength, CUFFT_D2Z));
    CUFFT_CALL(cufftPlan2d(&planZ2D, gridLength, gridLength, CUFFT_Z2D));
}

FastFourierTransformator::~FastFourierTransformator() {
    CUFFT_CALL(cufftDestroy(planD2Z));
    CUFFT_CALL(cufftDestroy(planZ2D));
}

void FastFourierTransformator::setStream(cudaStream_t stream) {
    CUFFT_CALL(cufftSetStream(planD2Z, stream));
    CUFFT_CALL(cufftSetStream(planZ2D, stream));
}

void FastFourierTransformator::forwardFFT(double* input,
                                          cufftDoubleComplex* output) const {
    CUFFT_CALL(cufftExecD2Z(planD2Z, input, output));
}

void FastFourierTransformator::inverseFFT(cufftDoubleComplex* input,
                                          double* output) const {
    CUFFT_CALL(cufftExecZ2D(planZ2D, input, output));
}

void FastFourierTransformator::forward(GpuDoubleBuffer2D& input,
                                       GpuComplexBuffer2D& output) const {
    CUFFT_CALL(cufftExecD2Z(planD2Z, input.data(), output.data()));
}

void FastFourierTransformator::inverse(GpuComplexBuffer2D& input,
                                       GpuDoubleBuffer2D& output) const {
    CUFFT_CALL(cufftExecZ2D(planZ2D, input.data(), output.data()));
}

}  // namespace mhd
