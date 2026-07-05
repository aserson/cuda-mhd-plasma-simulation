#pragma once

#include "cuda_runtime.h"

#include <iostream>

#define CUDA_CALL(result) \
    checkCudaError(result, __FUNCTION__, __FILE__, __LINE__)

inline void checkCudaError(cudaError_t result, const std::string& functionName,
                           const std::string& fileName, int lineNumber) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error in " << functionName << " at " << fileName
                  << ":" << lineNumber << " - " << cudaGetErrorString(result)
                  << std::endl;
    }
}

inline void printFunctionName(const char* functionName) {
    std::cout << "Running cuda kernel " << functionName << "..." << std::endl;
}

inline void checkCudaKernelError() {
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
}

class KernelCaller {
private:
    // Two-dimensional grid
    unsigned int _dimBlockX;
    unsigned int _dimBlockY;
    unsigned int _dimGridX;
    unsigned int _dimGridY;

    // One-dimensional grid
    unsigned int _dimBlockLinear;
    unsigned int _dimGridLinear;
    size_t _sharedSize;
    size_t _sharedSizeFloat;

    // Optional dedicated blocking stream: required for CUDA graph capture.
    // Without it kernels go to the legacy default stream, which serializes
    // with all blocking streams and synchronous cudaMemcpy calls
    cudaStream_t _stream;

public:
    KernelCaller(unsigned int gridLength, unsigned int dimBlockX,
                 unsigned int dimBlockY, unsigned int sharedLength,
                 bool ownStream = false) {
        _dimBlockX = dimBlockX;
        _dimBlockY = dimBlockY;
        _dimGridX = gridLength / dimBlockX;
        _dimGridY = gridLength / dimBlockY;

        _dimBlockLinear = sharedLength;
        _dimGridLinear = gridLength * gridLength / sharedLength;
        _sharedSize = sharedLength * sizeof(double);
        _sharedSizeFloat = sharedLength * sizeof(float);

        _stream = nullptr;
        if (ownStream) {
            CUDA_CALL(cudaStreamCreate(&_stream));
        }
    }

    KernelCaller(const KernelCaller&) = delete;
    KernelCaller& operator=(const KernelCaller&) = delete;

    ~KernelCaller() {
        if (_stream != nullptr) {
            cudaStreamDestroy(_stream);
        }
    }

    cudaStream_t stream() const { return _stream; }

    template <typename Kernel, typename... TArgs>
    void call(Kernel kernel, TArgs... args);

    template <typename Kernel, typename... TArgs>
    void callFull(Kernel kernel, TArgs... args);

    template <typename Kernel, typename... TArgs>
    void callLinear(Kernel kernel, TArgs... args);

    template <typename Kernel, typename... TArgs>
    void callLinearFloat(Kernel kernel, TArgs... args);

    template <typename Kernel, typename... TArgs>
    void callFinal(Kernel kernel, TArgs... args);

    template <typename Kernel, typename... TArgs>
    void callKernel(Kernel kernel, dim3 dimBlock, dim3 dimGrid,
                    size_t sharedSize, TArgs... args);
};

template <typename Kernel, typename... TArgs>
void KernelCaller::call(Kernel kernel, TArgs... args) {
    // Kernels index as idx = width * x + y with y taken from threadIdx.x,
    // so warps traverse the contiguous y-dimension (coalesced access):
    // grid covers gridLength / 2 columns (y) by gridLength rows (x)
    dim3 dimBlock = dim3(_dimBlockX, _dimBlockY, 1);
    dim3 dimGrid = dim3(_dimGridX / 2, _dimGridY, 1);
    size_t sharedSize = 0;

    callKernel(kernel, dimBlock, dimGrid, sharedSize, args...);
}

template <typename Kernel, typename... TArgs>
void KernelCaller::callFull(Kernel kernel, TArgs... args) {
    dim3 dimBlock = dim3(_dimBlockY, _dimBlockY, 1);
    dim3 dimGrid = dim3(_dimGridY, _dimGridY, 1);
    size_t sharedSize = 0;

    callKernel(kernel, dimBlock, dimGrid, sharedSize, args...);
}

template <typename Kernel, typename... TArgs>
void KernelCaller::callLinear(Kernel kernel, TArgs... args) {
    dim3 dimGrid = dim3(_dimGridLinear, 1, 1);
    dim3 dimBlock = dim3(_dimBlockLinear, 1, 1);
    size_t sharedSize = _sharedSize;

    callKernel(kernel, dimBlock, dimGrid, sharedSize, args...);
}

template <typename Kernel, typename... TArgs>
void KernelCaller::callLinearFloat(Kernel kernel, TArgs... args) {
    dim3 dimGrid = dim3(_dimGridLinear, 1, 1);
    dim3 dimBlock = dim3(_dimBlockLinear, 1, 1);
    size_t sharedSize = _sharedSizeFloat;

    callKernel(kernel, dimBlock, dimGrid, sharedSize, args...);
}

template <typename Kernel, typename... TArgs>
void KernelCaller::callFinal(Kernel kernel, TArgs... args) {
    dim3 dimGrid = dim3(1, 1, 1);
    dim3 dimBlock = dim3(_dimBlockLinear, 1, 1);
    size_t sharedSize = _sharedSize;

    callKernel(kernel, dimBlock, dimGrid, sharedSize, args...);
}

template <typename Kernel, typename... TArgs>
void KernelCaller::callKernel(Kernel kernel, dim3 dimBlock, dim3 dimGrid,
                              size_t sharedSize, TArgs... args) {
#ifdef __CUDACC__
    kernel<<<dimGrid, dimBlock, sharedSize, _stream>>>(args...);
    CUDA_CALL(cudaGetLastError());
#ifndef NDEBUG
    // Debug-only: surface asynchronous kernel errors at the launch site.
    // Release relies on stream ordering: the host must read results only
    // through synchronous cudaMemcpy, which waits for preceding kernels.
    CUDA_CALL(cudaDeviceSynchronize());
#endif
#endif  // __CUDACC__
}