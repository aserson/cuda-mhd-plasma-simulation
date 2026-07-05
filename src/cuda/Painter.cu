#include "cuda/Painter.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "cuda/HelperKernels.cuh"

namespace graphics {
static const unsigned int colorMapLength = 256;
__constant__ unsigned char colorMap[colorMapLength * 3];

template <typename T>
__global__ static void Max_kernel(const T* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;

    extern __shared__ float sharedBuffer[];
    sharedBuffer[tidx] = fabs(static_cast<float>(input[idx]));

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

template <typename T>
__global__ void DoubleToPixels_kernel(unsigned char* output, const T* input,
                                      unsigned int gridLength,
                                      float amplitude) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = gridLength * x + y;

    float value = static_cast<float>(input[idx]);
    value = (amplitude + value) / (2 * amplitude);
    uint8_t point = static_cast<uint8_t>(value * (colorMapLength - 1));

    output[3 * idx + 0] = colorMap[3 * point + 0];
    output[3 * idx + 1] = colorMap[3 * point + 1];
    output[3 * idx + 2] = colorMap[3 * point + 2];
}

Painter::Painter(const mhd::Configs& configs,
                 const std::filesystem::path& resPath)
    : _length(configs._gridLength),
      _cpuPixels(configs._gridLength),
      _gpuPixels(configs._gridLength),
      _caller(configs._gridLength, configs._dimBlockX, configs._dimBlockY,
              configs._sharedLength),
      _cpuAmplitude(1),
      _gpuFloat(configs._linearLength) {

    if (readColorMap(configs._colorMap, resPath))
        CUDA_CALL(
            cudaMemcpyToSymbol(colorMap, _colorMap.data(), _colorMap.size()));
}

bool Painter::readColorMap(const std::string& colorMapName,
                           const std::filesystem::path& resPath) {
    std::filesystem::path filePath = resPath / "colormaps" / colorMapName;
    if (!exists(filePath)) {
        std::cout << "Color Map by name " << colorMapName << " not exists"
                  << std::endl;
        return false;
    }

    float red, green, blue;
    unsigned int i = 0;

    std::ifstream file(filePath);
    if (file.is_open()) {
        while ((file >> red >> green >> blue) && (i < _colorMap.length())) {
            _colorMap.red(i) = (unsigned char)(float(255) * red);
            _colorMap.green(i) = (unsigned char)(float(255) * green);
            _colorMap.blue(i) = (unsigned char)(float(255) * blue);
            i++;
        }
    } else {
        std::cout << "File " << filePath.filename() << " can not be opened "
                  << std::endl;
        return false;
    }

    return true;
}

template <typename T>
float Painter::findAmplitude(const mhd::GpuBuffer2D<T>& src) {
    _caller.callLinearFloat(Max_kernel<T>, src.data(), _gpuFloat.data());
    // Writing the result over the partials is safe: the single block reads
    // all of them before thread 0 stores element 0
    _caller.callFinal(mhd::MaxFinal_kernel<float>, _gpuFloat.data(),
                      _gpuFloat.length(), _gpuFloat.data());
    _cpuAmplitude.copyFromDevice(_gpuFloat.data());

    return _cpuAmplitude[0];
}

template <typename T>
void Painter::doubleToPixels(const mhd::GpuBuffer2D<T>& src) {
    float amplitude = findAmplitude(src);

    _caller.callFull(DoubleToPixels_kernel<T>, _gpuPixels.data(), src.data(),
                     src.length(), amplitude);

    _cpuPixels.copyFromDevice(_gpuPixels.data());
}

// The solver runs in either double or float precision
template void Painter::doubleToPixels<double>(
    const mhd::GpuBuffer2D<double>& src);
template void Painter::doubleToPixels<float>(const mhd::GpuBuffer2D<float>& src);
}  // namespace graphics