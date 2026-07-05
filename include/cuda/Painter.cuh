#pragma once

#include <string>

#include "../Configs.h"

#include "Buffers.cuh"
#include "KernelCaller.cuh"

namespace graphics {

class Painter {
private:
    KernelCaller _caller;

    unsigned int _length;

    CpuColorMapBuffer _colorMap;

    CpuPixelBuffer2D _cpuPixels;
    GpuPixelBuffer2D _gpuPixels;

    // The amplitude reduction finishes on the GPU; only the final value
    // is copied to the host
    CpuFloatBuffer _cpuAmplitude;
    GpuFloatBuffer _gpuFloat;

    bool readColorMap(const std::string& colorMapName,
                      const std::filesystem::path& resPath);
    template <typename T>
    float findAmplitude(const mhd::GpuBuffer2D<T>& src);

public:
    Painter(const mhd::Configs& configs, const std::filesystem::path& resPath);

    template <typename T>
    void doubleToPixels(const mhd::GpuBuffer2D<T>& src);

    const CpuPixelBuffer2D& getPixels() const { return _cpuPixels; }
    unsigned int getLength() const { return _length; }
};
}  // namespace graphics
