#pragma once

#include <filesystem>
#include <string>

#include "Configs.h"

#include "cuda/Buffers.cuh"
#include "cuda/Helper.cuh"
#include "cuda/Painter.cuh"
#include "openGL/Creater.h"

namespace mhd {
class Writer {
private:
    // Fields are written to disk as double regardless of the solver
    // precision; the float staging buffer holds the device data before
    // conversion
    CpuDoubleBuffer2D _output;
    CpuBuffer2D<float> _outputFloat;
    graphics::Painter _painter;

    template <typename T>
    void save(const T* field, const std::filesystem::path& filePath);
    void clear();

    const std::filesystem::path _outputPath;

    double _outputTime;
    double _outputStep;
    double _outputStop;
    unsigned int _outputNumber;

    struct Settings {
        bool saveData;
        bool savePNG;
        bool saveVorticity;
        bool saveCurrent;
        bool saveStream;
        bool savePotential;
        bool showGraphics;
    } _settings;

public:
    Writer(const std::filesystem::path& outputPath, const mhd::Configs& configs,
           const std::filesystem::path& resPath);

    // Saves the due data files, renders the pixels of the displayed field
    // and prints the currents; returns true when a new frame was painted.
    // The pixels are available through getPixels afterwards
    template <typename T>
    bool saveData(mhd::Helper<T>& helper);

    // Same, but also uploads the painted frame as a texture of the
    // OpenGL window
    template <typename T>
    bool saveData(mhd::Helper<T>& helper, opengl::Creater& creater);

    const graphics::CpuPixelBuffer2D& getPixels() const {
        return _painter.getPixels();
    }
    unsigned int getPixelsLength() const { return _painter.getLength(); }

    void saveCurrents(const Currents& currents,
                      const std::filesystem::path& filePath);

    void printCurrents(const mhd::Currents& currents);

    bool shouldWrite(double time) {
        return ((time >= _outputTime) && (time <= _outputStop));
    }

    bool shouldPaint(double time) {
        return time >= _outputTime;
    }

    void step() {
        _outputNumber++;
        _outputTime += _outputStep;
    }

    static std::string uintToStr(unsigned int value);
};
}  // namespace mhd