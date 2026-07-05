#pragma once

#include <filesystem>

#include "../Configs.h"

#include "Buffers.cuh"
#include "FastFourierTransformator.cuh"
#include "KernelCaller.cuh"

namespace mhd {
class Helper {
protected:
    // Simulation configurations
    Configs _configs;

    // Auxiliary classes
    FastFourierTransformator _transformator;
    KernelCaller _caller;

    struct Fields {
        // Calculated Fields
        GpuComplexBuffer2D _vorticity;
        GpuComplexBuffer2D _stream;
        GpuComplexBuffer2D _current;
        GpuComplexBuffer2D _potential;

        // Auxiliary Fields: Equations
        GpuComplexBuffer2D _oldVorticity, _oldPotential, _rightPart;

        // Auxiliary Fields: Temporary GPU
        GpuComplexBuffer2D _complexBuffer, _complexBufferB;
        GpuDoubleBuffer2D _doubleBufferA, _doubleBufferB, _doubleBufferC,
            _doubleBufferD, _doubleBufferE, _doubleBufferF, _doubleBufferG,
            _doubleBufferH;

        // Auxiliary Fields: Temporary CPU
        CpuDoubleBuffer1D _cpuReducedValue;

        // Output buffer: Temporary CPU
        CpuDoubleBuffer2D _output;

        Fields(unsigned int gridLength)
            : _vorticity(gridLength),
              _stream(gridLength),
              _current(gridLength),
              _potential(gridLength),
              _oldVorticity(gridLength),
              _oldPotential(gridLength),
              _rightPart(gridLength),
              _complexBuffer(gridLength),
              _complexBufferB(gridLength),
              _doubleBufferA(gridLength),
              _doubleBufferB(gridLength),
              _doubleBufferC(gridLength),
              _doubleBufferD(gridLength),
              _doubleBufferE(gridLength),
              _doubleBufferF(gridLength),
              _doubleBufferG(gridLength),
              _doubleBufferH(gridLength),
              _cpuReducedValue(1),
              _output(gridLength) {}
    } _fields;

    void normallize(GpuComplexBuffer2D& field, double ratio);
    double maxRotorAmplitude(const GpuComplexBuffer2D& field);
    double calcEnergy(const GpuComplexBuffer2D& field);

public:
    Currents _currents;

    Helper(const Configs& configs);

    void fillNormally(unsigned long seed, int offset = 1);

    void updateStream();
    void updateVorticity();
    void updatePotential();
    void updateCurrent();
    void updateStreamCurrent();

    void timeStep();
    void updateTimeStep();
    void updateEnergies();

    void saveOldFields();

    const GpuDoubleBuffer2D& getVorticity();
    const GpuDoubleBuffer2D& getStream();
    const GpuDoubleBuffer2D& getCurrent();
    const GpuDoubleBuffer2D& getPotential();

    GpuComplexBuffer2D& Vorticity();
    GpuComplexBuffer2D& Stream();
    GpuComplexBuffer2D& Current();
    GpuComplexBuffer2D& Potential();
    GpuComplexBuffer2D& OldVorticity();
    GpuComplexBuffer2D& OldPotential();
    GpuComplexBuffer2D& RightPart();
    GpuComplexBuffer2D& ComplexBuffer();
    GpuComplexBuffer2D& ComplexBufferB();
    GpuDoubleBuffer2D& DoubleBufferA();
    GpuDoubleBuffer2D& DoubleBufferB();
    GpuDoubleBuffer2D& DoubleBufferC();
    GpuDoubleBuffer2D& DoubleBufferD();
    GpuDoubleBuffer2D& DoubleBufferE();
    GpuDoubleBuffer2D& DoubleBufferF();
    GpuDoubleBuffer2D& DoubleBufferG();
    GpuDoubleBuffer2D& DoubleBufferH();
    CpuDoubleBuffer1D& CpuReducedValue();
    CpuDoubleBuffer2D& Output();

    bool shouldContinue();
};

class CudaTimeCounter {
private:
    cudaEvent_t _start, _stop;
    float time;

public:
    CudaTimeCounter();
    ~CudaTimeCounter();

    void start();
    void stop();

    float getTime();
};
}  // namespace mhd
