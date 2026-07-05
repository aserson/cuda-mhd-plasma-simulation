#pragma once

#include <filesystem>

#include "../Configs.h"

#include "Buffers.cuh"
#include "FastFourierTransformator.cuh"
#include "KernelCaller.cuh"

namespace mhd {
// The whole solver stack is templated on the real type T (double or float);
// the host-facing interface (Currents, energies, time step) stays in double
template <typename T>
class Helper {
protected:
    // Simulation configurations
    Configs _configs;

    // Auxiliary classes
    FastFourierTransformator<T> _transformator;
    KernelCaller _caller;

    struct Fields {
        // Calculated Fields
        GpuComplexBuffer2D<T> _vorticity;
        GpuComplexBuffer2D<T> _stream;
        GpuComplexBuffer2D<T> _current;
        GpuComplexBuffer2D<T> _potential;

        // Auxiliary Fields: Equations
        GpuComplexBuffer2D<T> _oldVorticity, _oldPotential, _rightPart;

        // Auxiliary Fields: Temporary GPU
        GpuComplexBuffer2D<T> _complexBuffer, _complexBufferB;
        GpuBuffer2D<T> _doubleBufferA, _doubleBufferB, _doubleBufferC,
            _doubleBufferD, _doubleBufferE, _doubleBufferF, _doubleBufferG,
            _doubleBufferH;

        // Time step in device memory (read by the time scheme kernels,
        // so the CUDA graph does not need to be rebuilt when dt changes)
        GpuBuffer2D<T> _gpuTimeStep;

        // Auxiliary Fields: Temporary CPU
        CpuBuffer1D<T> _cpuReducedValue;

        // Output buffer: Temporary CPU
        CpuBuffer2D<T> _output;

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
              _gpuTimeStep(1),
              _cpuReducedValue(1),
              _output(gridLength) {}
    } _fields;

    void normallize(GpuComplexBuffer2D<T>& field, double ratio);
    double maxRotorAmplitude(const GpuComplexBuffer2D<T>& field);
    double calcEnergy(const GpuComplexBuffer2D<T>& field);

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

    const GpuBuffer2D<T>& getVorticity();
    const GpuBuffer2D<T>& getStream();
    const GpuBuffer2D<T>& getCurrent();
    const GpuBuffer2D<T>& getPotential();

    GpuComplexBuffer2D<T>& Vorticity();
    GpuComplexBuffer2D<T>& Stream();
    GpuComplexBuffer2D<T>& Current();
    GpuComplexBuffer2D<T>& Potential();
    GpuComplexBuffer2D<T>& OldVorticity();
    GpuComplexBuffer2D<T>& OldPotential();
    GpuComplexBuffer2D<T>& RightPart();
    GpuComplexBuffer2D<T>& ComplexBuffer();
    GpuComplexBuffer2D<T>& ComplexBufferB();
    GpuBuffer2D<T>& DoubleBufferA();
    GpuBuffer2D<T>& DoubleBufferB();
    GpuBuffer2D<T>& DoubleBufferC();
    GpuBuffer2D<T>& DoubleBufferD();
    GpuBuffer2D<T>& DoubleBufferE();
    GpuBuffer2D<T>& DoubleBufferF();
    GpuBuffer2D<T>& DoubleBufferG();
    GpuBuffer2D<T>& DoubleBufferH();
    GpuBuffer2D<T>& GpuTimeStep();
    CpuBuffer1D<T>& CpuReducedValue();
    CpuBuffer2D<T>& Output();

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
