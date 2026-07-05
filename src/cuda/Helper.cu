#include "cuda/Helper.cuh"

#include <iostream>
#include <string>

#include "cuda/Buffers.cuh"
#include "cuda/HelperKernels.cuh"

namespace mhd {
double Helper::maxRotorAmplitude(const GpuComplexBuffer2D& field) {
    _caller.call(DiffByX_kernel, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    _caller.callLinear(Max_kernel, DoubleBufferA().data(),
                       DoubleBufferB().data());
    _caller.callFinal(MaxFinal_kernel, DoubleBufferB().data(),
                      _configs._linearLength, DoubleBufferC().data());
    CpuReducedValue().copyFromDevice(DoubleBufferC().data());
    double maxX = CpuReducedValue()[0];

    _caller.call(DiffByY_kernel, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    _caller.callLinear(Max_kernel, DoubleBufferA().data(),
                       DoubleBufferB().data());
    _caller.callFinal(MaxFinal_kernel, DoubleBufferB().data(),
                      _configs._linearLength, DoubleBufferC().data());
    CpuReducedValue().copyFromDevice(DoubleBufferC().data());
    double maxY = CpuReducedValue()[0];

    return _configs._lambda * fmax(maxX, maxY);
}

double Helper::calcEnergy(const GpuComplexBuffer2D& field) {
    _caller.call(DiffByX_kernel, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());

    _caller.call(DiffByY_kernel, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferB());

    _caller.callFull(EnergyTransform_kernel, DoubleBufferA().data(),
                     DoubleBufferB().data(), DoubleBufferC().data(),
                     _configs._gridLength, _configs._lambda);
    _caller.callLinear(EnergyIntegrate_kernel, DoubleBufferC().data(),
                       DoubleBufferA().data());
    _caller.callFinal(SumFinal_kernel, DoubleBufferA().data(),
                      _configs._linearLength, DoubleBufferB().data());

    CpuReducedValue().copyFromDevice(DoubleBufferB().data());

    return (4. * M_PI * M_PI) * _configs._lambda * CpuReducedValue()[0];
}

void Helper::normallize(GpuComplexBuffer2D& field, double ratio) {
    _caller.call(MultComplex_kernel, field.data(), field.length(), ratio,
                 field.data());
}

Helper::Helper(const Configs& configs)
    : _configs(configs),
      _transformator(configs._gridLength),
      _fields(configs._gridLength),
      _caller(configs._gridLength, configs._dimBlockX, configs._dimBlockY,
              configs._sharedLength) {}

const GpuDoubleBuffer2D& Helper::getVorticity() {
    _caller.call(MultComplex_kernel, Vorticity().data(), Vorticity().length(),
                 _configs._lambda, ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

const GpuDoubleBuffer2D& Helper::getStream() {
    _caller.call(MultComplex_kernel, Stream().data(), Stream().length(),
                 _configs._lambda, ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

const GpuDoubleBuffer2D& Helper::getCurrent() {
    _caller.call(MultComplex_kernel, Current().data(), Current().length(),
                 _configs._lambda, ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

const GpuDoubleBuffer2D& Helper::getPotential() {
    _caller.call(MultComplex_kernel, Potential().data(), Potential().length(),
                 _configs._lambda, ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

GpuComplexBuffer2D& Helper::Vorticity() {
    return _fields._vorticity;
}

GpuComplexBuffer2D& Helper::Stream() {
    return _fields._stream;
}

GpuComplexBuffer2D& Helper::Current() {
    return _fields._current;
}

GpuComplexBuffer2D& Helper::Potential() {
    return _fields._potential;
}

GpuComplexBuffer2D& Helper::OldVorticity() {
    return _fields._oldVorticity;
}

GpuComplexBuffer2D& Helper::OldPotential() {
    return _fields._oldPotential;
}

GpuComplexBuffer2D& Helper::RightPart() {
    return _fields._rightPart;
}

GpuComplexBuffer2D& Helper::ComplexBuffer() {
    return _fields._complexBuffer;
}

GpuDoubleBuffer2D& Helper::DoubleBufferA() {
    return _fields._doubleBufferA;
}

GpuDoubleBuffer2D& Helper::DoubleBufferB() {
    return _fields._doubleBufferB;
}

GpuDoubleBuffer2D& Helper::DoubleBufferC() {
    return _fields._doubleBufferC;
}

CpuDoubleBuffer1D& Helper::CpuReducedValue() {
    return _fields._cpuReducedValue;
}

CpuDoubleBuffer2D& Helper::Output() {
    return _fields._output;
}

void Helper::updateEnergies() {
    _currents.kineticEnergy = calcEnergy(Stream());
    _currents.magneticEnergy = calcEnergy(Potential());
}

void Helper::updateStream() {
    _caller.call(MinusInverseLaplasOperator_kernel, Vorticity().data(),
                 Vorticity().length(), Stream().data());
}

void Helper::updateVorticity() {
    _caller.call(MinusLaplasOperator_kernel, Stream().data(), Stream().length(),
                 Vorticity().data());
}

void Helper::updatePotential() {
    _caller.call(InverseLaplasOperator_kernel, Current().data(),
                 Current().length(), Potential().data());
}

void Helper::updateCurrent() {
    _caller.call(LaplasOperator_kernel, Potential().data(),
                 Potential().length(), Current().data());
}

void Helper::timeStep() {
    _currents.time += _currents.timeStep;
    _currents.stepNumber++;
}

void Helper::saveOldFields() {
    Vorticity().copyToDevice(OldVorticity().data());
    Potential().copyToDevice(OldPotential().data());
}

void Helper::updateTimeStep() {
    if (_currents.stepNumber % _configs._timeStepUpdateInterval != 0)
        return;

    double cfl = _configs._cfl;
    double gridStep = _configs._gridStep;
    double maxTimeStep = _configs._maxTimeStep;

    _currents.maxVelocityField = maxRotorAmplitude(Stream());
    _currents.maxMagneticField = maxRotorAmplitude(Potential());

    _currents.timeStep =
        cfl * gridStep /
        fmax(_currents.maxVelocityField, _currents.maxMagneticField);

    _currents.timeStep = fmin(_currents.timeStep, maxTimeStep);
}

void Helper::fillNormally(unsigned long seed, int offset) {
    GpuStateBuffer2D state(_configs._gridLength);
    double ratio, energy;

    _caller.call(FillStates, state.data(), state.length(), offset);
    _caller.call(FillNormally_kernel, Stream().data(), state.data(),
                 Stream().length(), _configs._averageWN);
    energy = calcEnergy(Stream());
    ratio = (energy > 0) ? std::sqrt(_configs._kineticEnergy / energy) : 1.;
    normallize(Stream(), ratio);
    updateVorticity();

    _caller.call(FillStates, state.data(), state.length(), offset + offset);
    _caller.call(FillNormally_kernel, Potential().data(), state.data(),
                 Potential().length(), _configs._averageWN);
    energy = calcEnergy(Potential());
    ratio = (energy > 0) ? std::sqrt(_configs._magneticEnergy / energy) : 1.;
    normallize(Potential(), ratio);
    updateCurrent();
}

bool Helper::shouldContinue() {
    return _currents.time <= _configs._time;
}

CudaTimeCounter::CudaTimeCounter() {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
}

CudaTimeCounter::~CudaTimeCounter() {
    cudaEventDestroy(_stop);
    cudaEventDestroy(_start);
}

void CudaTimeCounter::start() {
    cudaEventRecord(_start, 0);
}

void CudaTimeCounter::stop() {
    cudaEventRecord(_stop, 0);
    cudaEventSynchronize(_stop);

    cudaEventElapsedTime(&time, _start, _stop);
}

float CudaTimeCounter::getTime() {
    return time / 1000;
}
}  // namespace mhd