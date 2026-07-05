#include "cuda/Helper.cuh"

#include <iostream>
#include <string>

#include "cuda/Buffers.cuh"
#include "cuda/HelperKernels.cuh"

namespace mhd {
template <typename T>
double Helper<T>::maxRotorAmplitude(const GpuComplexBuffer2D<T>& field) {
    _caller.call(DiffByX_kernel<T>, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    _caller.callLinear(Max_kernel<T>, DoubleBufferA().data(),
                       DoubleBufferB().data());
    _caller.callFinal(MaxFinal_kernel<T>, DoubleBufferB().data(),
                      _configs._linearLength, DoubleBufferC().data());
    CpuReducedValue().copyFromDevice(DoubleBufferC().data());
    double maxX = (double)CpuReducedValue()[0];

    _caller.call(DiffByY_kernel<T>, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    _caller.callLinear(Max_kernel<T>, DoubleBufferA().data(),
                       DoubleBufferB().data());
    _caller.callFinal(MaxFinal_kernel<T>, DoubleBufferB().data(),
                      _configs._linearLength, DoubleBufferC().data());
    CpuReducedValue().copyFromDevice(DoubleBufferC().data());
    double maxY = (double)CpuReducedValue()[0];

    return _configs._lambda * fmax(maxX, maxY);
}

template <typename T>
double Helper<T>::calcEnergy(const GpuComplexBuffer2D<T>& field) {
    _caller.call(DiffByX_kernel<T>, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());

    _caller.call(DiffByY_kernel<T>, field.data(), _configs._gridLength,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferB());

    _caller.callFull(EnergyTransform_kernel<T>, DoubleBufferA().data(),
                     DoubleBufferB().data(), DoubleBufferC().data(),
                     _configs._gridLength, (T)_configs._lambda);
    _caller.callLinear(EnergyIntegrate_kernel<T>, DoubleBufferC().data(),
                       DoubleBufferA().data());
    _caller.callFinal(SumFinal_kernel<T>, DoubleBufferA().data(),
                      _configs._linearLength, DoubleBufferB().data());

    CpuReducedValue().copyFromDevice(DoubleBufferB().data());

    return (4. * M_PI * M_PI) * _configs._lambda *
           (double)CpuReducedValue()[0];
}

template <typename T>
void Helper<T>::normallize(GpuComplexBuffer2D<T>& field, double ratio) {
    _caller.call(MultComplex_kernel<T>, field.data(), field.length(), (T)ratio,
                 field.data());
}

template <typename T>
Helper<T>::Helper(const Configs& configs)
    : _configs(configs),
      _transformator(configs._gridLength),
      _fields(configs._gridLength),
      _caller(configs._gridLength, configs._dimBlockX, configs._dimBlockY,
              configs._sharedLength, true) {
    _transformator.setStream(_caller.stream());
}

template <typename T>
const GpuBuffer2D<T>& Helper<T>::getVorticity() {
    _caller.call(MultComplex_kernel<T>, Vorticity().data(),
                 Vorticity().length(), (T)_configs._lambda,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

template <typename T>
const GpuBuffer2D<T>& Helper<T>::getStream() {
    _caller.call(MultComplex_kernel<T>, Stream().data(), Stream().length(),
                 (T)_configs._lambda, ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

template <typename T>
const GpuBuffer2D<T>& Helper<T>::getCurrent() {
    _caller.call(MultComplex_kernel<T>, Current().data(), Current().length(),
                 (T)_configs._lambda, ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

template <typename T>
const GpuBuffer2D<T>& Helper<T>::getPotential() {
    _caller.call(MultComplex_kernel<T>, Potential().data(),
                 Potential().length(), (T)_configs._lambda,
                 ComplexBuffer().data());
    _transformator.inverse(ComplexBuffer(), DoubleBufferA());
    return DoubleBufferA();
}

template <typename T>
GpuComplexBuffer2D<T>& Helper<T>::Vorticity() {
    return _fields._vorticity;
}

template <typename T>
GpuComplexBuffer2D<T>& Helper<T>::Stream() {
    return _fields._stream;
}

template <typename T>
GpuComplexBuffer2D<T>& Helper<T>::Current() {
    return _fields._current;
}

template <typename T>
GpuComplexBuffer2D<T>& Helper<T>::Potential() {
    return _fields._potential;
}

template <typename T>
GpuComplexBuffer2D<T>& Helper<T>::OldVorticity() {
    return _fields._oldVorticity;
}

template <typename T>
GpuComplexBuffer2D<T>& Helper<T>::OldPotential() {
    return _fields._oldPotential;
}

template <typename T>
GpuComplexBuffer2D<T>& Helper<T>::RightPart() {
    return _fields._rightPart;
}

template <typename T>
GpuComplexBuffer2D<T>& Helper<T>::ComplexBuffer() {
    return _fields._complexBuffer;
}

template <typename T>
GpuComplexBuffer2D<T>& Helper<T>::ComplexBufferB() {
    return _fields._complexBufferB;
}

template <typename T>
GpuBuffer2D<T>& Helper<T>::DoubleBufferA() {
    return _fields._doubleBufferA;
}

template <typename T>
GpuBuffer2D<T>& Helper<T>::DoubleBufferB() {
    return _fields._doubleBufferB;
}

template <typename T>
GpuBuffer2D<T>& Helper<T>::DoubleBufferC() {
    return _fields._doubleBufferC;
}

template <typename T>
GpuBuffer2D<T>& Helper<T>::DoubleBufferD() {
    return _fields._doubleBufferD;
}

template <typename T>
GpuBuffer2D<T>& Helper<T>::DoubleBufferE() {
    return _fields._doubleBufferE;
}

template <typename T>
GpuBuffer2D<T>& Helper<T>::DoubleBufferF() {
    return _fields._doubleBufferF;
}

template <typename T>
GpuBuffer2D<T>& Helper<T>::DoubleBufferG() {
    return _fields._doubleBufferG;
}

template <typename T>
GpuBuffer2D<T>& Helper<T>::DoubleBufferH() {
    return _fields._doubleBufferH;
}

template <typename T>
GpuBuffer2D<T>& Helper<T>::GpuTimeStep() {
    return _fields._gpuTimeStep;
}

template <typename T>
CpuBuffer1D<T>& Helper<T>::CpuReducedValue() {
    return _fields._cpuReducedValue;
}

template <typename T>
CpuBuffer2D<T>& Helper<T>::Output() {
    return _fields._output;
}

template <typename T>
void Helper<T>::updateEnergies() {
    _currents.kineticEnergy = calcEnergy(Stream());
    _currents.magneticEnergy = calcEnergy(Potential());
}

template <typename T>
void Helper<T>::updateStream() {
    _caller.call(MinusInverseLaplasOperator_kernel<T>, Vorticity().data(),
                 Vorticity().length(), Stream().data());
}

template <typename T>
void Helper<T>::updateVorticity() {
    _caller.call(MinusLaplasOperator_kernel<T>, Stream().data(),
                 Stream().length(), Vorticity().data());
}

template <typename T>
void Helper<T>::updatePotential() {
    _caller.call(InverseLaplasOperator_kernel<T>, Current().data(),
                 Current().length(), Potential().data());
}

template <typename T>
void Helper<T>::updateCurrent() {
    _caller.call(LaplasOperator_kernel<T>, Potential().data(),
                 Potential().length(), Current().data());
}

template <typename T>
void Helper<T>::updateStreamCurrent() {
    _caller.call(StreamCurrent_kernel<T>, Vorticity().data(),
                 Potential().data(), Stream().data(), Current().data(),
                 Vorticity().length());
}

template <typename T>
void Helper<T>::timeStep() {
    _currents.time += _currents.timeStep;
    _currents.stepNumber++;
}

template <typename T>
void Helper<T>::saveOldFields() {
    Vorticity().copyToDevice(OldVorticity().data());
    Potential().copyToDevice(OldPotential().data());
}

template <typename T>
void Helper<T>::updateTimeStep() {
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

    T deviceTimeStep = (T)_currents.timeStep;
    GpuTimeStep().copyFromHost(&deviceTimeStep);
}

template <typename T>
void Helper<T>::fillNormally(unsigned long seed, int offset) {
    GpuStateBuffer2D state(_configs._gridLength);
    double ratio, energy;

    _caller.call(FillStates, state.data(), state.length(), offset);
    _caller.call(FillNormally_kernel<T>, Stream().data(), state.data(),
                 Stream().length(), _configs._averageWN);
    energy = calcEnergy(Stream());
    ratio = (energy > 0) ? std::sqrt(_configs._kineticEnergy / energy) : 1.;
    normallize(Stream(), ratio);
    updateVorticity();

    _caller.call(FillStates, state.data(), state.length(), offset + offset);
    _caller.call(FillNormally_kernel<T>, Potential().data(), state.data(),
                 Potential().length(), _configs._averageWN);
    energy = calcEnergy(Potential());
    ratio = (energy > 0) ? std::sqrt(_configs._magneticEnergy / energy) : 1.;
    normallize(Potential(), ratio);
    updateCurrent();
}

template <typename T>
bool Helper<T>::shouldContinue() {
    return _currents.time <= _configs._time;
}

// The solver runs in either double or float precision
template class Helper<double>;
template class Helper<float>;

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
