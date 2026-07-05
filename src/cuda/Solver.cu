#include "cuda/Solver.cuh"

#include "cuda/SolverKernels.cuh"

namespace mhd {

void Solver::calcDerivatives(const GpuComplexBuffer2D& field,
                             GpuDoubleBuffer2D& derivativeX,
                             GpuDoubleBuffer2D& derivativeY) {
    _caller.call(DealaliasingDiffXY_kernel, field.data(),
                 ComplexBuffer().data(), ComplexBufferB().data(),
                 _configs._gridLength, _configs._dealWN);
    _transformator.inverse(ComplexBuffer(), derivativeX);
    _transformator.inverse(ComplexBufferB(), derivativeY);
}

Solver::Solver(const mhd::Configs& configs)
    : Helper(configs),
      _graph(nullptr),
      _graphExec(nullptr),
      _graphCreated(false) {}

Solver::~Solver() {
    if (_graphCreated) {
        cudaGraphExecDestroy(_graphExec);
        cudaGraphDestroy(_graph);
    }
}

void Solver::doStep() {
    // First step
    calcKineticRigthPart();
    timeSchemeKin();

    calcMagneticRightPart();
    timeSchemeMag();

    updateStreamCurrent();

    // Second step (the final time scheme also saves the fields as the
    // previous timelayer)
    calcKineticRigthPart();
    timeSchemeKinFinal();

    calcMagneticRightPart();
    timeSchemeMagFinal();

    updateStreamCurrent();
}

void Solver::step() {
#ifdef NDEBUG
    if (!_graphCreated) {
        CUDA_CALL(cudaStreamBeginCapture(_caller.stream(),
                                         cudaStreamCaptureModeGlobal));
        doStep();
        CUDA_CALL(cudaStreamEndCapture(_caller.stream(), &_graph));
        CUDA_CALL(cudaGraphInstantiate(&_graphExec, _graph, 0));
        _graphCreated = true;
    }
    CUDA_CALL(cudaGraphLaunch(_graphExec, _caller.stream()));
#else
    // Debug builds synchronize after every kernel launch, which is not
    // allowed during stream capture, so the step runs uncaptured
    doStep();
#endif
}

void Solver::calcKineticRigthPart() {
    calcDerivatives(Stream(), DoubleBufferA(), DoubleBufferB());
    calcDerivatives(Vorticity(), DoubleBufferC(), DoubleBufferD());
    calcDerivatives(Potential(), DoubleBufferE(), DoubleBufferF());
    calcDerivatives(Current(), DoubleBufferG(), DoubleBufferH());

    // J(stream, vorticity)
    _caller.callFull(Jacobian_kernel, DoubleBufferA().data(),
                     DoubleBufferD().data(), DoubleBufferB().data(),
                     DoubleBufferC().data(), DoubleBufferC().data(),
                     _configs._gridLength, _configs._lambda);
    _transformator.forward(DoubleBufferC(), ComplexBuffer());

    // J(potential, current)
    _caller.callFull(Jacobian_kernel, DoubleBufferE().data(),
                     DoubleBufferH().data(), DoubleBufferF().data(),
                     DoubleBufferG().data(), DoubleBufferG().data(),
                     _configs._gridLength, _configs._lambda);
    _transformator.forward(DoubleBufferG(), ComplexBufferB());

    _caller.call(KineticRigthPart_kernel, Vorticity().data(),
                 ComplexBuffer().data(), ComplexBufferB().data(),
                 RightPart().data(), _configs._gridLength, _configs._nu,
                 _configs._dealWN);

    // J(stream, potential) for the magnetic right part: reuses the stream
    // and potential derivatives computed above
    _caller.callFull(Jacobian_kernel, DoubleBufferA().data(),
                     DoubleBufferF().data(), DoubleBufferB().data(),
                     DoubleBufferE().data(), DoubleBufferA().data(),
                     _configs._gridLength, _configs._lambda);
    _transformator.forward(DoubleBufferA(), ComplexBuffer());
}

void Solver::calcMagneticRightPart() {
    // The Jacobian was prepared by calcKineticRigthPart; the stream and
    // potential it was built from are unchanged since then
    _caller.call(ThirdRigthPart_kernel, Potential().data(),
                 ComplexBuffer().data(), RightPart().data(),
                 _configs._gridLength, _configs._eta, _configs._dealWN);
}

void Solver::timeSchemeKin(double weight) {
    _caller.call(TimeScheme_kernel, Vorticity().data(), OldVorticity().data(),
                 RightPart().data(), Vorticity().length(),
                 (const double*)GpuTimeStep().data(), weight);
}

void Solver::timeSchemeMag(double weight) {
    _caller.call(TimeScheme_kernel, Potential().data(), OldPotential().data(),
                 RightPart().data(), Potential().length(),
                 (const double*)GpuTimeStep().data(), weight);
}

void Solver::timeSchemeKinFinal(double weight) {
    _caller.call(TimeSchemeFinal_kernel, Vorticity().data(),
                 OldVorticity().data(), RightPart().data(),
                 Vorticity().length(), (const double*)GpuTimeStep().data(),
                 weight);
}

void Solver::timeSchemeMagFinal(double weight) {
    _caller.call(TimeSchemeFinal_kernel, Potential().data(),
                 OldPotential().data(), RightPart().data(),
                 Potential().length(), (const double*)GpuTimeStep().data(),
                 weight);
}

};  // namespace mhd