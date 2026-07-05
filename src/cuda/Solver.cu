#include "cuda/Solver.cuh"

#include "cuda/SolverKernels.cuh"

namespace mhd {

// Members of the dependent base Helper<T> are accessed through this->
template <typename T>
void Solver<T>::calcDerivatives(const GpuComplexBuffer2D<T>& field,
                                GpuBuffer2D<T>& derivativeX,
                                GpuBuffer2D<T>& derivativeY) {
    this->_caller.call(DealaliasingDiffXY_kernel<T>, field.data(),
                       this->ComplexBuffer().data(),
                       this->ComplexBufferB().data(),
                       this->_configs._gridLength, this->_configs._dealWN);
    this->_transformator.inverse(this->ComplexBuffer(), derivativeX);
    this->_transformator.inverse(this->ComplexBufferB(), derivativeY);
}

template <typename T>
Solver<T>::Solver(const mhd::Configs& configs)
    : Helper<T>(configs),
      _graph(nullptr),
      _graphExec(nullptr),
      _graphCreated(false) {}

template <typename T>
Solver<T>::~Solver() {
    if (_graphCreated) {
        cudaGraphExecDestroy(_graphExec);
        cudaGraphDestroy(_graph);
    }
}

template <typename T>
void Solver<T>::doStep() {
    // First step
    calcKineticRigthPart();
    timeSchemeKin();

    calcMagneticRightPart();
    timeSchemeMag();

    this->updateStreamCurrent();

    // Second step (the final time scheme also saves the fields as the
    // previous timelayer)
    calcKineticRigthPart();
    timeSchemeKinFinal();

    calcMagneticRightPart();
    timeSchemeMagFinal();

    this->updateStreamCurrent();
}

template <typename T>
void Solver<T>::step() {
#ifdef NDEBUG
    if (!_graphCreated) {
        CUDA_CALL(cudaStreamBeginCapture(this->_caller.stream(),
                                         cudaStreamCaptureModeGlobal));
        doStep();
        CUDA_CALL(cudaStreamEndCapture(this->_caller.stream(), &_graph));
        CUDA_CALL(cudaGraphInstantiate(&_graphExec, _graph, 0));
        _graphCreated = true;
    }
    CUDA_CALL(cudaGraphLaunch(_graphExec, this->_caller.stream()));
#else
    // Debug builds synchronize after every kernel launch, which is not
    // allowed during stream capture, so the step runs uncaptured
    doStep();
#endif
}

template <typename T>
void Solver<T>::calcKineticRigthPart() {
    calcDerivatives(this->Stream(), this->DoubleBufferA(),
                    this->DoubleBufferB());
    calcDerivatives(this->Vorticity(), this->DoubleBufferC(),
                    this->DoubleBufferD());
    calcDerivatives(this->Potential(), this->DoubleBufferE(),
                    this->DoubleBufferF());
    calcDerivatives(this->Current(), this->DoubleBufferG(),
                    this->DoubleBufferH());

    // J(stream, vorticity)
    this->_caller.callFull(Jacobian_kernel<T>, this->DoubleBufferA().data(),
                           this->DoubleBufferD().data(),
                           this->DoubleBufferB().data(),
                           this->DoubleBufferC().data(),
                           this->DoubleBufferC().data(),
                           this->_configs._gridLength,
                           (T)this->_configs._lambda);
    this->_transformator.forward(this->DoubleBufferC(), this->ComplexBuffer());

    // J(potential, current)
    this->_caller.callFull(Jacobian_kernel<T>, this->DoubleBufferE().data(),
                           this->DoubleBufferH().data(),
                           this->DoubleBufferF().data(),
                           this->DoubleBufferG().data(),
                           this->DoubleBufferG().data(),
                           this->_configs._gridLength,
                           (T)this->_configs._lambda);
    this->_transformator.forward(this->DoubleBufferG(),
                                 this->ComplexBufferB());

    this->_caller.call(KineticRigthPart_kernel<T>, this->Vorticity().data(),
                       this->ComplexBuffer().data(),
                       this->ComplexBufferB().data(), this->RightPart().data(),
                       this->_configs._gridLength, (T)this->_configs._nu,
                       this->_configs._dealWN);

    // J(stream, potential) for the magnetic right part: reuses the stream
    // and potential derivatives computed above
    this->_caller.callFull(Jacobian_kernel<T>, this->DoubleBufferA().data(),
                           this->DoubleBufferF().data(),
                           this->DoubleBufferB().data(),
                           this->DoubleBufferE().data(),
                           this->DoubleBufferA().data(),
                           this->_configs._gridLength,
                           (T)this->_configs._lambda);
    this->_transformator.forward(this->DoubleBufferA(), this->ComplexBuffer());
}

template <typename T>
void Solver<T>::calcMagneticRightPart() {
    // The Jacobian was prepared by calcKineticRigthPart; the stream and
    // potential it was built from are unchanged since then
    this->_caller.call(ThirdRigthPart_kernel<T>, this->Potential().data(),
                       this->ComplexBuffer().data(), this->RightPart().data(),
                       this->_configs._gridLength, (T)this->_configs._eta,
                       this->_configs._dealWN);
}

template <typename T>
void Solver<T>::timeSchemeKin(double weight) {
    this->_caller.call(TimeScheme_kernel<T>, this->Vorticity().data(),
                       this->OldVorticity().data(), this->RightPart().data(),
                       this->Vorticity().length(),
                       (const T*)this->GpuTimeStep().data(), (T)weight);
}

template <typename T>
void Solver<T>::timeSchemeMag(double weight) {
    this->_caller.call(TimeScheme_kernel<T>, this->Potential().data(),
                       this->OldPotential().data(), this->RightPart().data(),
                       this->Potential().length(),
                       (const T*)this->GpuTimeStep().data(), (T)weight);
}

template <typename T>
void Solver<T>::timeSchemeKinFinal(double weight) {
    this->_caller.call(TimeSchemeFinal_kernel<T>, this->Vorticity().data(),
                       this->OldVorticity().data(), this->RightPart().data(),
                       this->Vorticity().length(),
                       (const T*)this->GpuTimeStep().data(), (T)weight);
}

template <typename T>
void Solver<T>::timeSchemeMagFinal(double weight) {
    this->_caller.call(TimeSchemeFinal_kernel<T>, this->Potential().data(),
                       this->OldPotential().data(), this->RightPart().data(),
                       this->Potential().length(),
                       (const T*)this->GpuTimeStep().data(), (T)weight);
}

// The solver runs in either double or float precision
template class Solver<double>;
template class Solver<float>;

};  // namespace mhd
