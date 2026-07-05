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

Solver::Solver(const mhd::Configs& configs) : Helper(configs) {}

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
                 RightPart().data(), Vorticity().length(), _currents.timeStep,
                 weight);
}

void Solver::timeSchemeMag(double weight) {
    _caller.call(TimeScheme_kernel, Potential().data(), OldPotential().data(),
                 RightPart().data(), Potential().length(), _currents.timeStep,
                 weight);
}

void Solver::timeSchemeKinFinal(double weight) {
    _caller.call(TimeSchemeFinal_kernel, Vorticity().data(),
                 OldVorticity().data(), RightPart().data(),
                 Vorticity().length(), _currents.timeStep, weight);
}

void Solver::timeSchemeMagFinal(double weight) {
    _caller.call(TimeSchemeFinal_kernel, Potential().data(),
                 OldPotential().data(), RightPart().data(),
                 Potential().length(), _currents.timeStep, weight);
}

};  // namespace mhd