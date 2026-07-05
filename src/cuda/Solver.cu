#include "cuda/Solver.cuh"

#include "cuda/SolverKernels.cuh"

namespace mhd {

void Solver::calcJacobian(const GpuComplexBuffer2D& leftField,
                          const GpuComplexBuffer2D& rightField,
                          GpuComplexBuffer2D& jacobian) {
    _caller.call(DealaliasingDiffByX_kernel, leftField.data(), jacobian.data(),
                 _configs._gridLength, _configs._dealWN);
    _transformator.inverse(jacobian, DoubleBufferA());

    _caller.call(DealaliasingDiffByY_kernel, rightField.data(),
                 jacobian.data(), _configs._gridLength, _configs._dealWN);
    _transformator.inverse(jacobian, DoubleBufferB());

    _caller.call(DealaliasingDiffByY_kernel, leftField.data(), jacobian.data(),
                 _configs._gridLength, _configs._dealWN);
    _transformator.inverse(jacobian, DoubleBufferC());

    _caller.call(DealaliasingDiffByX_kernel, rightField.data(),
                 jacobian.data(), _configs._gridLength, _configs._dealWN);
    _transformator.inverse(jacobian, DoubleBufferD());

    _caller.callFull(Jacobian_kernel, DoubleBufferA().data(),
                     DoubleBufferB().data(), DoubleBufferC().data(),
                     DoubleBufferD().data(), DoubleBufferA().data(),
                     _configs._gridLength, _configs._lambda);

    _transformator.forward(DoubleBufferA(), jacobian);
    _caller.call(Dealaliasing_kernel, jacobian.data(), _configs._gridLength,
                 _configs._dealWN);
}

Solver::Solver(const mhd::Configs& configs) : Helper(configs) {}

void Solver::calcKineticRigthPart() {
    calcJacobian(Stream(), Vorticity(), ComplexBuffer());
    calcJacobian(Potential(), Current(), ComplexBufferB());
    _caller.call(KineticRigthPart_kernel, Vorticity().data(),
                 ComplexBuffer().data(), ComplexBufferB().data(),
                 RightPart().data(), _configs._gridLength, _configs._nu);
}

void Solver::calcMagneticRightPart() {
    calcJacobian(Stream(), Potential(), ComplexBuffer());
    _caller.call(ThirdRigthPart_kernel, Potential().data(),
                 ComplexBuffer().data(), RightPart().data(),
                 _configs._gridLength, _configs._eta);
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