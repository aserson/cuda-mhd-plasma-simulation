#pragma once

#include "../Configs.h"
#include "Buffers.cuh"
#include "Helper.cuh"

namespace mhd {
template <typename T>
class Solver : public Helper<T> {
private:
    // Two-step time integration scheme captured into a CUDA graph
    cudaGraph_t _graph;
    cudaGraphExec_t _graphExec;
    bool _graphCreated;

    void calcDerivatives(const GpuComplexBuffer2D<T>& field,
                         GpuBuffer2D<T>& derivativeX,
                         GpuBuffer2D<T>& derivativeY);

    void doStep();

public:
    Solver(const mhd::Configs& configs);
    ~Solver();

    // Performs one full step of the two-step scheme; the kernel and FFT
    // sequence is captured into a CUDA graph on the first call and
    // launched as a single graph afterwards
    void step();

    // Computes the kinetic right part and prepares the Jacobian for the
    // magnetic right part from the same derivative fields, so it must be
    // called before calcMagneticRightPart
    void calcKineticRigthPart();
    void calcMagneticRightPart();

    void timeSchemeKin(double weight = 1.0);
    void timeSchemeMag(double weight = 1.0);

    // Final integration stage: also stores the result as the old field,
    // so a separate saveOldFields call is not needed
    void timeSchemeKinFinal(double weight = 1.0);
    void timeSchemeMagFinal(double weight = 1.0);
};
}  // namespace mhd
