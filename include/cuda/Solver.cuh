#pragma once

#include "../Configs.h"
#include "Buffers.cuh"
#include "Helper.cuh"

namespace mhd {
class Solver : public Helper {
private:
    void calcDerivatives(const GpuComplexBuffer2D& field,
                         GpuDoubleBuffer2D& derivativeX,
                         GpuDoubleBuffer2D& derivativeY);

public:
    Solver(const mhd::Configs& configs);

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