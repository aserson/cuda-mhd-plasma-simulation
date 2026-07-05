#pragma once

#include "../Configs.h"
#include "Buffers.cuh"
#include "Helper.cuh"

namespace mhd {
class Solver : public Helper {
private:
    void calcJacobian(const GpuComplexBuffer2D& leftField,
                      const GpuComplexBuffer2D& rightField,
                      GpuComplexBuffer2D& jacobian);

public:
    Solver(const mhd::Configs& configs);

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