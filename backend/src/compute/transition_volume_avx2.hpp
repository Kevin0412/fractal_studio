// compute/transition_volume_avx2.hpp

#pragma once

#include "cpu_features.hpp"
#include "transition_volume.hpp"

namespace fsd::compute {

bool buildTransitionVolumeAvx2(const TransitionVolumeParams& p, McField& field);

} // namespace fsd::compute
