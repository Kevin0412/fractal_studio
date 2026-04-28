// compute/transition_volume_avx2.hpp

#pragma once

#include "cpu_features.hpp"
#include "transition_volume.hpp"

namespace fsd::compute {

bool buildTransitionVolumeAvx2(const TransitionVolumeParams& p, McField& field);
bool buildTransitionVolumeAvx2Range(const TransitionVolumeParams& p, int N, int z_begin, int z_end, McField& field);

} // namespace fsd::compute
