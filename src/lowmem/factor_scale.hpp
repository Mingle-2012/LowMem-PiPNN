#pragma once

// Coordinates, PiPNN, topology and one-bit codes are never rescaled.
// In the rotated domain, with M=max ||x||_2 and p padded dimensions:
// |F_add| <= 4 M^2 (1+sqrt(p)), |F_rescale| <= 8 M for L2.
// A factor-two margin covers floating-point rotation/norm evaluation.
static float primary_choose_factor_scale(const ResidentInput& input) {
    double maximum_norm2 = 0;
    size_t nonfinite = 0;
#pragma omp parallel for num_threads(lowmem_threads()) reduction(max:maximum_norm2) reduction(+:nonfinite) schedule(static)
    for (size_t u = 0; u < input.n; ++u) {
        double norm2 = 0;
        for (size_t j = 0; j < input.d; ++j) {
            double value = input.data.data()[u * input.d + j];
            nonfinite += !std::isfinite(value);
            norm2 += value * value;
        }
        maximum_norm2 = std::max(maximum_norm2, norm2);
    }
    if (nonfinite || !std::isfinite(maximum_norm2)) throw std::runtime_error("nonfinite input");
    const size_t padded = ((input.d + 63) / 64) * 64;
    double bound = std::max(8 * maximum_norm2 * (1 + std::sqrt(double(padded))),
                            16 * std::sqrt(maximum_norm2));
    float scale = 1;
    while (bound / scale > 60000) {
        scale *= 2;
        if (!std::isfinite(scale)) throw std::runtime_error("factor scale outside FP32 range");
    }
    std::cout << "FACTOR_SCALE scale=" << scale << " max_input_norm2=" << maximum_norm2
              << " conservative_factor_bound=" << bound
              << " input_topology_codes_unchanged=1 per_edge_extra_bytes=0 global_scale_bytes=4\n" << std::flush;
    return scale;
}
