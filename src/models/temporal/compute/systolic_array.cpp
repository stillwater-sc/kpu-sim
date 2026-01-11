// systolic_array.cpp
//
// This file is intentionally minimal.
// SystolicArray is now a class template defined entirely in the header file.
// Template classes require their implementation to be visible at instantiation sites,
// so all code has been moved to systolic_array.hpp.
//
// Explicit instantiations can be added here if needed to reduce compile times:
// template class sw::kpu::SystolicArray<float>;
// template class sw::kpu::SystolicArray<double>;

#include <sw/kpu/models/temporal/compute/systolic_array.hpp>

namespace sw::kpu {

// Explicit template instantiations for common scalar types
template class SystolicArray<int8_t>;
template class SystolicArray<int32_t>;
template class SystolicArray<float>;
template class SystolicArray<double>;

} // namespace sw::kpu
