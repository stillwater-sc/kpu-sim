/**
 * @file schedule_compiler.hpp
 * @brief Compiles a Schedule DSL description into a DMProgram
 */

#pragma once

#include <sw/kpu/dsl/schedule.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>

namespace sw::kpu::dsl {

/**
 * @brief Compile a Schedule into a DMProgram instruction stream.
 *
 * The compiler walks the schedule's operation tree and emits
 * DMInstructions with proper tile coordinates, addresses, and
 * buffer assignments.
 */
DMProgram compile_schedule(const Schedule& sched);

} // namespace sw::kpu::dsl
