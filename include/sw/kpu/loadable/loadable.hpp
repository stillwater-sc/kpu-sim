// ============================================================================
// include/sw/kpu/loadable/loadable.hpp
// The KPU loadable (.kpuld) — the unit of deployment (#305).
//
// DECLARATION-ONLY, like platform/deployment_json.hpp and for the same reason: the
// definitions live in src/program/loadable.cpp, so nothing outside that translation
// unit sees <flatbuffers/...>. `schemas/kpu_loadable.fbs` is the source of truth and
// its C++ header is generated at build time into a PRIVATE directory.
//
// TWO CONSUMERS, AND ONLY ONE OF THEM USES THIS HEADER. Worth stating up front, or the
// zero-copy claim in the plan reads as fiction:
//
//   * HOST tooling — the writer, the tests, the reference orchestrator's outer loop —
//     uses the materialized `Loadable` below. It allocates, and that is fine: a host
//     has a heap.
//   * THE RV ORCHESTRATOR (increment 4) does NOT use this header. It includes the
//     generated FlatBuffers accessors and reads its operator and tensor tables IN
//     PLACE, out of the mapped file, with no parse step and no heap. That is the
//     property the container was chosen for.
//
// So this header is the convenient face of the format, not its only one, and a change
// here is not automatically a change to what the orchestrator sees.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Forward-declared rather than included: a capability check is inherently about a
// deployment, but this header should not hand every consumer of a loadable the whole
// platform layer.
namespace sw::kpu::program::platform { struct DeploymentSpec; }

namespace sw::kpu::loadable {

// ----------------------------------------------------------------------------
// Versions (R1/R3/R4)
// ----------------------------------------------------------------------------
struct Version {
    std::uint16_t major = 0, minor = 0, patch = 0;

    std::string str() const {
        return std::to_string(major) + "." + std::to_string(minor) + "." +
               std::to_string(patch);
    }
    bool operator==(const Version& o) const {
        return major == o.major && minor == o.minor && patch == o.patch;
    }
    bool operator<(const Version& o) const {
        if (major != o.major) return major < o.major;
        if (minor != o.minor) return minor < o.minor;
        return patch < o.patch;
    }
    bool operator<=(const Version& o) const { return *this < o || *this == o; }
};

// The container's structure, the op/operator surface, and what THIS build can read.
Version format_version();
Version reader_version();
Version producer_version();

// The four-character FlatBuffers file identifier — the magic, supplied by the format
// itself rather than hand-rolled (R1).
const char* file_identifier();

// ----------------------------------------------------------------------------
// Refusals carry a CAUSE, so a caller can tell "not for me" from "broken" without
// parsing a message — the distinction #265 found worth making and then needed.
// ----------------------------------------------------------------------------
class LoadError : public std::runtime_error {
public:
    enum class Cause {
        NotALoadable,        // wrong or missing file identifier
        Truncated,           // buffer too short, or the verifier rejected it
        MalformedContainer,  // structurally valid FlatBuffers, invalid as a loadable
        UnsupportedVersion,  // the file demands a newer reader (R4)
        MissingRequiredField,
        InconsistentRecord,  // the file contradicts itself
        CapabilityMismatch,  // the machine cannot satisfy the profile (R6)
    };
    LoadError(Cause c, const std::string& what) : std::runtime_error(what), cause_(c) {}
    Cause cause() const { return cause_; }

private:
    Cause cause_;
};

const char* to_string(LoadError::Cause c);

// ----------------------------------------------------------------------------
// The materialized form
// ----------------------------------------------------------------------------
enum class ScalarType : std::uint8_t { F32 = 0, F16, BF16, I32, I8, U8 };
enum class ComputeTileKind : std::uint8_t { Programmable = 0, FixedVio, FixedFft };
enum class OrchestrationKind : std::uint8_t { RiscvElf = 0 };

const char* to_string(ScalarType t);
const char* to_string(ComputeTileKind k);

// A tensor that lives OUTSIDE the file. 64-bit throughout, whatever the orchestrator's
// XLEN: an RV32 manager core cannot address a 100 GB tensor space, and narrowing these
// later would be a format change.
struct TensorRef {
    std::string name;
    ScalarType dtype = ScalarType::F32;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> tile_shape;      // empty = untiled

    std::uint64_t device_address = 0;           // where the DMA reads it
    std::uint64_t size_bytes = 0;

    // Absent for a tensor the run PRODUCES. An output has no source, and that is how
    // an input and an output are told apart — not by a naming convention.
    std::optional<std::string> source_uri;
    std::uint64_t source_offset = 0;
    std::uint64_t source_length = 0;

    // DECLARED by the producer. Whether anyone CHECKED it belongs to a load, not to a
    // file, so it is not stored here: hashing 100 GB at load time defeats the purpose of
    // mapping it, and a file claiming "verified" would assert what only a loader knows.
    std::string content_digest;

    bool is_input() const { return source_uri.has_value(); }
};

struct DomainFlowProgram {
    std::string name;
    std::string form;                           // names the representation, not assumes it
    std::vector<std::uint8_t> payload;
};

struct Operator {
    std::string name;
    // The #265 L0 TEXT, verbatim. Not a re-encoding: see the schema's comment on why a
    // second representation of one program is the failure this project keeps undoing.
    std::string l0_program;
    ComputeTileKind requires_tile = ComputeTileKind::Programmable;
    std::optional<std::string> domain_flow_program;
    std::optional<std::string> dataflow;        // the space-time map's NAME
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

struct Orchestration {
    OrchestrationKind kind = OrchestrationKind::RiscvElf;
    std::vector<std::uint8_t> image;
    std::string entry_symbol;
};

// R6's capability dimension: what machine this loadable needs. Checked at load and
// REFUSED on mismatch — "rejected, not mis-run".
struct MachineProfile {
    std::uint32_t min_compute_tiles = 1;
    std::vector<ComputeTileKind> required_tile_kinds;
    std::vector<ScalarType> required_dtypes;
    std::uint32_t min_l3_capacity_tiles = 0;    // 0 = no requirement
    std::optional<std::uint32_t> required_l3_modules;
    std::optional<std::uint32_t> required_l2_banks_per_tile;
    std::optional<std::uint32_t> required_l1_vectors;
};

struct Loadable {
    std::string name;
    MachineProfile profile;
    std::vector<Operator> operators;
    std::vector<TensorRef> tensors;
    std::vector<DomainFlowProgram> domain_flow_programs;
    // Absent is legitimate: operators and data with no orchestrator is what increment 1
    // writes, and a host-side driver can still run it.
    std::optional<Orchestration> orchestration;

    // Read back from a file; ignored by the writer, which derives them.
    Version file_format_version{};
    Version file_min_consumer{};
    std::string file_producer;
    Version file_producer_version{};
};

// ----------------------------------------------------------------------------
// The oldest reader that can be trusted with this loadable, DERIVED FROM THE RECORDS
// PRESENT rather than hand-set — the rule #265 increment 2 established and increment 4
// applied without rediscovering.
//
// It also covers the L0 `MIN_CONSUMER` of every embedded program, because a reader that
// accepted the container and then choked on its contents would have been told it was
// safe. That coupling is the price of embedding L0 verbatim, and it is the right price.
// ----------------------------------------------------------------------------
Version min_consumer_for(const Loadable& l);

// ----------------------------------------------------------------------------
// Bytes
// ----------------------------------------------------------------------------
// Canonical: the same Loadable always writes the same bytes, which is what lets a
// digest key a cache and a checked-in fixture be compared byte for byte.
std::string write(const Loadable& l);

// Verifies BEFORE trusting: the generated accessors do not bounds-check, so a malformed
// buffer read through them is undefined behaviour rather than an error. The FlatBuffers
// verifier runs first, then the structural checks this format needs on top of it.
Loadable read(const std::string& bytes);
Loadable read_file(const std::string& path);

// Over the canonical bytes. For provenance and cache lookup — not an identity claim;
// see platform/digest.hpp on why that distinction is kept.
std::string digest(const Loadable& l);

// ----------------------------------------------------------------------------
// Capability checking (R6) — and the distinction that makes it honest
// ----------------------------------------------------------------------------
// A loadable declares what machine it needs; a deployment declares what a machine has.
// Comparing them yields THREE outcomes, not two, and collapsing the third into either
// of the others is how a capability check starts lying:
//
//   SATISFIED     the deployment declares enough
//   MISMATCHED    the deployment declares, and it is not enough  -> REFUSE
//   UNVERIFIABLE  the deployment does not declare it at all      -> REPORT
//
// The third is not a technicality. A `DeploymentSpec` today declares no compute-tile
// KINDS and no dtype support, so a loadable needing an FFT tile or int8 cannot be
// checked against one. Refusing it would reject machines that may well be capable;
// passing it silently would be the "rejected, not mis-run" promise broken in the
// direction that hurts. So it is reported, exactly as `unmodelled_fields` reports a
// declared deployment field a level does not model — the same discipline, pointed the
// other way.
//
// #305 increment 5 adds the kinds to `DeviceSpecification`, and the moment it does,
// tile-kind requirements move from UNVERIFIABLE to checkable and a mismatch becomes a
// refusal. That progression is the plan's, and this is where it lands.
// ----------------------------------------------------------------------------
std::string capability_mismatch(const Loadable& l,
                                const program::platform::DeploymentSpec& spec,
                                std::uint32_t device = 0);

std::vector<std::string> unverifiable_requirements(const Loadable& l,
                                                   const program::platform::DeploymentSpec& spec,
                                                   std::uint32_t device = 0);

// Throws LoadError with Cause::CapabilityMismatch on a provable mismatch. Silent about
// the unverifiable ones by design: a caller that wants those reports them from
// unverifiable_requirements() into its provenance, which is where an unchecked claim
// belongs.
void require_capability(const Loadable& l, const program::platform::DeploymentSpec& spec,
                        std::uint32_t device = 0);

} // namespace sw::kpu::loadable
