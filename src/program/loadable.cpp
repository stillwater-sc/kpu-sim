// ============================================================================
// src/program/loadable.cpp
// The KPU loadable (.kpuld). See include/sw/kpu/loadable/loadable.hpp for the
// contract and schemas/kpu_loadable.fbs for the format.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/loadable/loadable.hpp>

#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/platform/digest.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>
#include <sw/kpu/version.hpp>

#include <kpu/loadable/kpu_loadable_generated.h>

#include <fstream>
#include <set>
#include <sstream>

namespace sw::kpu::loadable {

// ---- versions ---------------------------------------------------------------
Version format_version()   { return {1, 0, 0}; }
Version reader_version()   { return {1, 0, 0}; }

Version producer_version() {
    // The BUILD, not the format. R4's bad_producers needs a version that tracks the
    // build; reusing the format version here would make the field useless for the one
    // thing it exists for.
    return {static_cast<std::uint16_t>(sw::kpu::VERSION_MAJOR),
            static_cast<std::uint16_t>(sw::kpu::VERSION_MINOR),
            static_cast<std::uint16_t>(sw::kpu::VERSION_PATCH)};
}

const char* file_identifier() { return fb::LoadableIdentifier(); }

const char* to_string(LoadError::Cause c) {
    switch (c) {
        case LoadError::Cause::NotALoadable:         return "not a loadable";
        case LoadError::Cause::Truncated:            return "truncated";
        case LoadError::Cause::MalformedContainer:   return "malformed container";
        case LoadError::Cause::UnsupportedVersion:   return "unsupported version";
        case LoadError::Cause::MissingRequiredField: return "missing required field";
        case LoadError::Cause::InconsistentRecord:   return "inconsistent record";
        case LoadError::Cause::CapabilityMismatch:   return "capability mismatch";
    }
    return "?";
}

const char* to_string(ScalarType t) {
    switch (t) {
        case ScalarType::F32:  return "f32";
        case ScalarType::F16:  return "f16";
        case ScalarType::BF16: return "bf16";
        case ScalarType::I32:  return "i32";
        case ScalarType::I8:   return "i8";
        case ScalarType::U8:   return "u8";
    }
    return "?";
}

const char* to_string(ComputeTileKind k) {
    switch (k) {
        case ComputeTileKind::Programmable: return "programmable";
        case ComputeTileKind::FixedVio:     return "fixed:vio";
        case ComputeTileKind::FixedFft:     return "fixed:fft";
    }
    return "?";
}

// ---- min_consumer, derived from the records present -------------------------
Version min_consumer_for(const Loadable& l) {
    Version need = format_version();
    // Nothing in 1.0.0 is optional-with-semantics yet, so the container's own floor is
    // 1.0.0. The mechanism is here from the start anyway, because retrofitting it is what
    // #265 had to avoid and did: a future optional record that carries MEANING raises this,
    // and the writer never hand-sets it.
    //
    // THE EMBEDDED L0 PROGRAMS RAISE IT TOO. A reader told "1.0.0 is enough" that then
    // choked on a VALUES_ROW or a STREAMS record inside an operator would have been
    // misinformed by this very field. That coupling is the price of embedding L0 verbatim
    // rather than re-encoding it, and it is the right price -- but it has to be paid here
    // rather than assumed away.
    for (const Operator& op : l.operators) {
        program::serialize::LoadInfo info;
        try {
            (void)program::serialize::from_string(op.l0_program, &info);
        } catch (const std::exception&) {
            continue;   // read() validates these; min_consumer_for does not diagnose
        }
        const program::serialize::Version l0 =
            program::serialize::min_consumer_for(info.has_values, info.has_streams);
        // The L0 format and the container version their own axes independently, so the
        // container cannot simply adopt an L0 version number. What it must guarantee is
        // that a reader able to read THIS container can also read the programs inside it,
        // which for now is true of any 1.x reader because this build's L0 reader is 1.2.0.
        // Recorded as a check rather than a conversion: if a future L0 needs more than the
        // container's reader offers, that is a container bump, and this is where it is
        // noticed.
        if (!(l0 <= program::serialize::reader_version()))
            need = Version{static_cast<std::uint16_t>(need.major + 1), 0, 0};
    }
    return need;
}

namespace {

fb::Version to_fb(const Version& v) { return fb::Version(v.major, v.minor, v.patch); }
Version from_fb(const fb::Version* v) {
    return v ? Version{v->major(), v->minor(), v->patch()} : Version{};
}

[[noreturn]] void fail(LoadError::Cause c, const std::string& what) {
    throw LoadError(c, "loadable: " + what);
}

template <typename T>
std::vector<T> vec_of(const flatbuffers::Vector<T>* v) {
    std::vector<T> out;
    if (!v) return out;
    out.reserve(v->size());
    for (auto x : *v) out.push_back(x);
    return out;
}

std::vector<std::string> strings_of(
        const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>* v) {
    std::vector<std::string> out;
    if (!v) return out;
    out.reserve(v->size());
    for (auto s : *v) out.push_back(s ? s->str() : std::string());
    return out;
}

} // namespace

// ---- write ------------------------------------------------------------------
std::string write(const Loadable& l) {
    flatbuffers::FlatBufferBuilder b(1024);

    std::vector<flatbuffers::Offset<fb::Operator>> ops;
    for (const Operator& op : l.operators) {
        const auto name = b.CreateString(op.name);
        const auto prog = b.CreateString(op.l0_program);
        const auto dfp = op.domain_flow_program
                             ? b.CreateString(*op.domain_flow_program)
                             : flatbuffers::Offset<flatbuffers::String>();
        const auto flow = op.dataflow ? b.CreateString(*op.dataflow)
                                      : flatbuffers::Offset<flatbuffers::String>();
        const auto ins = b.CreateVectorOfStrings(op.inputs);
        const auto outs = b.CreateVectorOfStrings(op.outputs);
        fb::OperatorBuilder ob(b);
        ob.add_name(name);
        ob.add_l0_program(prog);
        ob.add_requires_tile(static_cast<fb::ComputeTileKind>(op.requires_tile));
        if (!dfp.IsNull()) ob.add_domain_flow_program(dfp);
        if (!flow.IsNull()) ob.add_dataflow(flow);
        ob.add_inputs(ins);
        ob.add_outputs(outs);
        ops.push_back(ob.Finish());
    }

    std::vector<flatbuffers::Offset<fb::TensorRef>> tensors;
    for (const TensorRef& t : l.tensors) {
        const auto name = b.CreateString(t.name);
        const auto shape = b.CreateVector(t.shape);
        const auto tile = t.tile_shape.empty()
                              ? flatbuffers::Offset<flatbuffers::Vector<std::uint64_t>>()
                              : b.CreateVector(t.tile_shape);
        const auto uri = t.source_uri ? b.CreateString(*t.source_uri)
                                      : flatbuffers::Offset<flatbuffers::String>();
        const auto dig = b.CreateString(t.content_digest);
        fb::TensorRefBuilder tb(b);
        tb.add_name(name);
        tb.add_dtype(static_cast<fb::ScalarType>(t.dtype));
        tb.add_shape(shape);
        if (!tile.IsNull()) tb.add_tile_shape(tile);
        tb.add_device_address(t.device_address);
        tb.add_size_bytes(t.size_bytes);
        if (!uri.IsNull()) {
            tb.add_source_uri(uri);
            tb.add_source_offset(t.source_offset);
            tb.add_source_length(t.source_length);
        }
        tb.add_content_digest(dig);
        tensors.push_back(tb.Finish());
    }

    std::vector<flatbuffers::Offset<fb::DomainFlowProgram>> dfps;
    for (const DomainFlowProgram& d : l.domain_flow_programs) {
        const auto name = b.CreateString(d.name);
        const auto form = b.CreateString(d.form);
        const auto payload = b.CreateVector(d.payload);
        fb::DomainFlowProgramBuilder db(b);
        db.add_name(name);
        db.add_form(form);
        db.add_payload(payload);
        dfps.push_back(db.Finish());
    }

    flatbuffers::Offset<fb::Orchestration> orch;
    if (l.orchestration) {
        const auto image = b.CreateVector(l.orchestration->image);
        const auto entry = b.CreateString(l.orchestration->entry_symbol);
        fb::OrchestrationBuilder ob(b);
        ob.add_kind(static_cast<fb::OrchestrationKind>(l.orchestration->kind));
        ob.add_image(image);
        ob.add_entry_symbol(entry);
        orch = ob.Finish();
    }

    const auto kinds = b.CreateVector(
        [&] {
            std::vector<fb::ComputeTileKind> v;
            for (ComputeTileKind k : l.profile.required_tile_kinds)
                v.push_back(static_cast<fb::ComputeTileKind>(k));
            return v;
        }());
    const auto dtypes = b.CreateVector(
        [&] {
            std::vector<fb::ScalarType> v;
            for (ScalarType t : l.profile.required_dtypes)
                v.push_back(static_cast<fb::ScalarType>(t));
            return v;
        }());
    fb::MachineProfileBuilder pb(b);
    pb.add_min_compute_tiles(l.profile.min_compute_tiles);
    pb.add_required_tile_kinds(kinds);
    pb.add_required_dtypes(dtypes);
    pb.add_min_l3_capacity_tiles(l.profile.min_l3_capacity_tiles);
    if (l.profile.required_l3_modules)
        pb.add_required_l3_modules(*l.profile.required_l3_modules);
    if (l.profile.required_l2_banks_per_tile)
        pb.add_required_l2_banks_per_tile(*l.profile.required_l2_banks_per_tile);
    if (l.profile.required_l1_vectors)
        pb.add_required_l1_vectors(*l.profile.required_l1_vectors);
    const auto profile = pb.Finish();

    const auto ops_v = b.CreateVector(ops);
    const auto tensors_v = b.CreateVector(tensors);
    const auto dfps_v = dfps.empty()
                            ? flatbuffers::Offset<flatbuffers::Vector<
                                  flatbuffers::Offset<fb::DomainFlowProgram>>>()
                            : b.CreateVector(dfps);
    const auto name = b.CreateString(l.name);
    const auto producer = b.CreateString("kpu-sim");

    // min_consumer is DERIVED, never taken from the caller: a hand-set floor is a promise
    // nobody checked.
    const fb::Version fv = to_fb(format_version());
    const fb::Version mc = to_fb(min_consumer_for(l));
    const fb::Version pv = to_fb(producer_version());

    fb::LoadableBuilder lb(b);
    lb.add_format_version(&fv);
    lb.add_min_consumer(&mc);
    lb.add_producer(producer);
    lb.add_producer_version(&pv);
    lb.add_name(name);
    lb.add_profile(profile);
    lb.add_operators(ops_v);
    lb.add_tensors(tensors_v);
    if (!dfps_v.IsNull()) lb.add_domain_flow_programs(dfps_v);
    if (!orch.IsNull()) lb.add_orchestration(orch);
    const auto root = lb.Finish();

    // FinishLoadableBuffer stamps the file identifier, so the magic comes from the format
    // rather than from a hand-rolled header (R1).
    fb::FinishLoadableBuffer(b, root);
    return std::string(reinterpret_cast<const char*>(b.GetBufferPointer()), b.GetSize());
}

// ---- read -------------------------------------------------------------------
Loadable read(const std::string& bytes) {
    // IDENTIFY FIRST. "Not for me" and "broken" are different answers, and a caller that
    // handed us a PNG deserves the first one.
    if (bytes.size() < 8)
        fail(LoadError::Cause::Truncated,
             "only " + std::to_string(bytes.size()) + " bytes; too short to be a loadable");
    const auto* raw = reinterpret_cast<const std::uint8_t*>(bytes.data());
    if (!fb::LoadableBufferHasIdentifier(raw))
        fail(LoadError::Cause::NotALoadable,
             std::string("expected file identifier '") + file_identifier() + "'");

    // VERIFY BEFORE TRUSTING. The generated accessors do not bounds-check, so reading a
    // malformed buffer through them is undefined behaviour rather than an error -- the one
    // way a binary format is more dangerous than the L0 text was.
    flatbuffers::Verifier v(raw, bytes.size());
    if (!fb::VerifyLoadableBuffer(v))
        fail(LoadError::Cause::Truncated, "the buffer failed FlatBuffers verification");

    const fb::Loadable* f = fb::GetLoadable(raw);
    if (!f) fail(LoadError::Cause::MalformedContainer, "no root table");

    if (!f->format_version() || !f->min_consumer() || !f->producer_version())
        fail(LoadError::Cause::MissingRequiredField, "a version field is absent");

    const Version file_format = from_fb(f->format_version());
    // A MAJOR bump means the container changed shape, so an older reader must not try.
    if (reader_version().major < file_format.major)
        fail(LoadError::Cause::UnsupportedVersion,
             "file format " + file_format.str() + " is newer than this reader (" +
                 reader_version().str() + ")");

    const Version need = from_fb(f->min_consumer());
    if (!(need <= reader_version()))
        fail(LoadError::Cause::UnsupportedVersion,
             "file requires a reader >= " + need.str() + "; this build is " +
                 reader_version().str());

    Loadable out;
    out.file_format_version = file_format;
    out.file_min_consumer = need;
    out.file_producer = f->producer() ? f->producer()->str() : std::string();
    out.file_producer_version = from_fb(f->producer_version());
    out.name = f->name() ? f->name()->str() : std::string();

    if (const fb::MachineProfile* p = f->profile()) {
        out.profile.min_compute_tiles = p->min_compute_tiles();
        for (auto k : vec_of(p->required_tile_kinds()))
            out.profile.required_tile_kinds.push_back(static_cast<ComputeTileKind>(k));
        for (auto t : vec_of(p->required_dtypes()))
            out.profile.required_dtypes.push_back(static_cast<ScalarType>(t));
        out.profile.min_l3_capacity_tiles = p->min_l3_capacity_tiles();
        // ABSENT IS NOT A DEFAULT: read through the field presence, not the value, so a
        // loadable that declares nothing about L3 modules is distinguishable from one that
        // declares zero of them.
        // OPTIONAL SCALARS, so absence is in the type rather than inferred from a value.
        if (const auto v = p->required_l3_modules()) out.profile.required_l3_modules = *v;
        if (const auto v = p->required_l2_banks_per_tile())
            out.profile.required_l2_banks_per_tile = *v;
        if (const auto v = p->required_l1_vectors()) out.profile.required_l1_vectors = *v;
    } else {
        fail(LoadError::Cause::MissingRequiredField, "no machine profile");
    }

    if (!f->tensors()) fail(LoadError::Cause::MissingRequiredField, "no tensor table");
    std::set<std::string> tensor_names;
    for (const fb::TensorRef* t : *f->tensors()) {
        TensorRef r;
        if (!t->name() || t->name()->size() == 0)
            fail(LoadError::Cause::MissingRequiredField, "a tensor has no name");
        r.name = t->name()->str();
        if (!tensor_names.insert(r.name).second)
            fail(LoadError::Cause::InconsistentRecord,
                 "two tensors are named \"" + r.name + "\"");
        r.dtype = static_cast<ScalarType>(t->dtype());
        r.shape = vec_of(t->shape());
        if (r.shape.empty())
            fail(LoadError::Cause::MissingRequiredField,
                 "tensor \"" + r.name + "\" has no shape");
        r.tile_shape = vec_of(t->tile_shape());
        if (!r.tile_shape.empty() && r.tile_shape.size() != r.shape.size())
            fail(LoadError::Cause::InconsistentRecord,
                 "tensor \"" + r.name + "\": tile shape has " +
                     std::to_string(r.tile_shape.size()) + " dimensions, the shape has " +
                     std::to_string(r.shape.size()));
        r.device_address = t->device_address();
        r.size_bytes = t->size_bytes();
        if (t->source_uri()) {
            r.source_uri = t->source_uri()->str();
            r.source_offset = t->source_offset();
            r.source_length = t->source_length();
            // A source that cannot hold the tensor is a file that contradicts itself, and
            // the DMA would read past the end of the blob rather than report anything.
            if (r.size_bytes && r.source_length && r.source_length < r.size_bytes)
                fail(LoadError::Cause::InconsistentRecord,
                     "tensor \"" + r.name + "\": source_length " +
                         std::to_string(r.source_length) + " is smaller than size_bytes " +
                         std::to_string(r.size_bytes));
        }
        if (t->content_digest()) r.content_digest = t->content_digest()->str();
        out.tensors.push_back(std::move(r));
    }

    if (f->domain_flow_programs()) {
        for (const fb::DomainFlowProgram* d : *f->domain_flow_programs()) {
            DomainFlowProgram p;
            if (!d->name() || !d->form() || !d->payload())
                fail(LoadError::Cause::MissingRequiredField,
                     "a domain flow program is missing a required field");
            p.name = d->name()->str();
            p.form = d->form()->str();
            p.payload = vec_of(d->payload());
            out.domain_flow_programs.push_back(std::move(p));
        }
    }

    if (!f->operators()) fail(LoadError::Cause::MissingRequiredField, "no operator table");
    std::set<std::string> op_names;
    for (const fb::Operator* o : *f->operators()) {
        Operator op;
        if (!o->name() || o->name()->size() == 0)
            fail(LoadError::Cause::MissingRequiredField, "an operator has no name");
        op.name = o->name()->str();
        if (!op_names.insert(op.name).second)
            fail(LoadError::Cause::InconsistentRecord,
                 "two operators are named \"" + op.name + "\"");
        if (!o->l0_program() || o->l0_program()->size() == 0)
            fail(LoadError::Cause::MissingRequiredField,
                 "operator \"" + op.name + "\" carries no L0 program");
        op.l0_program = o->l0_program()->str();
        // THE EMBEDDED PROGRAM IS VALIDATED HERE, not on first execution. A container that
        // loads and then fails to run is the worst of both: it reports success and dies
        // later, somewhere else.
        try {
            (void)program::serialize::from_string(op.l0_program);
        } catch (const std::exception& e) {
            fail(LoadError::Cause::InconsistentRecord,
                 "operator \"" + op.name + "\": its L0 program does not load (" + e.what() +
                     ")");
        }
        op.requires_tile = static_cast<ComputeTileKind>(o->requires_tile());
        if (o->domain_flow_program())
            op.domain_flow_program = o->domain_flow_program()->str();
        if (o->dataflow()) op.dataflow = o->dataflow()->str();
        op.inputs = strings_of(o->inputs());
        op.outputs = strings_of(o->outputs());

        // Every operand must name a declared tensor. An undeclared operand would be a
        // descriptor the orchestrator could not issue, discovered at run time.
        for (const auto& which : {std::cref(op.inputs), std::cref(op.outputs)})
            for (const std::string& t : which.get())
                if (!tensor_names.count(t))
                    fail(LoadError::Cause::InconsistentRecord,
                         "operator \"" + op.name + "\" names tensor \"" + t +
                             "\", which the tensor table does not declare");

        // A programmable tile needs a program; a fixed tile must not be given one. Both
        // directions, because either mismatch means the loadable describes a machine
        // configuration it did not intend.
        if (op.requires_tile == ComputeTileKind::Programmable) {
            if (op.domain_flow_program) {
                bool found = false;
                for (const DomainFlowProgram& d : out.domain_flow_programs)
                    found = found || d.name == *op.domain_flow_program;
                if (!found)
                    fail(LoadError::Cause::InconsistentRecord,
                         "operator \"" + op.name + "\" names domain flow program \"" +
                             *op.domain_flow_program + "\", which is not in the file");
            }
        } else if (op.domain_flow_program) {
            fail(LoadError::Cause::InconsistentRecord,
                 "operator \"" + op.name + "\" requires a " +
                     to_string(op.requires_tile) +
                     " tile and also names a domain flow program; a fixed-ISA tile takes "
                     "no program");
        }
        out.operators.push_back(std::move(op));
    }

    if (const fb::Orchestration* orch = f->orchestration()) {
        Orchestration o;
        o.kind = static_cast<OrchestrationKind>(orch->kind());
        if (!orch->image() || orch->image()->size() == 0)
            fail(LoadError::Cause::MissingRequiredField,
                 "the orchestration section carries no image");
        o.image = vec_of(orch->image());
        if (orch->entry_symbol()) o.entry_symbol = orch->entry_symbol()->str();
        out.orchestration = std::move(o);
    }

    return out;
}

Loadable read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(LoadError::Cause::NotALoadable, "cannot read '" + path + "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    try {
        return read(ss.str());
    } catch (const LoadError& e) {
        // The path belongs in the message the moment anything reads more than one file.
        throw LoadError(e.cause(), std::string(e.what()) + " (in '" + path + "')");
    }
}

std::string digest(const Loadable& l) { return program::platform::digest_of(write(l)); }

// ---- capability checking ----------------------------------------------------
namespace {

// A required count against what the deployment DECLARES. `declared` absent means the
// deployment says nothing, which is neither satisfaction nor mismatch -- see the header on
// why that third outcome is kept rather than collapsed.
std::string shortfall(const char* what, std::uint32_t required,
                      const std::optional<program::Dim>& declared) {
    if (!declared) return {};            // unverifiable, reported elsewhere
    if (*declared >= required) return {};
    return std::string(what) + ": the loadable needs " + std::to_string(required) +
           ", the deployment declares " + std::to_string(*declared);
}

} // namespace

std::string capability_mismatch(const Loadable& l,
                                const program::platform::DeploymentSpec& spec,
                                std::uint32_t device) {
    if (device >= spec.devices.size())
        return "device " + std::to_string(device) + " is not in this deployment";
    const program::platform::DeviceSpecification& d =
        spec.device(static_cast<program::Dim>(device));

    if (d.compute_tiles < l.profile.min_compute_tiles)
        return "compute tiles: the loadable needs " +
               std::to_string(l.profile.min_compute_tiles) + ", the deployment declares " +
               std::to_string(d.compute_tiles);

    // 0 means UNBOUNDED in a DeviceSpecification, so it satisfies any requirement. Reading
    // it as "zero capacity" would refuse every loadable on the default deployment, which is
    // the same conflation DeviceDescriptor::l3_tiles already warns about.
    if (l.profile.min_l3_capacity_tiles != 0 && d.l3.capacity_tiles != 0 &&
        d.l3.capacity_tiles < l.profile.min_l3_capacity_tiles)
        return "l3 capacity: the loadable needs " +
               std::to_string(l.profile.min_l3_capacity_tiles) +
               " tiles, the deployment declares " + std::to_string(d.l3.capacity_tiles);

    if (l.profile.required_l3_modules) {
        const std::string bad =
            shortfall("l3 modules", *l.profile.required_l3_modules, d.l3.tiles);
        if (!bad.empty()) return bad;
    }
    if (l.profile.required_l2_banks_per_tile) {
        const std::string bad = shortfall("l2 banks per tile",
                                          *l.profile.required_l2_banks_per_tile,
                                          d.l2.banks_per_tile);
        if (!bad.empty()) return bad;
    }
    if (l.profile.required_l1_vectors) {
        const std::string bad =
            shortfall("l1 vectors", *l.profile.required_l1_vectors, d.l1.vectors);
        if (!bad.empty()) return bad;
    }

    // An operator needing a compute tile the DEPLOYMENT cannot be shown to have is
    // unverifiable rather than mismatched today (see the header). What IS checkable: an
    // operator cannot run at all on a device with no compute tiles, and every operator
    // needs one.
    if (!l.operators.empty() && d.compute_tiles == 0)
        return "the deployment declares no compute tiles";

    return {};
}

std::vector<std::string> unverifiable_requirements(
        const Loadable& l, const program::platform::DeploymentSpec& spec,
        std::uint32_t device) {
    std::vector<std::string> out;
    if (device >= spec.devices.size()) return out;
    const program::platform::DeviceSpecification& d =
        spec.device(static_cast<program::Dim>(device));

    // A DeploymentSpec declares no compute-tile KINDS, so a fixed-ISA requirement cannot be
    // checked against one. #305 increment 5 adds the kinds and this moves to a refusal.
    std::set<ComputeTileKind> kinds(l.profile.required_tile_kinds.begin(),
                                    l.profile.required_tile_kinds.end());
    for (const Operator& op : l.operators) kinds.insert(op.requires_tile);
    for (ComputeTileKind k : kinds)
        if (k != ComputeTileKind::Programmable)
            out.push_back(std::string("compute-tile kind ") + to_string(k) +
                          " is required, and a deployment does not declare kinds yet "
                          "(#305 increment 5)");

    // Nor does it declare dtype support. element_bytes is NOT a substitute: it says how
    // wide an element is, not which types the fabric implements, and inferring one from the
    // other would be a check that looks like evidence and is not.
    for (ScalarType t : l.profile.required_dtypes)
        out.push_back(std::string("dtype ") + to_string(t) +
                      " is required, and a deployment does not declare dtype support "
                      "(element_bytes is " + std::to_string(d.element_bytes) +
                      ", which is a width and not a type list)");

    if (l.profile.required_l3_modules && !d.l3.tiles)
        out.push_back("l3.tiles is required but the deployment does not declare it");
    if (l.profile.required_l2_banks_per_tile && !d.l2.banks_per_tile)
        out.push_back("l2.banks_per_tile is required but the deployment does not declare it");
    if (l.profile.required_l1_vectors && !d.l1.vectors)
        out.push_back("l1.vectors is required but the deployment does not declare it");
    return out;
}

void require_capability(const Loadable& l, const program::platform::DeploymentSpec& spec,
                        std::uint32_t device) {
    const std::string bad = capability_mismatch(l, spec, device);
    if (!bad.empty()) fail(LoadError::Cause::CapabilityMismatch, bad);
}

} // namespace sw::kpu::loadable
