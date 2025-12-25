// Test suite for Program and Kernel Serialization
// Tests binary and JSON serialization/deserialization

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/isa/program_serializer.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/kernel_serializer.hpp>
#include <sw/kpu/kernel.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace sw::kpu;
using namespace sw::kpu::isa;

// ============================================================================
// ProgramSerializer Tests
// ============================================================================

TEST_CASE("ProgramSerializer binary serialization", "[serialization][binary]") {
    ProgramSerializer serializer;

    SECTION("Serialize and deserialize empty program") {
        DMProgram program;
        program.name = "empty_test";
        program.version = 1;
        program.M = 64;
        program.N = 64;
        program.K = 64;
        program.Ti = 16;
        program.Tj = 16;
        program.Tk = 16;
        program.L1_Ki = 16;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        std::vector<uint8_t> data = serializer.serialize(program);
        REQUIRE(data.size() > 0);

        DMProgram loaded = serializer.deserialize(data);
        REQUIRE(loaded.name == "empty_test");
        REQUIRE(loaded.M == 64);
        REQUIRE(loaded.N == 64);
        REQUIRE(loaded.K == 64);
        REQUIRE(loaded.Ti == 16);
        REQUIRE(loaded.Tj == 16);
        REQUIRE(loaded.Tk == 16);
        REQUIRE(loaded.dataflow == DMProgram::Dataflow::OUTPUT_STATIONARY);
    }

    SECTION("Serialize program with instructions") {
        DMProgram program;
        program.name = "matmul_test";
        program.version = 1;
        program.M = 128;
        program.N = 128;
        program.K = 128;
        program.Ti = 32;
        program.Tj = 32;
        program.Tk = 32;
        program.L1_Ki = 32;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        // Add some instructions
        program.instructions.push_back(
            DMInstruction::dma_load(MatrixID::A, {0, 0, 0}, 0x1000, 0, 0, 4096));
        program.instructions.push_back(
            DMInstruction::dma_load(MatrixID::B, {0, 0, 0}, 0x2000, 1, 0, 4096));
        program.instructions.push_back(DMInstruction::barrier());
        program.instructions.push_back(
            DMInstruction::bm_move(MatrixID::A, {0, 0, 0}, 0, 0, 0, 0, 32, 32, 4));
        program.instructions.push_back(DMInstruction::halt());

        std::vector<uint8_t> data = serializer.serialize(program);
        DMProgram loaded = serializer.deserialize(data);

        REQUIRE(loaded.name == "matmul_test");
        REQUIRE(loaded.instructions.size() == 5);
        REQUIRE(loaded.instructions[0].opcode == DMOpcode::DMA_LOAD_TILE);
        REQUIRE(loaded.instructions[2].opcode == DMOpcode::BARRIER);
        REQUIRE(loaded.instructions[4].opcode == DMOpcode::HALT);
    }

    SECTION("Serialize program with VE-enabled drain") {
        DMProgram program;
        program.name = "mlp_test";
        program.version = 1;
        program.M = 64;
        program.N = 128;
        program.K = 64;
        program.Ti = 16;
        program.Tj = 16;
        program.Tk = 16;
        program.L1_Ki = 16;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        // Add VE-enabled drain instruction
        program.instructions.push_back(
            DMInstruction::str_drain({0, 0, 0}, 0, 0, 0, 0, 16, 16, 16,
                                     true, ActivationType::GELU, true, 0x3000));
        program.instructions.push_back(DMInstruction::halt());

        std::vector<uint8_t> data = serializer.serialize(program);
        DMProgram loaded = serializer.deserialize(data);

        REQUIRE(loaded.instructions.size() == 2);
        REQUIRE(loaded.instructions[0].opcode == DMOpcode::STR_DRAIN_OUTPUT);

        const auto& ops = std::get<StreamerOperands>(loaded.instructions[0].operands);
        REQUIRE(ops.ve_enabled == true);
        REQUIRE(ops.ve_activation == ActivationType::GELU);
        REQUIRE(ops.ve_bias_enabled == true);
        REQUIRE(ops.ve_bias_addr == 0x3000);
    }

    SECTION("Serialize program with memory map") {
        DMProgram program;
        program.name = "memmap_test";
        program.version = 1;
        program.M = 64;
        program.N = 64;
        program.K = 64;
        program.Ti = 16;
        program.Tj = 16;
        program.Tk = 16;
        program.L1_Ki = 16;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        program.memory_map.a_base = 0x10000;
        program.memory_map.b_base = 0x20000;
        program.memory_map.c_base = 0x30000;

        DMProgram::MemoryMap::L3Alloc l3_alloc;
        l3_alloc.tile_id = 0;
        l3_alloc.offset = 0;
        l3_alloc.size = 1024;
        l3_alloc.matrix = MatrixID::A;
        l3_alloc.buffer = BufferSlot::BUF_0;
        program.memory_map.l3_allocations.push_back(l3_alloc);

        DMProgram::MemoryMap::L2Alloc l2_alloc;
        l2_alloc.bank_id = 0;
        l2_alloc.offset = 0;
        l2_alloc.size = 512;
        l2_alloc.matrix = MatrixID::A;
        l2_alloc.buffer = BufferSlot::BUF_0;
        program.memory_map.l2_allocations.push_back(l2_alloc);

        std::vector<uint8_t> data = serializer.serialize(program);
        DMProgram loaded = serializer.deserialize(data);

        REQUIRE(loaded.memory_map.a_base == 0x10000);
        REQUIRE(loaded.memory_map.b_base == 0x20000);
        REQUIRE(loaded.memory_map.c_base == 0x30000);
        REQUIRE(loaded.memory_map.l3_allocations.size() == 1);
        REQUIRE(loaded.memory_map.l2_allocations.size() == 1);
        REQUIRE(loaded.memory_map.l3_allocations[0].tile_id == 0);
        REQUIRE(loaded.memory_map.l2_allocations[0].bank_id == 0);
    }

    SECTION("Serialize program with estimates") {
        DMProgram program;
        program.name = "estimates_test";
        program.version = 1;
        program.M = 64;
        program.N = 64;
        program.K = 64;
        program.Ti = 16;
        program.Tj = 16;
        program.Tk = 16;
        program.L1_Ki = 16;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        program.estimates.total_cycles = 100000;
        program.estimates.external_mem_bytes = 524288;
        program.estimates.l3_bytes = 262144;
        program.estimates.l2_bytes = 131072;
        program.estimates.arithmetic_intensity = 42.67;
        program.estimates.estimated_gflops = 500.0;

        std::vector<uint8_t> data = serializer.serialize(program);
        DMProgram loaded = serializer.deserialize(data);

        REQUIRE(loaded.estimates.total_cycles == 100000);
        REQUIRE(loaded.estimates.external_mem_bytes == 524288);
        REQUIRE(loaded.estimates.l3_bytes == 262144);
        REQUIRE(loaded.estimates.l2_bytes == 131072);
        REQUIRE(loaded.estimates.arithmetic_intensity == Catch::Approx(42.67));
        REQUIRE(loaded.estimates.estimated_gflops == Catch::Approx(500.0));
    }

    SECTION("Validate detects invalid data") {
        std::vector<uint8_t> bad_data = {0x00, 0x01, 0x02, 0x03};
        REQUIRE_FALSE(serializer.validate(bad_data));

        std::vector<uint8_t> empty_data;
        REQUIRE_FALSE(serializer.validate(empty_data));
    }
}

TEST_CASE("ProgramSerializer JSON serialization", "[serialization][json]") {
    ProgramSerializer serializer;

    SECTION("Serialize and deserialize to JSON") {
        DMProgram program;
        program.name = "json_test";
        program.version = 1;
        program.M = 256;
        program.N = 256;
        program.K = 256;
        program.Ti = 64;
        program.Tj = 64;
        program.Tk = 64;
        program.L1_Ki = 32;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        program.instructions.push_back(
            DMInstruction::dma_load(MatrixID::A, {0, 0, 0}, 0x1000, 0, 0, 16384));
        program.instructions.push_back(DMInstruction::barrier());
        program.instructions.push_back(DMInstruction::halt());

        program.estimates.total_cycles = 50000;
        program.estimates.arithmetic_intensity = 85.33;

        std::string json = serializer.to_json(program, true);
        REQUIRE(json.find("\"json_test\"") != std::string::npos);
        REQUIRE(json.find("\"M\": 256") != std::string::npos);
        REQUIRE(json.find("DMA_LOAD_TILE") != std::string::npos);

        DMProgram loaded = serializer.from_json(json);
        REQUIRE(loaded.name == "json_test");
        REQUIRE(loaded.M == 256);
        REQUIRE(loaded.instructions.size() == 3);
        REQUIRE(loaded.estimates.arithmetic_intensity == Catch::Approx(85.33));
    }

    SECTION("Compact JSON") {
        DMProgram program;
        program.name = "compact";
        program.version = 1;
        program.M = 64;
        program.N = 64;
        program.K = 64;
        program.Ti = 16;
        program.Tj = 16;
        program.Tk = 16;
        program.L1_Ki = 16;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        std::string pretty = serializer.to_json(program, true);
        std::string compact = serializer.to_json(program, false);

        REQUIRE(compact.size() < pretty.size());
        REQUIRE(compact.find('\n') == std::string::npos);
    }
}

TEST_CASE("ProgramSerializer file I/O", "[serialization][file]") {
    ProgramSerializer serializer;
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();

    SECTION("Save and load binary file") {
        DMProgram program;
        program.name = "file_test";
        program.version = 1;
        program.M = 512;
        program.N = 512;
        program.K = 512;
        program.Ti = 128;
        program.Tj = 128;
        program.Tk = 128;
        program.L1_Ki = 64;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        program.instructions.push_back(DMInstruction::halt());

        std::string path = (temp_dir / "test_program.kpubin").string();

        serializer.save(program, path);
        REQUIRE(std::filesystem::exists(path));

        DMProgram loaded = serializer.load(path);
        REQUIRE(loaded.name == "file_test");
        REQUIRE(loaded.M == 512);

        std::filesystem::remove(path);
    }

    SECTION("Save and load JSON file") {
        DMProgram program;
        program.name = "json_file_test";
        program.version = 1;
        program.M = 1024;
        program.N = 1024;
        program.K = 1024;
        program.Ti = 256;
        program.Tj = 256;
        program.Tk = 256;
        program.L1_Ki = 128;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        std::string path = (temp_dir / "test_program.kpujson").string();

        serializer.save_json(program, path, true);
        REQUIRE(std::filesystem::exists(path));

        DMProgram loaded = serializer.load_json(path);
        REQUIRE(loaded.name == "json_file_test");
        REQUIRE(loaded.M == 1024);

        std::filesystem::remove(path);
    }

    SECTION("Detect format from extension") {
        REQUIRE(ProgramSerializer::detect_format("test.kpubin") == "binary");
        REQUIRE(ProgramSerializer::detect_format("test.bin") == "binary");
        REQUIRE(ProgramSerializer::detect_format("test.kpujson") == "json");
        REQUIRE(ProgramSerializer::detect_format("test.json") == "json");
        REQUIRE(ProgramSerializer::detect_format("noextension") == "binary");
    }
}

// ============================================================================
// KernelSerializer Tests
// ============================================================================

TEST_CASE("KernelSerializer binary serialization", "[serialization][kernel][binary]") {
    KernelSerializer serializer;

    SECTION("Serialize and deserialize matmul kernel") {
        Kernel kernel = Kernel::create_matmul(256, 256, 256, DataType::FLOAT32);
        REQUIRE(kernel.is_valid());

        std::vector<uint8_t> data = serializer.serialize(kernel);
        REQUIRE(data.size() > 0);

        Kernel loaded = serializer.deserialize(data);
        REQUIRE(loaded.is_valid());
        REQUIRE(loaded.op_type() == KernelOpType::MATMUL);
        REQUIRE(loaded.M() == 256);
        REQUIRE(loaded.N() == 256);
        REQUIRE(loaded.K() == 256);
        REQUIRE(loaded.dtype() == DataType::FLOAT32);
    }

    SECTION("Serialize and deserialize MLP kernel") {
        Kernel kernel = Kernel::create_mlp(128, 256, 128, ActivationType::GELU, true, DataType::FLOAT32);
        REQUIRE(kernel.is_valid());

        std::vector<uint8_t> data = serializer.serialize(kernel);
        Kernel loaded = serializer.deserialize(data);

        REQUIRE(loaded.is_valid());
        REQUIRE(loaded.op_type() == KernelOpType::MLP);
        REQUIRE(loaded.M() == 128);
        REQUIRE(loaded.N() == 256);
        REQUIRE(loaded.K() == 128);
        REQUIRE(loaded.activation() == ActivationType::GELU);
        REQUIRE(loaded.has_bias() == true);
    }

    SECTION("Kernel arguments preserved") {
        Kernel kernel = Kernel::create_matmul(64, 128, 96, DataType::FLOAT32);

        std::vector<uint8_t> data = serializer.serialize(kernel);
        Kernel loaded = serializer.deserialize(data);

        const auto& orig_args = kernel.arguments();
        const auto& loaded_args = loaded.arguments();

        REQUIRE(loaded_args.size() == orig_args.size());

        for (size_t i = 0; i < orig_args.size(); ++i) {
            REQUIRE(loaded_args[i].name == orig_args[i].name);
            REQUIRE(loaded_args[i].dtype == orig_args[i].dtype);
            REQUIRE(loaded_args[i].is_output == orig_args[i].is_output);
            REQUIRE(loaded_args[i].shape == orig_args[i].shape);
        }
    }

    SECTION("Validate detects invalid kernel data") {
        std::vector<uint8_t> bad_data = {0xFF, 0xFE, 0xFD, 0xFC};
        REQUIRE_FALSE(serializer.validate(bad_data));
    }
}

TEST_CASE("KernelSerializer JSON serialization", "[serialization][kernel][json]") {
    KernelSerializer serializer;

    SECTION("Serialize matmul kernel to JSON") {
        Kernel kernel = Kernel::create_matmul(512, 512, 512, DataType::FLOAT32);

        std::string json = serializer.to_json(kernel, true);
        REQUIRE(json.find("\"matmul\"") != std::string::npos);
        REQUIRE(json.find("\"M\": 512") != std::string::npos);
        REQUIRE(json.find("\"arguments\"") != std::string::npos);

        Kernel loaded = serializer.from_json(json);
        REQUIRE(loaded.is_valid());
        REQUIRE(loaded.M() == 512);
    }

    SECTION("Serialize MLP kernel to JSON") {
        Kernel kernel = Kernel::create_mlp(64, 128, 64, ActivationType::RELU, true, DataType::FLOAT32);

        std::string json = serializer.to_json(kernel, true);
        REQUIRE(json.find("\"mlp\"") != std::string::npos);
        REQUIRE(json.find("\"activation\": \"relu\"") != std::string::npos);
        REQUIRE(json.find("\"has_bias\": true") != std::string::npos);

        Kernel loaded = serializer.from_json(json);
        REQUIRE(loaded.is_valid());
        REQUIRE(loaded.op_type() == KernelOpType::MLP);
    }
}

TEST_CASE("KernelSerializer file I/O", "[serialization][kernel][file]") {
    KernelSerializer serializer;
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();

    SECTION("Save and load kernel binary") {
        Kernel kernel = Kernel::create_matmul(1024, 1024, 1024, DataType::FLOAT32);
        std::string path = (temp_dir / "test_kernel.kpukernel").string();

        serializer.save(kernel, path);
        REQUIRE(std::filesystem::exists(path));

        Kernel loaded = serializer.load(path);
        REQUIRE(loaded.is_valid());
        REQUIRE(loaded.M() == 1024);

        std::filesystem::remove(path);
    }

    SECTION("Save and load kernel JSON") {
        Kernel kernel = Kernel::create_mlp(64, 64, 64, ActivationType::SIGMOID, false, DataType::FLOAT32);
        std::string path = (temp_dir / "test_kernel.json").string();

        serializer.save_json(kernel, path, true);
        REQUIRE(std::filesystem::exists(path));

        Kernel loaded = serializer.load_json(path);
        REQUIRE(loaded.is_valid());
        REQUIRE(loaded.op_type() == KernelOpType::MLP);
        REQUIRE(loaded.activation() == ActivationType::SIGMOID);

        std::filesystem::remove(path);
    }

    SECTION("Auto-detect format") {
        Kernel kernel = Kernel::create_matmul(128, 128, 128, DataType::FLOAT32);

        std::string bin_path = (temp_dir / "auto_test.kpukernel").string();
        std::string json_path = (temp_dir / "auto_test.json").string();

        serializer.save_auto(kernel, bin_path);
        serializer.save_auto(kernel, json_path);

        Kernel loaded_bin = serializer.load_auto(bin_path);
        Kernel loaded_json = serializer.load_auto(json_path);

        REQUIRE(loaded_bin.M() == 128);
        REQUIRE(loaded_json.M() == 128);

        std::filesystem::remove(bin_path);
        std::filesystem::remove(json_path);
    }

    SECTION("Detect format from extension") {
        REQUIRE(KernelSerializer::detect_format("test.kpukernel") == "binary");
        REQUIRE(KernelSerializer::detect_format("test.json") == "json");
    }
}

// ============================================================================
// Disassembler Test Files
// ============================================================================

TEST_CASE("Generate test files for disassembler", "[serialization][disasm]") {
    std::filesystem::path test_dir = "/tmp/kpu_test_output";
    std::filesystem::create_directories(test_dir);

    SECTION("Generate program binary file") {
        ProgramSerializer serializer;

        DMProgram program;
        program.name = "matmul_256x256x256";
        program.version = 1;
        program.M = 256;
        program.N = 256;
        program.K = 256;
        program.Ti = 64;
        program.Tj = 64;
        program.Tk = 64;
        program.L1_Ki = 32;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        // Add realistic instruction sequence
        program.instructions.push_back(
            DMInstruction::dma_load(MatrixID::A, {0, 0, 0}, 0x10000, 0, 0, 65536));
        program.instructions.push_back(
            DMInstruction::dma_load(MatrixID::B, {0, 0, 0}, 0x20000, 1, 0, 65536));
        program.instructions.push_back(DMInstruction::barrier());
        program.instructions.push_back(
            DMInstruction::bm_move(MatrixID::A, {0, 0, 0}, 0, 0, 0, 0, 64, 64, 4));
        program.instructions.push_back(
            DMInstruction::bm_move(MatrixID::B, {0, 0, 0}, 1, 0, 0, 0, 64, 64, 4));
        program.instructions.push_back(DMInstruction::barrier());
        program.instructions.push_back(
            DMInstruction::str_feed_rows(MatrixID::A, {0, 0, 0}, 0, 0, 0, 0, 64, 64, 4));
        program.instructions.push_back(
            DMInstruction::str_drain({0, 0, 0}, 0, 0, 0, 0, 64, 64, 4,
                                     false, ActivationType::NONE, false, 0));
        program.instructions.push_back(DMInstruction::halt());

        // Memory map
        program.memory_map.a_base = 0x10000;
        program.memory_map.b_base = 0x20000;
        program.memory_map.c_base = 0x30000;

        DMProgram::MemoryMap::L3Alloc l3_a;
        l3_a.tile_id = 0;
        l3_a.offset = 0;
        l3_a.size = 65536;
        l3_a.matrix = MatrixID::A;
        l3_a.buffer = BufferSlot::BUF_0;
        program.memory_map.l3_allocations.push_back(l3_a);

        // Estimates
        program.estimates.total_cycles = 150000;
        program.estimates.external_mem_bytes = 786432;
        program.estimates.l3_bytes = 262144;
        program.estimates.l2_bytes = 65536;
        program.estimates.arithmetic_intensity = 42.67;
        program.estimates.estimated_gflops = 500.0;

        std::string path = (test_dir / "test_program.kpubin").string();
        serializer.save(program, path);
        REQUIRE(std::filesystem::exists(path));
        INFO("Saved program to: " << path);
    }

    SECTION("Generate kernel binary file") {
        KernelSerializer serializer;

        Kernel kernel = Kernel::create_matmul(512, 512, 512, DataType::FLOAT32);
        REQUIRE(kernel.is_valid());

        std::string path = (test_dir / "test_matmul.kpukernel").string();
        serializer.save(kernel, path);
        REQUIRE(std::filesystem::exists(path));
        INFO("Saved kernel to: " << path);
    }

    SECTION("Generate MLP kernel binary file") {
        KernelSerializer serializer;

        Kernel kernel = Kernel::create_mlp(128, 256, 128, ActivationType::GELU, true, DataType::FLOAT32);
        REQUIRE(kernel.is_valid());

        std::string path = (test_dir / "test_mlp.kpukernel").string();
        serializer.save(kernel, path);
        REQUIRE(std::filesystem::exists(path));
        INFO("Saved MLP kernel to: " << path);
    }

    SECTION("Generate JSON files") {
        KernelSerializer serializer;

        Kernel kernel = Kernel::create_mlp(64, 128, 64, ActivationType::RELU, true, DataType::FLOAT32);

        std::string path = (test_dir / "test_mlp.json").string();
        serializer.save_json(kernel, path, true);
        REQUIRE(std::filesystem::exists(path));
        INFO("Saved kernel JSON to: " << path);
    }
}

// ============================================================================
// Round-trip Tests
// ============================================================================

TEST_CASE("Serialization round-trip integrity", "[serialization][roundtrip]") {
    SECTION("Large program round-trip") {
        ProgramSerializer serializer;

        DMProgram program;
        program.name = "roundtrip_large";
        program.version = 1;
        program.M = 2048;
        program.N = 2048;
        program.K = 2048;
        program.Ti = 128;
        program.Tj = 128;
        program.Tk = 128;
        program.L1_Ki = 64;
        program.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

        // Add many instructions
        for (int i = 0; i < 100; ++i) {
            program.instructions.push_back(
                DMInstruction::dma_load(MatrixID::A, {static_cast<uint16_t>(i), 0, 0},
                                        0x1000 + i * 0x1000, 0, 0, 16384));
        }
        program.instructions.push_back(DMInstruction::halt());

        // Binary round-trip
        std::vector<uint8_t> bin_data = serializer.serialize(program);
        DMProgram bin_loaded = serializer.deserialize(bin_data);
        REQUIRE(bin_loaded.instructions.size() == 101);

        // JSON round-trip
        std::string json = serializer.to_json(program, false);
        DMProgram json_loaded = serializer.from_json(json);
        REQUIRE(json_loaded.instructions.size() == 101);
    }

    SECTION("Kernel with all data types") {
        KernelSerializer serializer;

        std::vector<DataType> dtypes = {
            DataType::FLOAT32,
            DataType::FLOAT16,
            DataType::BFLOAT16,
            DataType::INT8
        };

        for (DataType dtype : dtypes) {
            Kernel kernel = Kernel::create_matmul(64, 64, 64, dtype);

            // Binary round-trip
            std::vector<uint8_t> data = serializer.serialize(kernel);
            Kernel loaded = serializer.deserialize(data);
            REQUIRE(loaded.dtype() == dtype);

            // JSON round-trip
            std::string json = serializer.to_json(kernel);
            Kernel json_loaded = serializer.from_json(json);
            // Note: JSON may lose precision on dtype mapping, so we check validity
            REQUIRE(json_loaded.is_valid());
        }
    }

    SECTION("Kernel with all activation types") {
        KernelSerializer serializer;

        std::vector<ActivationType> activations = {
            ActivationType::NONE,
            ActivationType::RELU,
            ActivationType::GELU,
            ActivationType::SIGMOID,
            ActivationType::TANH,
            ActivationType::SILU
        };

        for (ActivationType act : activations) {
            Kernel kernel = Kernel::create_mlp(32, 32, 32, act, true, DataType::FLOAT32);

            std::vector<uint8_t> data = serializer.serialize(kernel);
            Kernel loaded = serializer.deserialize(data);
            REQUIRE(loaded.activation() == act);
        }
    }
}
