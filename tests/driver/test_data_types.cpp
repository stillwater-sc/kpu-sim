// Test suite for DataType enum and utilities
// Tests data type properties, conversions, and edge cases

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/data_types.hpp>

using namespace sw::kpu;

TEST_CASE("DataType size in bytes", "[data_types]") {
    SECTION("Standard floating point types") {
        REQUIRE(dtype_size(DataType::FLOAT32) == 4);
        REQUIRE(dtype_size(DataType::FLOAT16) == 2);
        REQUIRE(dtype_size(DataType::BFLOAT16) == 2);
    }

    SECTION("Integer types") {
        REQUIRE(dtype_size(DataType::INT32) == 4);
        REQUIRE(dtype_size(DataType::INT8) == 1);
        REQUIRE(dtype_size(DataType::UINT8) == 1);
        REQUIRE(dtype_size(DataType::INT4) == 1);  // Minimum addressable unit
    }
}

TEST_CASE("DataType size in bits", "[data_types]") {
    SECTION("Floating point types") {
        REQUIRE(dtype_bits(DataType::FLOAT32) == 32);
        REQUIRE(dtype_bits(DataType::FLOAT16) == 16);
        REQUIRE(dtype_bits(DataType::BFLOAT16) == 16);
    }

    SECTION("Integer types") {
        REQUIRE(dtype_bits(DataType::INT32) == 32);
        REQUIRE(dtype_bits(DataType::INT8) == 8);
        REQUIRE(dtype_bits(DataType::UINT8) == 8);
        REQUIRE(dtype_bits(DataType::INT4) == 4);
    }
}

TEST_CASE("DataType classification", "[data_types]") {
    SECTION("Integer type detection") {
        REQUIRE(dtype_is_integer(DataType::INT32) == true);
        REQUIRE(dtype_is_integer(DataType::INT8) == true);
        REQUIRE(dtype_is_integer(DataType::UINT8) == true);
        REQUIRE(dtype_is_integer(DataType::INT4) == true);
        REQUIRE(dtype_is_integer(DataType::FLOAT32) == false);
        REQUIRE(dtype_is_integer(DataType::FLOAT16) == false);
        REQUIRE(dtype_is_integer(DataType::BFLOAT16) == false);
    }

    SECTION("Floating point type detection") {
        REQUIRE(dtype_is_floating(DataType::FLOAT32) == true);
        REQUIRE(dtype_is_floating(DataType::FLOAT16) == true);
        REQUIRE(dtype_is_floating(DataType::BFLOAT16) == true);
        REQUIRE(dtype_is_floating(DataType::INT32) == false);
        REQUIRE(dtype_is_floating(DataType::INT8) == false);
        REQUIRE(dtype_is_floating(DataType::UINT8) == false);
        REQUIRE(dtype_is_floating(DataType::INT4) == false);
    }

    SECTION("Signed type detection") {
        REQUIRE(dtype_is_signed(DataType::FLOAT32) == true);
        REQUIRE(dtype_is_signed(DataType::FLOAT16) == true);
        REQUIRE(dtype_is_signed(DataType::INT32) == true);
        REQUIRE(dtype_is_signed(DataType::INT8) == true);
        REQUIRE(dtype_is_signed(DataType::INT4) == true);
        REQUIRE(dtype_is_signed(DataType::UINT8) == false);  // Only unsigned type
    }

    SECTION("Packed type detection") {
        REQUIRE(dtype_is_packed(DataType::INT4) == true);
        REQUIRE(dtype_is_packed(DataType::INT8) == false);
        REQUIRE(dtype_is_packed(DataType::FLOAT32) == false);
    }
}

TEST_CASE("DataType elements per byte", "[data_types]") {
    REQUIRE(dtype_elements_per_byte(DataType::INT4) == 2);
    REQUIRE(dtype_elements_per_byte(DataType::INT8) == 1);
    REQUIRE(dtype_elements_per_byte(DataType::UINT8) == 1);
}

TEST_CASE("Accumulator type mapping", "[data_types]") {
    SECTION("Floating point accumulation") {
        REQUIRE(accumulator_type(DataType::FLOAT32) == DataType::FLOAT32);
        REQUIRE(accumulator_type(DataType::FLOAT16) == DataType::FLOAT32);
        REQUIRE(accumulator_type(DataType::BFLOAT16) == DataType::FLOAT32);
    }

    SECTION("Integer accumulation") {
        REQUIRE(accumulator_type(DataType::INT8) == DataType::INT32);
        REQUIRE(accumulator_type(DataType::UINT8) == DataType::INT32);
        REQUIRE(accumulator_type(DataType::INT4) == DataType::INT32);
        REQUIRE(accumulator_type(DataType::INT32) == DataType::INT32);
    }
}

TEST_CASE("Bytes for elements calculation", "[data_types]") {
    SECTION("Standard types") {
        REQUIRE(dtype_bytes_for_elements(DataType::FLOAT32, 10) == 40);
        REQUIRE(dtype_bytes_for_elements(DataType::FLOAT16, 10) == 20);
        REQUIRE(dtype_bytes_for_elements(DataType::INT8, 10) == 10);
    }

    SECTION("Packed INT4 type") {
        REQUIRE(dtype_bytes_for_elements(DataType::INT4, 2) == 1);  // 2 elements per byte
        REQUIRE(dtype_bytes_for_elements(DataType::INT4, 3) == 2);  // Round up
        REQUIRE(dtype_bytes_for_elements(DataType::INT4, 4) == 2);
        REQUIRE(dtype_bytes_for_elements(DataType::INT4, 5) == 3);  // Round up
    }

    SECTION("Edge cases") {
        REQUIRE(dtype_bytes_for_elements(DataType::FLOAT32, 0) == 0);
        REQUIRE(dtype_bytes_for_elements(DataType::INT4, 0) == 0);
        REQUIRE(dtype_bytes_for_elements(DataType::INT4, 1) == 1);  // Minimum 1 byte
    }
}

TEST_CASE("DataType name conversion", "[data_types]") {
    SECTION("To string") {
        REQUIRE(dtype_name(DataType::FLOAT32) == "float32");
        REQUIRE(dtype_name(DataType::FLOAT16) == "float16");
        REQUIRE(dtype_name(DataType::BFLOAT16) == "bfloat16");
        REQUIRE(dtype_name(DataType::INT32) == "int32");
        REQUIRE(dtype_name(DataType::INT8) == "int8");
        REQUIRE(dtype_name(DataType::UINT8) == "uint8");
        REQUIRE(dtype_name(DataType::INT4) == "int4");
    }

    SECTION("From string - standard names") {
        REQUIRE(dtype_from_name("float32") == DataType::FLOAT32);
        REQUIRE(dtype_from_name("float16") == DataType::FLOAT16);
        REQUIRE(dtype_from_name("bfloat16") == DataType::BFLOAT16);
        REQUIRE(dtype_from_name("int32") == DataType::INT32);
        REQUIRE(dtype_from_name("int8") == DataType::INT8);
        REQUIRE(dtype_from_name("uint8") == DataType::UINT8);
        REQUIRE(dtype_from_name("int4") == DataType::INT4);
    }

    SECTION("From string - aliases") {
        REQUIRE(dtype_from_name("f32") == DataType::FLOAT32);
        REQUIRE(dtype_from_name("float") == DataType::FLOAT32);
        REQUIRE(dtype_from_name("f16") == DataType::FLOAT16);
        REQUIRE(dtype_from_name("half") == DataType::FLOAT16);
        REQUIRE(dtype_from_name("bf16") == DataType::BFLOAT16);
        REQUIRE(dtype_from_name("i32") == DataType::INT32);
        REQUIRE(dtype_from_name("i8") == DataType::INT8);
        REQUIRE(dtype_from_name("u8") == DataType::UINT8);
        REQUIRE(dtype_from_name("i4") == DataType::INT4);
    }

    SECTION("Case insensitivity") {
        REQUIRE(dtype_from_name("FLOAT32") == DataType::FLOAT32);
        REQUIRE(dtype_from_name("Float32") == DataType::FLOAT32);
        REQUIRE(dtype_from_name("INT8") == DataType::INT8);
    }

    SECTION("Invalid name throws") {
        REQUIRE_THROWS_AS(dtype_from_name("invalid"), std::invalid_argument);
        REQUIRE_THROWS_AS(dtype_from_name(""), std::invalid_argument);
    }
}

TEST_CASE("DataType value ranges", "[data_types]") {
    SECTION("Maximum values") {
        REQUIRE(dtype_max_value(DataType::INT8) == Catch::Approx(127.0));
        REQUIRE(dtype_max_value(DataType::UINT8) == Catch::Approx(255.0));
        REQUIRE(dtype_max_value(DataType::INT4) == Catch::Approx(7.0));
        REQUIRE(dtype_max_value(DataType::INT32) == Catch::Approx(2147483647.0));
    }

    SECTION("Minimum values") {
        REQUIRE(dtype_min_value(DataType::INT8) == Catch::Approx(-128.0));
        REQUIRE(dtype_min_value(DataType::UINT8) == Catch::Approx(0.0));
        REQUIRE(dtype_min_value(DataType::INT4) == Catch::Approx(-8.0));
        REQUIRE(dtype_min_value(DataType::INT32) == Catch::Approx(-2147483648.0));
    }

    SECTION("Floating point ranges are sensible") {
        REQUIRE(dtype_max_value(DataType::FLOAT32) > 1e30);
        REQUIRE(dtype_min_value(DataType::FLOAT32) < -1e30);
        REQUIRE(dtype_max_value(DataType::FLOAT16) > 60000);
        REQUIRE(dtype_min_value(DataType::FLOAT16) < -60000);
    }
}

TEST_CASE("DataType constexpr correctness", "[data_types]") {
    // All utility functions should be constexpr
    constexpr Size float32_size = dtype_size(DataType::FLOAT32);
    constexpr Size int8_bits = dtype_bits(DataType::INT8);
    constexpr bool int4_is_int = dtype_is_integer(DataType::INT4);
    constexpr bool float32_is_float = dtype_is_floating(DataType::FLOAT32);
    constexpr DataType int8_acc = accumulator_type(DataType::INT8);
    constexpr Size int4_bytes = dtype_bytes_for_elements(DataType::INT4, 10);

    REQUIRE(float32_size == 4);
    REQUIRE(int8_bits == 8);
    REQUIRE(int4_is_int == true);
    REQUIRE(float32_is_float == true);
    REQUIRE(int8_acc == DataType::INT32);
    REQUIRE(int4_bytes == 5);
}
