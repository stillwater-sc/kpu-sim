// ============================================================================
// src/models/behavioral/datamovement/block_mover.cpp
// Behavioral BlockMover implementation
// ============================================================================

#include <sw/kpu/behavioral/block_mover.hpp>
#include <stdexcept>
#include <cstring>

namespace sw::kpu::behavioral {

void BehavioralBlockMover::transfer(const TransferDescriptor& desc, BehavioralMemoryModel* memory) {
    if (!memory) {
        throw std::invalid_argument("BehavioralBlockMover::transfer: memory is null");
    }

    // Validate source/destination types
    if (desc.src_type != MemoryRegionType::L3_TILE && desc.src_type != MemoryRegionType::L2_BANK) {
        throw std::invalid_argument("BehavioralBlockMover: source must be L3_TILE or L2_BANK");
    }
    if (desc.dst_type != MemoryRegionType::L3_TILE && desc.dst_type != MemoryRegionType::L2_BANK) {
        throw std::invalid_argument("BehavioralBlockMover: destination must be L3_TILE or L2_BANK");
    }

    // Get source and destination regions
    MemoryRegion* src_region = memory->get_region(desc.src_type, desc.src_region_id);
    MemoryRegion* dst_region = memory->get_region(desc.dst_type, desc.dst_region_id);

    if (!src_region) {
        throw std::runtime_error("BehavioralBlockMover: source region not found");
    }
    if (!dst_region) {
        throw std::runtime_error("BehavioralBlockMover: destination region not found");
    }

    // Compute strides (0 = contiguous)
    uint32_t row_bytes = desc.width * desc.element_size;
    uint32_t src_stride = (desc.src_stride > 0) ? desc.src_stride : row_bytes;
    uint32_t dst_stride = (desc.dst_stride > 0) ? desc.dst_stride : row_bytes;

    // Perform the transfer
    if (desc.transpose) {
        // Destination stride for transposed data: width becomes height
        uint32_t transposed_row_bytes = desc.height * desc.element_size;
        uint32_t transposed_dst_stride = (desc.dst_stride > 0) ? desc.dst_stride : transposed_row_bytes;

        copy_block_transpose(
            src_region, desc.src_offset, src_stride,
            dst_region, desc.dst_offset, transposed_dst_stride,
            desc.height, desc.width, desc.element_size);

        stats_.transpose_count++;
    } else {
        copy_block(
            src_region, desc.src_offset, src_stride,
            dst_region, desc.dst_offset, dst_stride,
            desc.height, desc.width, desc.element_size);
    }

    // Update statistics
    stats_.transfers++;
    stats_.bytes_moved += static_cast<Size>(desc.height) * desc.width * desc.element_size;

    if (desc.src_type == MemoryRegionType::L3_TILE && desc.dst_type == MemoryRegionType::L2_BANK) {
        stats_.l3_to_l2_transfers++;
    } else if (desc.src_type == MemoryRegionType::L2_BANK && desc.dst_type == MemoryRegionType::L3_TILE) {
        stats_.l2_to_l3_transfers++;
    }
}

void BehavioralBlockMover::l3_to_l2(
    uint32_t l3_tile_id, Size l3_offset,
    uint32_t l2_bank_id, Size l2_offset,
    uint32_t height, uint32_t width, uint32_t element_size,
    BehavioralMemoryModel* memory,
    bool transpose)
{
    TransferDescriptor desc;
    desc.src_type = MemoryRegionType::L3_TILE;
    desc.src_region_id = l3_tile_id;
    desc.src_offset = l3_offset;
    desc.dst_type = MemoryRegionType::L2_BANK;
    desc.dst_region_id = l2_bank_id;
    desc.dst_offset = l2_offset;
    desc.height = height;
    desc.width = width;
    desc.element_size = element_size;
    desc.src_stride = 0;
    desc.dst_stride = 0;
    desc.transpose = transpose;

    transfer(desc, memory);
}

void BehavioralBlockMover::l2_to_l3(
    uint32_t l2_bank_id, Size l2_offset,
    uint32_t l3_tile_id, Size l3_offset,
    uint32_t height, uint32_t width, uint32_t element_size,
    BehavioralMemoryModel* memory)
{
    TransferDescriptor desc;
    desc.src_type = MemoryRegionType::L2_BANK;
    desc.src_region_id = l2_bank_id;
    desc.src_offset = l2_offset;
    desc.dst_type = MemoryRegionType::L3_TILE;
    desc.dst_region_id = l3_tile_id;
    desc.dst_offset = l3_offset;
    desc.height = height;
    desc.width = width;
    desc.element_size = element_size;
    desc.src_stride = 0;
    desc.dst_stride = 0;
    desc.transpose = false;

    transfer(desc, memory);
}

void BehavioralBlockMover::l3_to_l2_strided(
    uint32_t l3_tile_id, Size l3_offset, uint32_t l3_stride,
    uint32_t l2_bank_id, Size l2_offset,
    uint32_t height, uint32_t width, uint32_t element_size,
    BehavioralMemoryModel* memory,
    bool transpose)
{
    TransferDescriptor desc;
    desc.src_type = MemoryRegionType::L3_TILE;
    desc.src_region_id = l3_tile_id;
    desc.src_offset = l3_offset;
    desc.dst_type = MemoryRegionType::L2_BANK;
    desc.dst_region_id = l2_bank_id;
    desc.dst_offset = l2_offset;
    desc.height = height;
    desc.width = width;
    desc.element_size = element_size;
    desc.src_stride = l3_stride;
    desc.dst_stride = 0;  // Destination is contiguous
    desc.transpose = transpose;

    transfer(desc, memory);
}

void BehavioralBlockMover::copy_block(
    MemoryRegion* src_region, Size src_offset, uint32_t src_stride,
    MemoryRegion* dst_region, Size dst_offset, uint32_t dst_stride,
    uint32_t height, uint32_t width, uint32_t element_size)
{
    uint32_t row_bytes = width * element_size;

    // If both are contiguous and strides match row bytes, use single memcpy
    if (src_stride == row_bytes && dst_stride == row_bytes) {
        Size total_bytes = static_cast<Size>(height) * row_bytes;
        const uint8_t* src_ptr = src_region->data_ptr(src_offset);
        uint8_t* dst_ptr = dst_region->data_ptr(dst_offset);
        std::memcpy(dst_ptr, src_ptr, total_bytes);
    } else {
        // Row-by-row copy for strided access
        for (uint32_t row = 0; row < height; ++row) {
            Size src_row_offset = src_offset + static_cast<Size>(row) * src_stride;
            Size dst_row_offset = dst_offset + static_cast<Size>(row) * dst_stride;

            const uint8_t* src_ptr = src_region->data_ptr(src_row_offset);
            uint8_t* dst_ptr = dst_region->data_ptr(dst_row_offset);
            std::memcpy(dst_ptr, src_ptr, row_bytes);
        }
    }
}

void BehavioralBlockMover::copy_block_transpose(
    MemoryRegion* src_region, Size src_offset, uint32_t src_stride,
    MemoryRegion* dst_region, Size dst_offset, uint32_t dst_stride,
    uint32_t height, uint32_t width, uint32_t element_size)
{
    // Transpose: src[row, col] -> dst[col, row]
    // Source dimensions: height x width
    // Destination dimensions: width x height

    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            // Source position: row * src_stride + col * element_size
            Size src_elem_offset = src_offset +
                static_cast<Size>(row) * src_stride +
                static_cast<Size>(col) * element_size;

            // Destination position: col * dst_stride + row * element_size
            Size dst_elem_offset = dst_offset +
                static_cast<Size>(col) * dst_stride +
                static_cast<Size>(row) * element_size;

            const uint8_t* src_ptr = src_region->data_ptr(src_elem_offset);
            uint8_t* dst_ptr = dst_region->data_ptr(dst_elem_offset);

            std::memcpy(dst_ptr, src_ptr, element_size);
        }
    }
}

} // namespace sw::kpu::behavioral
