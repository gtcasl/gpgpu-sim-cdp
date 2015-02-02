#pragma once

#include "cuda-sim/memory.h"

//Class definition for aggregated block group
class agg_block_group_t {
public:

    agg_block_group_t(dim3 agg_dim, 
        dim3 block_dim, kernel_info_t * kernel) {
        m_agg_block_group_dim = agg_dim;
        m_agg_block_dim = block_dim;
        m_kernel = kernel;
        m_param_mem = new memory_space_impl<256>("param", 256);
    }
    dim3 get_agg_dim() const {
        return m_agg_block_group_dim;
    }

    dim3 get_block_dim() const {
        return m_agg_block_dim;
    }

    size_t get_num_blocks() const {
        return m_agg_block_group_dim.x *
            m_agg_block_group_dim.y *
            m_agg_block_group_dim.z;
    }

    class memory_space * get_param_memory() const {
        return m_param_mem;
    }

private:
    dim3 m_agg_block_group_dim; // aggregated block group dim, similar to kernel grid dim
    dim3 m_agg_block_dim; // aggregated block dim
    kernel_info_t * m_kernel; // the native kernel to be aggregated to
    class memory_space * m_param_mem; // parameter buffer

};
