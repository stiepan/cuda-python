# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from libc.stdint cimport int64_t, uint32_t, intptr_t
from libcpp cimport vector

ctypedef int64_t extent_t
ctypedef int64_t stride_t
ctypedef int axis_t

ctypedef uint32_t axes_mask_t  # MUST be exactly STRIDED_LAYOUT_MAX_NDIM bits wide
ctypedef uint32_t property_mask_t

ctypedef vector.vector[extent_t] shape_t
ctypedef vector.vector[stride_t] strides_t
ctypedef vector.vector[axis_t] axis_order_t


cdef enum OrderFlag:
    ORDER_NONE = 0
    ORDER_C = 1
    ORDER_F = 2
    ORDER_PERM = 3


cdef enum SliceMask:
    SLICE_PROP_SINGLE_ELEMENT = 1
    SLICE_PROP_START = 2
    SLICE_PROP_STOP = 4
    SLICE_PROP_STEP = 8


ctypedef uint32_t slice_mask_t

cdef struct Slice:
    int64_t start
    int64_t stop
    int64_t step
    slice_mask_t mask


ctypedef vector.vector[Slice] slices_t


@cython.final
cdef class StridedLayout:

    # Defining values
    cdef:
        shape_t shape
        strides_t strides
        
        readonly:
            int itemsize
            stride_t slice_offset

    # Properties that must be set with the defining values
    cdef readonly:
        int ndim
        int64_t volume

    # Lazy properties computed from the defining values.
    cdef:
        # Set to 0 to invalidate all properties, 
        # whenever a defining value is changed
        property_mask_t _prop_mask

        # C and Python properties
        property_mask_t _boolean_props
        int64_t _required_size_in_bytes

        # Python properties
        tuple _py_shape
        tuple _py_strides
        tuple _py_strides_in_bytes
        tuple _py_stride_order

    # C API
    
    # New layout setup
    cdef int init(StridedLayout self, shape_t& shape, strides_t& strides, int itemsize, bint strides_in_bytes=*) except -1 nogil
    cdef stride_t init_dense(StridedLayout self, shape_t& shape, int itemsize, OrderFlag order_flag, axis_order_t* stride_order=*) except -1 nogil
    
    # Layout manipulation
    cdef int reshape_into(StridedLayout self, StridedLayout out_layout, shape_t& shape) except -1 nogil
    cdef int permute_into(StridedLayout self, StridedLayout out_layout, axis_order_t& axis_order) except -1 nogil
    cdef int permute_inplace(StridedLayout self, axis_order_t& axis_order) except -1 nogil
    cdef int flatten_into(StridedLayout self, StridedLayout out_layout, axes_mask_t axis_mask=*) except -1 nogil
    cdef int flatten_inplace(StridedLayout self, axes_mask_t axis_mask=*) except -1 nogil
    cdef int squeeze_into(StridedLayout self, StridedLayout out_layout) except -1 nogil
    cdef int squeeze_inplace(StridedLayout self) except -1 nogil
    cdef int pack_into(StridedLayout self, StridedLayout out_layout, int itemsize, intptr_t data_ptr, bint keep_dim, int axis=*) except -1 nogil
    cdef int pack_inplace(StridedLayout self, int itemsize, intptr_t data_ptr, bint keep_dim, int axis=*) except -1 nogil
    cdef int unpack_into(StridedLayout self, StridedLayout out_layout, int itemsize, int axis=*) except -1 nogil
    cdef int unpack_inplace(StridedLayout self, int itemsize, int axis=*) except -1 nogil
    cdef int slice_into(StridedLayout self, StridedLayout out_layout, slices_t& slices) except -1 nogil
    cdef int slice_inplace(StridedLayout self, slices_t& slices) except -1 nogil

    # Properties
    cdef int get_stride_order(StridedLayout self, axis_order_t& stride_order) except -1 nogil
    cdef int get_strides_in_bytes(StridedLayout self, strides_t& strides) except -1 nogil
    cdef bint get_is_unique(StridedLayout self) except -1 nogil
    cdef bint get_is_contiguous_c(StridedLayout self) except -1 nogil
    cdef bint get_is_contiguous_f(StridedLayout self) except -1 nogil
    cdef bint get_is_contiguous_any(StridedLayout self) except -1 nogil
    cdef int get_offset_bounds(StridedLayout self, stride_t& min_offset, stride_t& max_offset) except -1 nogil
    cdef int64_t get_required_size_in_bytes(StridedLayout self) except -1 nogil
    cdef int64_t get_volume_in_bytes(StridedLayout self) except -1 nogil
    cdef int64_t get_slice_offset_in_bytes(StridedLayout self) except -1 nogil
    cdef axes_mask_t get_flattened_axis_mask(StridedLayout self) except? -1 nogil
    cdef int get_max_compatible_itemsize(StridedLayout self, int max_itemsize, intptr_t base_ptr, int axis=*) except -1 nogil

