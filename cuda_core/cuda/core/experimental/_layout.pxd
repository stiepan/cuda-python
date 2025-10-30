# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from cython.operator cimport dereference as deref

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

from cuda.core.experimental._utils cimport cuda_utils


ctypedef fused vector_t:
    shape_t
    strides_t
    axis_order_t


ctypedef fused integer_t:
    int64_t
    int


cdef extern from "include/layout.hpp":

    cdef int STRIDED_LAYOUT_MAX_NDIM
    cdef int AXIS_MASK_ALL
    int64_t _c_abs(int64_t x) nogil
    void _order_from_strides(axis_order_t& indices, shape_t& shape, strides_t& strides) except + nogil
    void _swap(shape_t &a, shape_t &b) noexcept nogil
    void _swap(strides_t &a, strides_t &b) noexcept nogil
    void _swap(axis_order_t &a, axis_order_t &b) noexcept nogil


cdef enum OrderFlag:
    ORDER_NONE = 0
    ORDER_C = 1
    ORDER_F = 2
    ORDER_PERM = 3


cdef enum Property:
    PROP_IS_UNIQUE = 1 << 0
    PROP_IS_CONTIGUOUS_C = 1 << 1
    PROP_IS_CONTIGUOUS_F = 1 << 2
    PROP_IS_CONTIGUOUS_ANY = 1 << 3
    PROP_REQUIRED_SIZE_IN_BYTES = 1 << 4
    PROP_SHAPE = 1 << 5
    PROP_STRIDES = 1 << 6
    PROP_STRIDES_IN_BYTES = 1 << 7
    PROP_STRIDE_ORDER = 1 << 8


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
    cdef inline int init(StridedLayout self, shape_t& shape, strides_t& strides, int itemsize, bint strides_in_bytes=False) except -1 nogil:
        _setup_shape(self, shape)
        _setup_itemsize(self, itemsize)

        if strides.size() != <size_t>self.ndim:
            raise ValueError("strides must have the same length as shape")
        _swap(self.strides, strides)
        if strides_in_bytes:
            divide_strides(self.strides, self.itemsize)
        return 0
    
    cdef inline stride_t init_dense(StridedLayout self, shape_t& shape, int itemsize, OrderFlag order_flag, axis_order_t* stride_order=NULL) except -1 nogil:
        _setup_shape(self, shape)
        _setup_itemsize(self, itemsize)

        cdef stride_t volume
        if order_flag == ORDER_C:
            volume = _dense_strides_c(self.strides, self.shape)
        elif order_flag == ORDER_F:
            volume = _dense_strides_f(self.strides, self.shape)
        elif order_flag == ORDER_PERM:
            if stride_order == NULL: # should never happen
                raise AssertionError("stride_order is required for ORDER_PERM")
            volume = _dense_strides_in_order(self.strides, self.shape, deref(stride_order))
        else:
            raise ValueError("The stride_order must be 'C', 'F', or a permutation.")
        if volume == 0:
            zeros(self.strides, self.ndim)
        return volume
    
    # Layout manipulation
    cdef int reshape_into(StridedLayout self, StridedLayout out_layout, shape_t& shape) except -1 nogil
    cdef int permute_into(StridedLayout self, StridedLayout out_layout, axis_order_t& axis_order) except -1 nogil
    cdef int flatten_into(StridedLayout self, StridedLayout out_layout, axes_mask_t axis_mask=*) except -1 nogil
    cdef int squeeze_into(StridedLayout self, StridedLayout out_layout) except -1 nogil
    cdef int pack_into(StridedLayout self, StridedLayout out_layout, int itemsize, intptr_t data_ptr, bint keep_dim, int axis=*) except -1 nogil
    cdef int unpack_into(StridedLayout self, StridedLayout out_layout, int itemsize, int axis=*) except -1 nogil
    cdef int slice_into(StridedLayout self, StridedLayout out_layout, tuple slices) except -1

    # Properties
    cdef inline tuple get_shape_tuple(StridedLayout self):
        if not _has_valid_property(self, PROP_SHAPE):
            self._py_shape = cuda_utils.carray_int64_t_to_tuple(self.shape.data(), self.ndim)
            _mark_property_valid(self, PROP_SHAPE)
        return self._py_shape

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


cdef inline bint _has_valid_property(StridedLayout self, Property prop) noexcept nogil:
    return self._prop_mask & prop


cdef inline void _mark_property_valid(StridedLayout self, Property prop) noexcept nogil:
    self._prop_mask |= prop


cdef inline bint _boolean_property(StridedLayout self, Property prop) noexcept nogil:
    return self._boolean_props & prop


cdef inline bint _set_boolean_property(StridedLayout self, Property prop, bint value) noexcept nogil:
    if value:
        self._boolean_props |= prop
    else:
        self._boolean_props &= ~prop
    _mark_property_valid(self, prop)
    return value


cdef inline bint _normalize_axis(integer_t& axis, integer_t extent) except -1 nogil:
    if axis < -extent or axis >= extent:
        return False
    if axis < 0:
        axis += extent
    return True


cdef inline int _setup_shape(StridedLayout layout, shape_t& shape) except -1 nogil:
    cdef int ndim = shape.size()
    if ndim > STRIDED_LAYOUT_MAX_NDIM:
        raise ValueError(f"Unsupported number of dimensions: {ndim}. Max supported ndim is {STRIDED_LAYOUT_MAX_NDIM}")
    for i in range(ndim):
        if shape[i] < 0:
            raise ValueError("Extents must be non-negative")
    layout.volume = _volume(shape)
    layout.ndim = ndim
    _swap(layout.shape, shape)
    return 0


cdef inline int _setup_itemsize(StridedLayout layout, int itemsize) except -1 nogil:
    if itemsize <= 0:
        raise ValueError("itemsize must be positive")
    if itemsize & (itemsize - 1):
        raise ValueError("itemsize must be a power of two")
    layout.itemsize = itemsize
    return 0


@cython.overflowcheck(True)
cdef inline int64_t _volume(shape_t& shape) except? -1 nogil:
    cdef int64_t vol = 1
    for i in range(shape.size()):
        vol *= shape[i]
    return vol


cdef inline int divide_strides(strides_t &strides, int itemsize) except -1 nogil:
    cdef stride_t stride
    for i in range(strides.size()):
        stride = strides[i] // itemsize
        if stride * itemsize != strides[i]:
            raise ValueError("strides must be divisible by itemsize")
        strides[i] = stride
    return 0


cdef inline int zeros(vector_t& vec, int ndim) except -1 nogil:
    vec.clear()
    vec.resize(ndim, 0)
    return 0


cdef inline stride_t _dense_strides_c(strides_t& strides, shape_t& shape) except? -1 nogil:
    cdef int ndim = shape.size()
    strides.resize(ndim)
    cdef stride_t stride = 1
    cdef int i = ndim - 1
    while i >= 0:
        strides[i] = stride
        stride *= shape[i]
        i -= 1
    return stride


cdef inline stride_t _dense_strides_f(strides_t& strides, shape_t& shape) except? -1 nogil:
    cdef int ndim = shape.size()
    strides.clear()
    strides.reserve(ndim)
    cdef stride_t stride = 1
    cdef int i = 0
    while i < ndim:
        strides.push_back(stride)
        stride *= shape[i]
        i += 1
    return stride


cdef inline stride_t _dense_strides_in_order(strides_t& strides, shape_t& shape, axis_order_t& stride_order) except? -1 nogil:
    cdef int ndim = shape.size()
    if <size_t>ndim != stride_order.size():
        raise ValueError(f"stride_order must have the same length as shape. Shape has {ndim} dimensions, but stride_order has {stride_order.size()} elements.")
    strides.resize(ndim)
    cdef stride_t stride = 1
    cdef int i = ndim - 1
    cdef axes_mask_t axis_order_mask = 0
    cdef axes_mask_t axis_mask
    cdef axis_t axis
    while i >= 0:
        axis = stride_order[i]
        if not _normalize_axis(axis, ndim):
            raise ValueError(f"Invalid stride order: axis {axis} out of range for {ndim}D tensor")
        axis_mask = 1 << axis
        if axis_order_mask & axis_mask:
            raise ValueError(f"The stride order must be a permutation. Axis {axis} appears multiple times.")
        axis_order_mask |= axis_mask
        strides[axis] = stride
        stride *= shape[axis]
        i -= 1
    return stride


cdef inline int tuple2vec(vector_t &vec, object t) except -1:
    cdef int ndim = len(t)
    vec.resize(ndim)
    for i in range(ndim):
        vec[i] = t[i]
    return 0
