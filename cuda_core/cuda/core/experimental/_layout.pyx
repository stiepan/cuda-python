# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from cython.operator cimport dereference as deref

from libc.stdint cimport int64_t, uint32_t, intptr_t
from libcpp cimport vector


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
    int64_t c_abs(int64_t x) nogil
    void order_from_strides(axis_order_t& indices, shape_t& shape, strides_t& strides) except + nogil
    void swap(shape_t &a, shape_t &b) noexcept nogil
    void swap(strides_t &a, strides_t &b) noexcept nogil
    void swap(axis_order_t &a, axis_order_t &b) noexcept nogil


cdef enum Property:
    PROP_IS_UNIQUE = 0
    PROP_IS_CONTIGUOUS_C = 1
    PROP_IS_CONTIGUOUS_F = 2
    PROP_IS_CONTIGUOUS_ANY = 3
    PROP_REQUIRED_SIZE_IN_BYTES = 4
    PROP_SHAPE = 5
    PROP_STRIDES = 6
    PROP_STRIDES_IN_BYTES = 7
    PROP_STRIDE_ORDER = 8


@cython.final
cdef class StridedLayout:

    def __init__(StridedLayout self, object shape, object strides, int itemsize, bint strides_in_bytes=False):
        cdef shape_t shape_vec
        cdef strides_t strides_vec
        tuple2vec(shape_vec, shape)
        tuple2vec(strides_vec, strides)
        self.init(shape_vec, strides_vec, itemsize, strides_in_bytes)

    def __repr__(StridedLayout self):
        if self.slice_offset == 0:
            return (
                f"StridedLayout(shape={tuple(self.shape)}, strides={tuple(self.strides)}, itemsize={self.itemsize})"
            )
        else:
            return (
                f"StridedLayout(shape={tuple(self.shape)}, strides={tuple(self.strides)}, itemsize={self.itemsize}, slice_offset={self.slice_offset})"
            )

    def __eq__(StridedLayout self, StridedLayout other):
        return self.itemsize == other.itemsize and self.slice_offset == other.slice_offset and self.strides == other.strides and self.shape == other.shape
    
    @classmethod
    def dense(cls, object shape, int itemsize, object stride_order='C'):
        cdef StridedLayout new_layout = StridedLayout.__new__(cls)
        cdef shape_t shape_vec
        tuple2vec(shape_vec, shape)

        cdef axis_order_t stride_order_vec
        cdef OrderFlag order_flag = stride_order2vec(stride_order_vec, stride_order)
        
        if order_flag == ORDER_NONE:
            raise ValueError(f"The stride_order must be 'C', 'F', or a permutation tuple. Got: {stride_order}")

        new_layout.init_dense(shape_vec, itemsize, order_flag, &stride_order_vec)
        return new_layout
    
    @classmethod
    def dense_like(cls, StridedLayout other, object stride_order="K"):
        cdef StridedLayout new_layout = StridedLayout.__new__(cls)
        cdef axis_order_t stride_order_vec

        if stride_order == "K":
            order_from_strides(stride_order_vec, other.shape, other.strides)
            new_layout.init_dense(other.shape, other.itemsize, ORDER_PERM, &stride_order_vec)
            return new_layout
        
        cdef OrderFlag order_flag = stride_order2vec(stride_order_vec, stride_order)
        if order_flag == ORDER_NONE:
            raise ValueError(f"The stride_order must be 'K', 'C', 'F', or a permutation tuple. Got: {stride_order}")

        new_layout.init_dense(other.shape, other.itemsize, order_flag, &stride_order_vec)
        return new_layout

    def reshaped(self, object shape):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        cdef shape_t shape_vec
        tuple2vec(shape_vec, shape)
        self.reshape_into(new_layout, shape_vec)
        return new_layout

    def permuted(self, object axis_order):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        cdef axis_order_t axis_order_vec
        tuple2vec(axis_order_vec, axis_order)
        self.permute_into(new_layout, axis_order_vec)
        return new_layout
    
    def _permute(self, object axis_order):
        # TODO(ktokarski) Remove me, Python API should not be able
        # to mutate the layout in place
        cdef axis_order_t axis_order_vec
        tuple2vec(axis_order_vec, axis_order)
        self.permute_inplace(axis_order_vec)
        return self

    def flattened(self, start_axis=0, end_axis=-1, mask=None):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        cdef axes_mask_t axis_mask = mask if mask is not None else axis_mask_from_range(self.ndim, start_axis, end_axis)
        self.flatten_into(new_layout, axis_mask)
        return new_layout
    
    def _flatten(self, start_axis=0, end_axis=-1, mask=None):
        cdef axes_mask_t axis_mask = mask if mask is not None else axis_mask_from_range(self.ndim, start_axis, end_axis)
        self.flatten_inplace(axis_mask)
        return self
    
    def flattened_axis_mask(self):
        return self.get_flattened_axis_mask()
    
    def squeezed(self):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        self.squeeze_into(new_layout)
        return new_layout
    
    def _squeeze(self):
        self.squeeze_inplace()
        return self
    
    def packed(self, int itemsize, intptr_t data_ptr=0, int axis=-1, bint keep_dim=True):
        if itemsize == self.itemsize:
            return self
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        self.pack_into(new_layout, itemsize, data_ptr, keep_dim, axis)
        return new_layout
    
    def _pack(self, int itemsize, intptr_t data_ptr=0, int axis=-1, bint keep_dim=True):
        if itemsize != self.itemsize:
            self.pack_inplace(itemsize, data_ptr, keep_dim, axis)
        return self
    
    def unpacked(self, int itemsize, int axis=-1):
        if itemsize == self.itemsize:
            return self
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        self.unpack_into(new_layout, itemsize, axis)
        return new_layout
    
    def _unpack(self, int itemsize, int axis=-1):
        self.unpack_inplace(itemsize, axis)
        return self
    
    def max_compatible_itemsize(self, int max_itemsize=16, intptr_t data_ptr=0, int axis=-1):
        return self.get_max_compatible_itemsize(max_itemsize, data_ptr, axis)
    
    def sliced(self, object slices):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        cdef slices_t slices_vec
        slices2slices_t(slices_vec, slices)
        self.slice_into(new_layout, slices_vec)
        return new_layout
    
    def _slice(self, object slices):
        cdef slices_t slices_vec
        slices2slices_t(slices_vec, slices)
        self.slice_inplace(slices_vec)
        return self
    
    def __getitem__(StridedLayout self, object slices):
        return self.sliced(slices)    
    
    @property
    def is_unique(StridedLayout self):
        return self.get_is_unique()
    
    @property
    def is_contiguous_c(StridedLayout self):
        return self.get_is_contiguous_c()
    
    @property
    def is_contiguous_f(StridedLayout self):
        return self.get_is_contiguous_f()
    
    @property
    def is_contiguous_any(StridedLayout self):
        return self.get_is_contiguous_any()
    
    @property
    def offset_bounds(StridedLayout self):
        cdef stride_t min_offset = 0
        cdef stride_t max_offset = 0
        self.get_offset_bounds(min_offset, max_offset)
        return min_offset, max_offset

    @property
    def required_size_in_bytes(StridedLayout self):
        return self.get_required_size_in_bytes()
    
    @property
    def slice_offset_in_bytes(StridedLayout self):
        return self.get_slice_offset_in_bytes()
    
    @property
    def shape(StridedLayout self):
        if not has_valid_property(self, PROP_SHAPE):
            self._py_shape = tuple(self.shape)
            mark_property_valid(self, PROP_SHAPE)
        return self._py_shape
    
    @property
    def strides(StridedLayout self):
        if not has_valid_property(self, PROP_STRIDES):
            self._py_strides = tuple(self.strides)
            mark_property_valid(self, PROP_STRIDES)
        return self._py_strides
    
    @property
    def strides_in_bytes(StridedLayout self):
        if has_valid_property(self, PROP_STRIDES_IN_BYTES):
            return self._py_strides_in_bytes
        cdef strides_t strides_in_bytes
        self.get_strides_in_bytes(strides_in_bytes)
        self._py_strides_in_bytes = tuple(strides_in_bytes)
        mark_property_valid(self, PROP_STRIDES_IN_BYTES)
        return self._py_strides_in_bytes
    
    @property
    def stride_order(StridedLayout self):
        if has_valid_property(self, PROP_STRIDE_ORDER):
            return self._py_stride_order
        cdef axis_order_t stride_order
        self.get_stride_order(stride_order)
        self._py_stride_order = tuple(stride_order)
        mark_property_valid(self, PROP_STRIDE_ORDER)
        return self._py_stride_order
    
    # ==============================
    # C API
    # ==============================

    cdef int init(StridedLayout self, shape_t& shape, strides_t& strides, int itemsize, bint strides_in_bytes=False) except -1 nogil:
        setup_shape(self, shape)
        setup_itemsize(self, itemsize)

        if strides.size() != <size_t>self.ndim:
            raise ValueError("strides must have the same length as shape")
        swap(self.strides, strides)
        if strides_in_bytes:
            divide_strides(self.strides, self.itemsize)
        return 0
    
    cdef stride_t init_dense(StridedLayout self, shape_t& shape, int itemsize, OrderFlag order_flag, axis_order_t* stride_order=NULL) except -1 nogil:
        setup_shape(self, shape)
        setup_itemsize(self, itemsize)

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
    
    cdef int reshape_into(StridedLayout self, StridedLayout out_layout, shape_t& shape) except -1 nogil:
       # Reset all memoized properties
        out_layout._prop_mask = 0

        # Copy preserved attributes
        out_layout.slice_offset = self.slice_offset
        out_layout.itemsize = self.itemsize

        setup_reshaped_shape(out_layout, shape, self.volume)
        zeros(out_layout.strides, out_layout.ndim)

        if out_layout.volume != self.volume:
            raise ValueError("The new shape has different volume from the reshaped layout. The new volume is {out_layout.volume} and the original volume is {self.volume}.")
        elif out_layout.volume == 0:
            return 0
        cdef shape_t flattened_shape
        cdef strides_t flattened_strides
        flatten_strides_in_c_index_order(flattened_shape, flattened_strides, self.shape, self.strides, AXIS_MASK_ALL)
        if not split_strides_in_c_index_order(out_layout.shape, out_layout.strides, flattened_shape, flattened_strides):
            raise ValueError("Layout strides are incompatible with the new shape")
        return 0
    
    cdef int permute_into(StridedLayout self, StridedLayout out_layout, axis_order_t& axis_order) except -1 nogil:
        if axis_order.size() != <size_t>self.ndim:
            raise ValueError(f"Permutation must have the same length as the number of dimensions, got {axis_order.size()} for {self.ndim}D tensor.")

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize
        out_layout.ndim = self.ndim
        out_layout.volume = self.volume
        out_layout.slice_offset = self.slice_offset

        permute_extents(out_layout.shape, out_layout.strides, self.shape, self.strides, axis_order)
        return 0
    
    cdef int permute_inplace(StridedLayout self, axis_order_t& axis_order) except -1 nogil:
        if axis_order.size() != <size_t>self.ndim:
            raise ValueError(f"Permutation must have the same length as the number of dimensions, got {axis_order.size()} for {self.ndim}D tensor.")
        
        # Reset all memoized properties
        self._prop_mask = 0

        cdef shape_t new_shape
        cdef strides_t new_strides
        permute_extents(new_shape, new_strides, self.shape, self.strides, axis_order)
        swap(self.shape, new_shape)
        swap(self.strides, new_strides)
    
    cdef int flatten_into(StridedLayout self, StridedLayout out_layout, axes_mask_t axis_mask=AXIS_MASK_ALL) except -1 nogil:
        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize
        out_layout.volume = self.volume
        out_layout.slice_offset = self.slice_offset

        out_layout.ndim = flatten_strides_in_c_index_order(out_layout.shape, out_layout.strides, self.shape, self.strides, axis_mask)
        return 0
    
    cdef int flatten_inplace(StridedLayout self, axes_mask_t axis_mask=AXIS_MASK_ALL) except -1 nogil:
        # Reset all memoized properties
        self._prop_mask = 0

        self.ndim = flatten_strides_in_c_index_order(self.shape, self.strides, self.shape, self.strides, axis_mask)
        return 0
    
    cdef int squeeze_into(StridedLayout self, StridedLayout out_layout) except -1 nogil:
        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize
        out_layout.volume = self.volume
        out_layout.slice_offset = self.slice_offset

        out_layout.ndim = squeeze_extents(out_layout.shape, out_layout.strides, self.shape, self.strides)
        return 0
    
    cdef int squeeze_inplace(StridedLayout self) except -1 nogil:
        # Reset all memoized properties
        self._prop_mask = 0

        self.ndim = squeeze_extents(self.shape, self.strides, self.shape, self.strides)
        return 0

    cdef int pack_into(StridedLayout self, StridedLayout out_layout, int itemsize, intptr_t data_ptr, bint keep_dim, int axis=-1) except -1 nogil:
        # Reset all memoized properties
        out_layout._prop_mask = 0

        cdef int vec_size = pack_extents(
            out_layout.slice_offset,
            out_layout.shape,
            out_layout.strides,
            self.slice_offset,
            self.shape,
            self.strides,
            self.itemsize,
            itemsize,
            data_ptr,
            keep_dim,
            axis
        )
        if vec_size > 1:
            out_layout.itemsize = itemsize
            out_layout.volume = self.volume // vec_size
            out_layout.ndim = out_layout.shape.size()
        else:
            copy_layout(out_layout, self)
        return vec_size

    cdef int pack_inplace(StridedLayout self, int itemsize, intptr_t data_ptr, bint keep_dim, int axis=-1) except -1 nogil:
        """
        Vectorizes the layout: i.e. multiplies the itemsize by vec_size
        and divides the strides and last extent by the vector size.
        """
        # Reset all memoized properties
        self._prop_mask = 0

        cdef stride_t new_slice_offset = 0
        cdef shape_t new_shape
        cdef strides_t new_strides
        cdef int vec_size = pack_extents(
            new_slice_offset,
            new_shape,
            new_strides,
            self.slice_offset,
            self.shape,
            self.strides,
            self.itemsize,
            itemsize,
            data_ptr,
            keep_dim,
            axis
        )
        if vec_size > 1:
            self.itemsize = itemsize
            self.slice_offset = new_slice_offset
            self.volume = self.volume // vec_size
            self.ndim = new_shape.size()
            swap(self.shape, new_shape)
            swap(self.strides, new_strides)
        return vec_size
    
    cdef int unpack_into(StridedLayout self, StridedLayout out_layout, int itemsize, int axis=-1) except -1 nogil:
        cdef int vec_size = unpack_extents(
            out_layout.shape,
            out_layout.strides,
            self.shape,
            self.strides,
            self.itemsize,
            itemsize,
            axis
        )
        if vec_size > 1:
            out_layout.itemsize = itemsize
            out_layout.volume = overflow_checked_mul(self.volume, vec_size)
            out_layout.slice_offset = overflow_checked_mul(self.slice_offset, vec_size)
            out_layout.ndim = out_layout.shape.size()
        else:
            copy_layout(out_layout, self)
        return vec_size
    
    cdef int unpack_inplace(StridedLayout self, int itemsize, int axis=-1) except -1 nogil:
        cdef shape_t new_shape
        cdef strides_t new_strides
        cdef stride_t new_slice_offset = 0
        cdef int64_t new_volume = 0
        cdef int vec_size = unpack_extents(
            new_shape,
            new_strides,
            self.shape,
            self.strides,
            self.itemsize,
            itemsize,
            axis
        )
        if vec_size > 1:
            new_slice_offset = overflow_checked_mul(self.slice_offset, vec_size)
            new_volume = overflow_checked_mul(self.volume, vec_size)
            self.itemsize = itemsize
            self.volume = new_volume
            self.slice_offset = new_slice_offset
            self.ndim = new_shape.size()
            swap(self.shape, new_shape)
            swap(self.strides, new_strides)
        return vec_size
    
    cdef int slice_into(StridedLayout self, StridedLayout out_layout, slices_t& slices) except -1 nogil:
        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize

        out_layout.slice_offset = self.slice_offset
        out_layout.slice_offset += slice_extents(out_layout.shape, out_layout.strides, self.shape, self.strides, slices)
        out_layout.volume = volume(out_layout.shape)
        out_layout.ndim = out_layout.shape.size()
        return 0
    
    cdef int slice_inplace(StridedLayout self, slices_t& slices) except -1 nogil:
        # Reset all memoized properties
        self._prop_mask = 0

        cdef shape_t new_shape
        cdef strides_t new_strides
        self.slice_offset += slice_extents(new_shape, new_strides, self.shape, self.strides, slices)
        self.volume = volume(new_shape)
        swap(self.shape, new_shape)
        swap(self.strides, new_strides)
        self.ndim = self.shape.size()
        return 0

    cdef int get_stride_order(StridedLayout self, axis_order_t& stride_order) except -1 nogil:
        order_from_strides(stride_order, self.shape, self.strides)
        return 0
    
    cdef int get_strides_in_bytes(StridedLayout self, strides_t& strides) except -1 nogil:
        strides_in_bytes(strides, self.strides, self.itemsize)
        return 0
    
    cdef bint get_is_unique(StridedLayout self) except -1 nogil:
        if has_valid_property(self, PROP_IS_UNIQUE):
            return boolean_property(self, PROP_IS_UNIQUE)
        cdef axis_order_t stride_order
        self.get_stride_order(stride_order)
        return set_boolean_property(self, PROP_IS_UNIQUE, is_unique(self.shape, self.strides, stride_order))

    cdef bint get_is_contiguous_c(StridedLayout self) except -1 nogil:
        if has_valid_property(self, PROP_IS_CONTIGUOUS_C):
            return boolean_property(self, PROP_IS_CONTIGUOUS_C)
        return set_boolean_property(self, PROP_IS_CONTIGUOUS_C, is_contiguous_c(self.shape, self.strides))

    cdef bint get_is_contiguous_f(StridedLayout self) except -1 nogil:
        if has_valid_property(self, PROP_IS_CONTIGUOUS_F):
            return boolean_property(self, PROP_IS_CONTIGUOUS_F)
        return set_boolean_property(self, PROP_IS_CONTIGUOUS_F, is_contiguous_f(self.shape, self.strides))

    cdef bint get_is_contiguous_any(StridedLayout self) except -1 nogil:
        if has_valid_property(self, PROP_IS_CONTIGUOUS_ANY):
            return boolean_property(self, PROP_IS_CONTIGUOUS_ANY)
        cdef axis_order_t stride_order
        self.get_stride_order(stride_order)
        return set_boolean_property(self, PROP_IS_CONTIGUOUS_ANY, is_contiguous_in_order(self.shape, self.strides, stride_order))
    
    cdef int get_offset_bounds(StridedLayout self, stride_t& min_offset, stride_t& max_offset) except -1 nogil:
        min_offset = 0
        max_offset = 0
        offset_bounds(min_offset, max_offset, self.shape, self.strides)
        return 0
    
    cdef int64_t get_required_size_in_bytes(StridedLayout self) except -1 nogil:
        if not has_valid_property(self, PROP_REQUIRED_SIZE_IN_BYTES):
            self._required_size_in_bytes = required_size_in_bytes(self.slice_offset, self.shape, self.strides)
            mark_property_valid(self, PROP_REQUIRED_SIZE_IN_BYTES)
        return self._required_size_in_bytes
    
    cdef int64_t get_volume_in_bytes(StridedLayout self) except -1 nogil:
        return overflow_checked_mul(self.volume, self.itemsize)
    
    cdef int64_t get_slice_offset_in_bytes(StridedLayout self) except -1 nogil:
        return overflow_checked_mul(self.slice_offset, self.itemsize)

    cdef axes_mask_t get_flattened_axis_mask(StridedLayout self) except? -1 nogil:
        return flattened_strides_in_c_index_order_mask(self.shape, self.strides)
    
    cdef int get_max_compatible_itemsize(StridedLayout self, int max_itemsize, intptr_t data_ptr, int axis=-1) except -1 nogil:
        return max_compatible_itemsize(self.slice_offset, self.itemsize, self.shape, self.strides, max_itemsize, axis, data_ptr)


# ==============================
# Implementation details - StridedLayout class helpers
# ==============================

cdef inline bint has_valid_property(StridedLayout self, Property prop) except -1 nogil:
    return self._prop_mask & (1 << prop)


cdef inline bint mark_property_valid(StridedLayout self, Property prop) except -1 nogil:
    self._prop_mask |= 1 << prop
    return 0


cdef inline bint boolean_property(StridedLayout self, Property prop) except -1 nogil:
    return self._boolean_props & (1 << prop)


cdef inline bint set_boolean_property(StridedLayout self, Property prop, bint value) except -1 nogil:
    if value:
        self._boolean_props |= 1 << prop
    else:
        self._boolean_props &= ~(1 << prop)
    mark_property_valid(self, prop)
    return value


cdef inline int setup_shape(StridedLayout layout, shape_t& shape) except -1 nogil:
    cdef int ndim = shape.size()
    if ndim > STRIDED_LAYOUT_MAX_NDIM:
        raise ValueError(f"Unsupported number of dimensions: {ndim}. Max supported ndim is {STRIDED_LAYOUT_MAX_NDIM}")
    for i in range(ndim):
        if shape[i] < 0:
            raise ValueError("Extents must be non-negative")
    layout.volume = volume(shape)
    layout.ndim = ndim
    swap(layout.shape, shape)
    return 0


cdef inline int setup_reshaped_shape(StridedLayout layout, shape_t& shape, int64_t previous_volume) except -1 nogil:
    cdef int ndim = shape.size()
    if ndim > STRIDED_LAYOUT_MAX_NDIM:
        raise ValueError(f"Unsupported number of dimensions: {ndim}. Max supported ndim is {STRIDED_LAYOUT_MAX_NDIM}")
    cdef int axis = -1
    cdef extent_t extent
    cdef int64_t new_volume = 1
    for i in range(ndim):
        extent = shape[i]
        if extent < -1:
            raise ValueError("Extents must be non-negative")
        elif extent == -1:
            if axis == -1:
                axis = i
            else:
                raise ValueError("There can be at most one -1 extent in a shape")
    new_volume = c_abs(volume(shape))
    if axis != -1:
        extent = previous_volume // new_volume
        if extent * new_volume != previous_volume:
            raise ValueError(f"The original volume {previous_volume} must be divisible by the specified sub-volume {new_volume}.")
        shape[axis] = extent
    elif new_volume != previous_volume:
        raise ValueError(f"The original volume {previous_volume} and the new volume {new_volume} must be equal.")
    layout.volume = previous_volume
    layout.ndim = ndim
    swap(layout.shape, shape)
    return 0


cdef inline int setup_itemsize(StridedLayout layout, int itemsize) except -1 nogil:
    if itemsize <= 0:
        raise ValueError("itemsize must be positive")
    if itemsize & (itemsize - 1):
        raise ValueError("itemsize must be a power of two")
    layout.itemsize = itemsize
    return 0


cdef inline int copy_layout(StridedLayout out_layout, StridedLayout in_layout) except -1 nogil:
    out_layout._prop_mask = 0
    out_layout.itemsize = in_layout.itemsize
    out_layout.slice_offset = in_layout.slice_offset
    out_layout.volume = in_layout.volume
    out_layout.ndim = in_layout.ndim
    out_layout.shape = in_layout.shape
    out_layout.strides = in_layout.strides
    return 0


# ==============================
# Implementation details - python <-> C conversions
# ==============================

cdef inline int tuple2vec(vector_t &vec, object t) except -1:
    cdef int ndim = len(t)
    vec.clear()
    vec.reserve(ndim)
    for i in range(ndim):
        vec.push_back(t[i])
    return 0


@cython.overflowcheck(True)
cdef inline bint normalize_axis(integer_t& axis, integer_t extent) except -1 nogil:
    if axis < -extent or axis >= extent:
        return False
    if axis < 0:
        axis += extent
    return True


@cython.overflowcheck(True)
cdef inline int64_t div_ceil(int64_t a, int64_t b) except? -1 nogil:
    return (a + b - 1) // b


cdef inline OrderFlag stride_order2vec(axis_order_t& stride_order_vec, object stride_order) except? ORDER_NONE:
    if stride_order == 'C':
        return ORDER_C
    elif stride_order == 'F':
        return ORDER_F
    elif isinstance(stride_order, tuple | list): 
        tuple2vec(stride_order_vec, stride_order)
        return ORDER_PERM
    return ORDER_NONE


cdef inline axes_mask_t axis_mask_from_range(int ndim, int start_axis, int end_axis) except? -1 nogil:
    cdef axes_mask_t axis_mask = AXIS_MASK_ALL
    if not normalize_axis(start_axis, ndim):
        raise ValueError(f"Invalid start axis: {start_axis} out of range for {ndim}D tensor")
    if not normalize_axis(end_axis, ndim):
        raise ValueError(f"Invalid end axis: {end_axis} out of range for {ndim}D tensor")
    if start_axis > 0:
        axis_mask &= (AXIS_MASK_ALL << start_axis + 1)
    if end_axis < ndim:
        axis_mask &= (AXIS_MASK_ALL >> (STRIDED_LAYOUT_MAX_NDIM - end_axis - 1))
    return axis_mask


cdef inline int slice2slice_struct(Slice& c_slice, object py_slice) except -1:
    c_slice.mask = 0
    if isinstance(py_slice, int):
        c_slice.mask = SLICE_PROP_SINGLE_ELEMENT
        c_slice.start = py_slice
        return 0
    elif isinstance(py_slice, slice):
        if py_slice.start is not None:
            c_slice.start = py_slice.start
            c_slice.mask |= SLICE_PROP_START
        if py_slice.stop is not None:
            c_slice.stop = py_slice.stop
            c_slice.mask |= SLICE_PROP_STOP
        if py_slice.step is not None:
            c_slice.step = py_slice.step
            c_slice.mask |= SLICE_PROP_STEP
        return 0
    
    raise ValueError(f"Invalid slice: {py_slice}. Expected slice instance or tuple/list of slices.")


cdef inline int slices2slices_t(slices_t& slices, object py_slice) except -1:
    if isinstance(py_slice, tuple | list):
        slices.resize(len(py_slice))
        for i in range(len(py_slice)):
            slice2slice_struct(slices[i], py_slice[i])
        return 0
    else:
        slices.resize(1)
        slice2slice_struct(slices[0], py_slice)
        return 0


# ==============================
# Implementation details - C helpers
# ==============================


cdef inline int64_t gcd(int64_t a, int64_t b) except -1 nogil:
    while b != 0:
        a, b = b, a % b
    return a


@cython.overflowcheck(True)
cdef inline int64_t volume(shape_t& shape) except? -1 nogil:
    cdef int64_t vol = 1
    for i in range(shape.size()):
        vol *= shape[i]
    return vol


@cython.overflowcheck(True)
cdef inline int64_t overflow_checked_mul(int64_t stride, int64_t b) except -1 nogil:
    return stride * b


cdef inline int divide_strides(strides_t &strides, int itemsize) except -1 nogil:
    cdef stride_t stride
    for i in range(strides.size()):
        stride = strides[i] // itemsize
        if stride * itemsize != strides[i]:
            raise ValueError("strides must be divisible by itemsize")
        strides[i] = stride
    return 0


cdef inline int strides_in_bytes(strides_t &out_strides, strides_t &in_strides, int itemsize) except -1 nogil:
    cdef int ndim = in_strides.size()
    out_strides.clear()
    out_strides.reserve(ndim)
    for i in range(ndim):
        out_strides.push_back(overflow_checked_mul(in_strides[i], itemsize))
    return 0


cdef inline bint is_unique(shape_t& shape, strides_t& strides, axis_order_t& stride_order) except -1 nogil:
    cdef int64_t cur_max_offset = 0
    cdef int i = shape.size() - 1
    cdef int64_t stride
    cdef axis_t axis
    cdef extent_t extent
    while i >= 0:
        axis = stride_order[i]
        extent = shape[axis]
        if extent != 1:
            stride = c_abs(strides[axis])
            if cur_max_offset >= stride:
                return False
            cur_max_offset += overflow_checked_mul(stride, (extent - 1))
        i -= 1
    return True


cdef inline int offset_bounds(stride_t& min_offset, stride_t& max_offset, shape_t& shape, strides_t& strides) except -1 nogil:
    min_offset = 0
    max_offset = 0
    cdef stride_t stride
    cdef extent_t extent
    for i in range(shape.size()):
        stride = strides[i]  # can be negative
        extent = shape[i]  # must be non-negative
        if stride <= 0:
            min_offset += overflow_checked_mul(stride, (extent - 1))
        else:
            max_offset += overflow_checked_mul(stride, (extent - 1))
    return 0


@cython.overflowcheck(True)
cdef inline int64_t required_size_in_bytes(stride_t slice_offset, shape_t& shape, strides_t& strides) except -1 nogil:
    cdef stride_t min_offset = 0
    cdef stride_t max_offset = 0
    offset_bounds(min_offset, max_offset, shape, strides)
    min_offset = min(min_offset, -slice_offset)
    max_offset = max(max_offset, -slice_offset)
    return max_offset - min_offset + 1
    

cdef inline int zeros(vector_t& vec, int ndim) except -1 nogil:
    vec.clear()
    vec.resize(ndim, 0)
    return 0


cdef inline stride_t _dense_strides_c(strides_t& strides, shape_t& shape) except -1 nogil:
    cdef int ndim = shape.size()
    strides.resize(ndim)
    cdef stride_t stride = 1
    cdef int i = ndim - 1
    while i >= 0:
        strides[i] = stride
        stride *= shape[i]
        i -= 1
    return stride


cdef inline stride_t _dense_strides_f(strides_t& strides, shape_t& shape) except -1 nogil:
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


cdef inline stride_t _dense_strides_in_order(strides_t& strides, shape_t& shape, axis_order_t& stride_order) except -1 nogil:
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
        if not normalize_axis(axis, ndim):
            raise ValueError(f"Invalid stride order: axis {axis} out of range for {ndim}D tensor")
        axis_mask = 1 << axis
        if axis_order_mask & axis_mask:
            raise ValueError(f"The stride order must be a permutation. Axis {axis} appears multiple times.")
        axis_order_mask |= axis_mask
        strides[axis] = stride
        stride *= shape[axis]
        i -= 1
    return stride


cdef inline bint is_contiguous_c(shape_t& shape, strides_t& strides) except -1 nogil:
    cdef int64_t stride = 1
    cdef int64_t j = shape.size() - 1
    cdef extent_t extent
    while j >= 0:
        extent = shape[j]
        if extent != 1:
            if strides[j] != stride:
                return False
            stride *= extent
        j -= 1
    return True


cdef inline bint is_contiguous_f(shape_t& shape, strides_t& strides) except -1 nogil:
    cdef int ndim = shape.size()
    cdef int64_t stride = 1
    cdef int64_t j = 0
    cdef extent_t extent
    while j < ndim:
        extent = shape[j]
        if extent != 1:
            if strides[j] != stride:
                return False
            stride *= extent
        j += 1
    return True


cdef inline bint is_contiguous_in_order(shape_t& shape, strides_t& strides, axis_order_t& axis_order) except -1 nogil:
    cdef int64_t stride = 1
    cdef int64_t j = shape.size() - 1
    cdef axis_t axis
    cdef extent_t extent
    while j >= 0:
        axis = axis_order[j]
        extent = shape[axis]
        if extent != 1:
            if strides[axis] != stride:
                return False
            stride *= extent
        j -= 1
    return True


cdef inline bint split_strides_in_c_index_order(shape_t& new_shape, strides_t& new_strides, shape_t& shape, strides_t& strides) except -1 nogil:
    cdef int i = shape.size() - 1
    cdef int new_i = new_shape.size() - 1
    cdef extent_t extent
    cdef extent_t new_extent
    cdef extent_t group_vol
    cdef stride_t group_stride
    while i >= 0:
        extent = shape[i]
        group_vol = 1
        group_stride = strides[i]
        while new_i >= 0 and group_vol < extent:
            new_extent = new_shape[new_i]
            if new_extent == 0:
                return False
            group_vol *= new_extent
            new_strides[new_i] = group_stride
            group_stride *= new_extent
            new_i -= 1
        if group_vol != extent:
            return False
        i -= 1
    return True


cdef inline int flatten_strides_in_c_index_order(shape_t& out_shape, strides_t& out_strides, shape_t& shape, strides_t& strides, axes_mask_t axis_mask) except -1 nogil:
    cdef int ndim = shape.size()
    out_shape.resize(ndim)
    out_strides.resize(ndim)
    cdef int group_start = 0
    cdef int group_end = 0
    cdef int64_t group_vol
    cdef int64_t group_stride
    cdef int out_i = 0
    while group_start < ndim:
        group_vol = shape[group_start] 
        group_stride = strides[group_start]
        group_end = group_start + 1
        while group_end < ndim and (axis_mask & (1 << group_end)) and group_stride == strides[group_end] * shape[group_end]:
            group_vol *= shape[group_end]
            group_stride = strides[group_end]
            group_end += 1
        out_shape[out_i] = group_vol
        out_strides[out_i] = group_stride
        out_i += 1
        group_start = group_end
    if out_i != ndim:
        out_shape.resize(out_i)
        out_strides.resize(out_i)
    return out_i


cdef inline axes_mask_t flattened_strides_in_c_index_order_mask(shape_t& shape, strides_t& strides) except? -1 nogil:
    cdef axes_mask_t axis_mask = 0
    cdef int ndim = shape.size()
    cdef int group_start = 0
    cdef int group_end = 0
    cdef int64_t group_vol
    cdef int64_t group_stride
    while group_start < ndim:
        group_vol = shape[group_start] 
        group_stride = strides[group_start]
        group_end = group_start + 1
        while group_end < ndim and group_stride == strides[group_end] * shape[group_end]:
            group_vol *= shape[group_end]
            group_stride = strides[group_end]
            axis_mask |= (1 << group_end)
            group_end += 1
        group_start = group_end
    return axis_mask


cdef inline int permute_extents(shape_t& out_shape, strides_t& out_strides, shape_t& shape, strides_t& strides, axis_order_t& axis_order) except -1 nogil:
    cdef int ndim = shape.size()
    out_shape.clear()
    out_shape.reserve(ndim)
    out_strides.clear()
    out_strides.reserve(ndim)
    cdef axes_mask_t axis_order_mask = 0
    cdef axes_mask_t axis_mask
    cdef axis_t axis
    for i in range(ndim):
        axis = axis_order[i]
        if not normalize_axis(axis, ndim):
            raise ValueError(f"Invalid permutation: axis {axis} out of range for {ndim}D tensor")
        axis_mask = 1 << axis
        if axis_order_mask & axis_mask:
            raise ValueError(f"Invalid permutation: axis {axis} appears multiple times.")
        axis_order_mask |= axis_mask
        out_shape.push_back(shape[axis])
        out_strides.push_back(strides[axis])
    return 0


cdef inline int squeeze_extents(shape_t& out_shape, strides_t& out_strides, shape_t& shape, strides_t& strides) except -1 nogil:
    cdef int ndim = shape.size()
    out_shape.clear()
    out_shape.reserve(ndim)
    out_strides.clear()
    out_strides.reserve(ndim)
    cdef int out_ndim = 0
    cdef extent_t extent
    for i in range(ndim):
        extent = shape[i]
        if extent == 0:
            zeros(out_shape, 1)
            zeros(out_strides, 1)
            return 1
        if extent != 1:
            out_shape.push_back(extent)
            out_strides.push_back(strides[i])
            out_ndim += 1
    return out_ndim


cdef inline int pack_extents(stride_t& out_slice_offset, shape_t& out_shape, strides_t& out_strides, stride_t slice_offset, shape_t& shape, strides_t& strides, int itemsize, int new_itemsize, intptr_t data_ptr, bint keep_dim, int axis) except -1 nogil:
    cdef int ndim = shape.size()
    if new_itemsize <= 0 or new_itemsize & (new_itemsize - 1):
        raise ValueError(f"new itemsize must be a power of two, got {new_itemsize}.")
    if itemsize <= 0 or itemsize & (itemsize - 1):
        raise ValueError(f"itemsize must be a power of two, got {itemsize}.")
    if new_itemsize <= itemsize:
        if new_itemsize == itemsize:
            return 1
        raise ValueError(f"new itemsize ({new_itemsize}) must be greater than or equal to itemsize ({itemsize}).")
    if not normalize_axis(axis, ndim):
        raise ValueError(f"Invalid axis: {axis} out of range for {ndim}D tensor")
    if strides[axis] != 1:
        raise ValueError(f"The axis {axis} stride must be 1, got {strides[axis]}.")
    if data_ptr % new_itemsize != 0:
        raise ValueError(f"The data pointer ({data_ptr}) must be aligned to the packed itemsize ({new_itemsize}).")

    cdef int vec_size = new_itemsize // itemsize
    cdef extent_t packed_extent = shape[axis]
    if packed_extent == 0:
        raise ValueError(f"The axis {axis} extent must be non-zero, got {shape[axis]}.")
    packed_extent //= vec_size
    if packed_extent * vec_size != shape[axis]:
        raise ValueError(f"The axis {axis} extent ({shape[axis]}) must be divisible by {vec_size}.")

    cdef stride_t new_slice_offset = slice_offset // vec_size
    if new_slice_offset * vec_size != slice_offset:
        raise ValueError(f"The slice offset ({slice_offset}) must be divisible by {vec_size}.")
    out_slice_offset = new_slice_offset

    out_shape.clear()
    out_strides.clear()
    out_shape.reserve(ndim)
    out_strides.reserve(ndim)
    cdef stride_t packed_stride
    for i in range(ndim):
        if i == axis:
            if keep_dim or packed_extent != 1:  # omit the packed axis if it is reduced to 1
                out_shape.push_back(packed_extent)
                out_strides.push_back(1)
        else:
            packed_stride = strides[i] // vec_size
            if packed_stride * vec_size != strides[i]:
                raise ValueError(f"The {i} axis stride ({strides[i]}) must be divisible by {vec_size}.")
            out_shape.push_back(shape[i])
            out_strides.push_back(packed_stride)
    return vec_size


cdef inline int unpack_extents(shape_t& out_shape, strides_t& out_strides, shape_t& shape, strides_t& strides, int itemsize, int new_itemsize, int axis) except -1 nogil:
    cdef int ndim = shape.size()
    if not normalize_axis(axis, ndim):
        raise ValueError(f"Invalid axis: {axis} out of range for {ndim}D tensor")
    if new_itemsize <= 0 or new_itemsize & (new_itemsize - 1):
        raise ValueError(f"new itemsize must be a power of two, got {new_itemsize}.")
    if itemsize <= 0 or itemsize & (itemsize - 1):
        raise ValueError(f"itemsize must be a power of two, got {itemsize}.")
    if new_itemsize >= itemsize:
        if new_itemsize == itemsize:
            return 1
        raise ValueError(f"new itemsize ({new_itemsize}) must be less than or equal to itemsize ({itemsize}).")
    if strides[axis] != 1:
        raise ValueError(f"The axis {axis} stride must be 1, got {strides[axis]}.")
    cdef int vec_size = itemsize // new_itemsize
    cdef extent_t unpacked_extent = shape[axis]
    if unpacked_extent == 0:
        raise ValueError(f"The axis {axis} extent must be non-zero, got {shape[axis]}.")
    unpacked_extent = overflow_checked_mul(unpacked_extent, vec_size)
    for i in range(ndim):
        if i == axis:
            out_shape.push_back(unpacked_extent)
            out_strides.push_back(1)
        else:
            out_shape.push_back(shape[i])
            out_strides.push_back(overflow_checked_mul(strides[i], vec_size))
    return vec_size


cdef inline int max_compatible_itemsize(stride_t slice_offset, int itemsize, shape_t& shape, strides_t& strides, int max_itemsize, int axis, intptr_t data_ptr) except -1 nogil:
    cdef int ndim = shape.size()
    if max_itemsize <= 0 or max_itemsize & (max_itemsize - 1):
        raise ValueError(f"max_itemsize must be a power of two, got {max_itemsize}.")
    if itemsize <= 0 or itemsize & (itemsize - 1):
        raise ValueError(f"itemsize must be a power of two, got {itemsize}.")
    if not normalize_axis(axis, ndim):
        raise ValueError(f"Invalid axis: {axis} out of range for {ndim}D tensor")
    max_itemsize = gcd(max_itemsize, c_abs(data_ptr))
    if ndim < 1 or strides[axis] != 1 or shape[axis] == 0:
        return min(max_itemsize, itemsize)
    max_itemsize = gcd(max_itemsize, overflow_checked_mul(slice_offset, itemsize))
    max_itemsize = gcd(max_itemsize, overflow_checked_mul(shape[axis], itemsize))
    for i in range(ndim):
        if i == axis:
            continue
        max_itemsize = gcd(max_itemsize, overflow_checked_mul(c_abs(strides[i]), itemsize))
    return max_itemsize


@cython.overflowcheck(True)
cdef inline int64_t normalize_clamp_index(int64_t index, int64_t extent, bint has_negative_step) except -1 nogil:
    # first translate negative indexing to respective positive one
    if index < 0:
        index += extent
    # then clamp the index
    # * for positive step, [0, extent] is the valid range, where:
    #   * start=extent indicates no elements should be taken 
    #   * and stop=extent indicates no trimming from the end
    # * for negative step, the respective valid range is [-1, extent-1], where:
    #   * start=-1 indicates no elements should be taken
    #   * and stop=-1 indicates no trimming from the end,
    #   but there's no way to specify -1 explicitly 
    #   (as -1 is first translated to last element, i.e. extent-1)
    if index < 0:
        index = 0
    elif index >= extent:
        if has_negative_step:
            index = extent - 1
        else:
            index = extent
    return index


@cython.overflowcheck(True)
cdef inline stride_t slice_extents(shape_t& out_shape, strides_t& out_strides, shape_t& shape, strides_t& strides, slices_t& slices) except -1 nogil:
    cdef int ndim = shape.size()
    cdef int num_slices = slices.size()
    if num_slices > ndim:
        raise ValueError(f"The number of slices ({num_slices}) is greater than the number of dimensions ({ndim}).")
    out_shape.clear()
    out_shape.reserve(ndim)
    out_strides.clear()
    out_strides.reserve(ndim)
    cdef stride_t slice_offset = 0
    cdef int64_t step
    cdef bint has_negative_step
    cdef int64_t start
    cdef int64_t stop
    cdef extent_t extent
    cdef extent_t new_extent
    cdef extent_t extents_range
    for i in range(num_slices):
        extent = shape[i]
        if slices[i].mask & SLICE_PROP_SINGLE_ELEMENT:
            start = slices[i].start
            # the single index must be in [0, extent) range
            # ([-extent, -1] are valid too and translated to [0, extent-1] range)
            if not normalize_axis(start, extent):
                raise ValueError(f"Invalid index: {start} out of range for axis {i} with extent {extent}")
            slice_offset += start * strides[i]
            # single element index removes extent from the shape
            continue
        
        if slices[i].mask & SLICE_PROP_STEP:
            step = slices[i].step
            if step == 0:
                raise ValueError("The slice step cannot be zero.")
        else:
            step = 1
        has_negative_step = step < 0
        if slices[i].mask & SLICE_PROP_START:
            start = normalize_clamp_index(slices[i].start, extent, has_negative_step)
        else:
            start = extent - 1 if has_negative_step else 0
        if slices[i].mask & SLICE_PROP_STOP:
            stop = normalize_clamp_index(slices[i].stop, extent, has_negative_step)
        else:
            stop = -1 if has_negative_step else extent
        # start and stop are now in [0, extent] range for positive step,
        # and [-1, extent - 1] for negative step
        extents_range = start - stop if has_negative_step else stop - start
        if extents_range < 0:
            extents_range = 0
        new_extent = div_ceil(extents_range, c_abs(step))
        if new_extent > 0:
            # if start is not in [0, extent - 1] range, the
            # new_extent will be 0, so this way we avoid modifying 
            # offset with invalid start index, even though for
            # zero-volume layout, strides and offsets won't be used anyway
            slice_offset += start * strides[i]
        out_shape.push_back(new_extent)
        out_strides.push_back(step * strides[i])
    for i in range(num_slices, ndim):
        out_shape.push_back(shape[i])
        out_strides.push_back(strides[i])
    return slice_offset
