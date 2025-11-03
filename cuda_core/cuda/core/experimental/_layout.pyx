# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython

from libc.stdint cimport int64_t, intptr_t
from libcpp cimport vector

from cpython.object cimport PyObject


cdef extern from "Python.h":
    int _PySlice_Unpack "PySlice_Unpack" (PyObject *slice, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step) except -1
    Py_ssize_t _PySlice_AdjustIndices "PySlice_AdjustIndices" (Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t step) noexcept nogil


@cython.final
cdef class StridedLayout:

    def __init__(StridedLayout self, object shape, object strides, int itemsize, bint strides_in_bytes=False):
        self.init_from_tuple(shape, strides, itemsize, strides_in_bytes)
    
    @classmethod
    def dense(cls, object shape, int itemsize, object stride_order='C'):
        cdef StridedLayout new_layout = StridedLayout.__new__(cls)
        new_layout.init_dense_from_tuple(shape, itemsize, stride_order)
        return new_layout

    def __repr__(StridedLayout self):
        if self.slice_offset == 0:
            return (
                f"StridedLayout(shape={self.shape}, strides={self.strides}, itemsize={self.itemsize})"
            )
        else:
            return (
                f"StridedLayout(shape={self.shape}, strides={self.strides}, itemsize={self.itemsize}, _slice_offset={self.slice_offset})"
            )

    def __eq__(StridedLayout self, StridedLayout other):
        return self.itemsize == other.itemsize and self.slice_offset == other.slice_offset and _base_layout_equal(self.base, other.base)

    @property
    def ndim(StridedLayout self) -> int:
        return self.base.ndim

    @property
    def shape(StridedLayout self) -> tuple:
        return self.get_shape_tuple()

    @property
    def strides(StridedLayout self) -> tuple | None:
        return self.get_strides_tuple()

    @property
    def strides_in_bytes(StridedLayout self) -> tuple | None:
        return self.get_strides_in_bytes_tuple()

    @property
    def stride_order(StridedLayout self) -> tuple:
        return self.get_stride_order_tuple()

    @property
    def volume(StridedLayout self) -> int:
        return self.get_volume()

    @property
    def is_unique(StridedLayout self) -> bool:
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
    
    def flattened_axis_mask(StridedLayout self):
        return self.get_flattened_axis_mask()
    
    def reshaped(self, object shape):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        cdef BaseLayout new_shape
        _init_layout(new_shape, len(shape))
        for i in range(len(shape)):
            new_shape.shape[i] = shape[i]
        self.reshape_into(new_layout, new_shape)
        return new_layout

    def permuted(self, object axis_order):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        cdef axis_order_t axis_order_vec
        _tuple2axis_order(axis_order_vec, axis_order)
        self.permute_into(new_layout, axis_order_vec)
        return new_layout
#
    #def flattened(self, start_axis=0, end_axis=-1, mask=None):
    #    cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
    #    cdef axes_mask_t axis_mask = mask if mask is not None else axis_mask_from_range(self.ndim, start_axis, end_axis)
    #    self.flatten_into(new_layout, axis_mask)
    #    return new_layout
    #
    #def flattened_axis_mask(self):
    #    return self.get_flattened_axis_mask()
    #
    #def squeezed(self):
    #    cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
    #    self.squeeze_into(new_layout)
    #    return new_layout
    #
    #def packed(self, int itemsize, intptr_t data_ptr=0, int axis=-1, bint keep_dim=True):
    #    if itemsize == self.itemsize:
    #        return self
    #    cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
    #    self.pack_into(new_layout, itemsize, data_ptr, keep_dim, axis)
    #    return new_layout
    #
    #def unpacked(self, int itemsize, int axis=-1):
    #    if itemsize == self.itemsize:
    #        return self
    #    cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
    #    self.unpack_into(new_layout, itemsize, axis)
    #    return new_layout
    #
    #def max_compatible_itemsize(self, int max_itemsize=16, intptr_t data_ptr=0, int axis=-1):
    #    return self.get_max_compatible_itemsize(max_itemsize, data_ptr, axis)
    #
    def sliced(self, object slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        self.slice_into(new_layout, slices)
        return new_layout

    def __getitem__(StridedLayout self, object slices):
        return self.sliced(slices)

    cdef axes_mask_t get_flattened_axis_mask(StridedLayout self) except? -1 nogil:
        return flattened_strides_in_c_index_order_mask(self.base)

    cdef int reshape_into(StridedLayout self, StridedLayout out_layout, BaseLayout& new_shape) except -1 nogil:
        cdef int64_t old_volume = self.get_volume()
        validate_reshaped_shape(new_shape, old_volume)
        
        cdef int ndim = new_shape.ndim
        _zero_strides(new_shape)

        cdef BaseLayout flattened
        if old_volume != 0:
            flatten_strides_in_c_index_order(flattened, self.base, AXIS_MASK_ALL)
            if not split_strides_in_c_index_order(new_shape, flattened):
                raise ValueError("Layout strides are incompatible with the new shape")
        
        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Copy preserved attributes
        out_layout.slice_offset = self.slice_offset
        out_layout.itemsize = self.itemsize
        maybe_copy_volume(out_layout, self)

        # Set new attributes
        _swap_layout(out_layout.base, new_shape)
        return 0

    cdef int permute_into(StridedLayout self, StridedLayout out_layout, axis_order_t& axis_order) except -1 nogil:
        if axis_order.size() != <size_t>self.base.ndim:
            raise ValueError(f"Permutation must have the same length as the number of dimensions, got {axis_order.size()} for {self.ndim}D tensor.")

        cdef BaseLayout permuted
        permute_extents(permuted, self.base, axis_order)
    
        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize
        out_layout.slice_offset = self.slice_offset
        maybe_copy_volume(out_layout, self)

        # Set new attributes
        _swap_layout(out_layout.base, permuted)
        return 0

    #cdef int flatten_into(StridedLayout self, StridedLayout out_layout, axes_mask_t axis_mask=AXIS_MASK_ALL) except -1 nogil:
    #    cdef shape_t new_shape
    #    cdef strides_t new_strides
    #    cdef int ndim = flatten_strides_in_c_index_order(new_shape, new_strides, self.shape, self.strides, axis_mask)

    #    if out_layout is self and ndim == self.ndim:
    #        return 0

    #    # Reset all memoized properties
    #    out_layout._prop_mask = 0

    #    # Preserved attributes
    #    out_layout.itemsize = self.itemsize
    #    out_layout.volume = self.volume
    #    out_layout.slice_offset = self.slice_offset

    #    # Set new attributes
    #    out_layout.ndim = ndim
    #    _swap(out_layout.shape, new_shape)
    #    _swap(out_layout.strides, new_strides)
    #    return 0
    #
    #cdef int squeeze_into(StridedLayout self, StridedLayout out_layout) except -1 nogil:
    #    cdef shape_t new_shape
    #    cdef strides_t new_strides
    #    cdef int ndim = squeeze_extents(new_shape, new_strides, self.shape, self.strides)
    #    
    #    if out_layout is self and ndim == self.ndim:
    #        return 0

    #    # Reset all memoized properties
    #    out_layout._prop_mask = 0

    #    # Preserved attributes
    #    out_layout.itemsize = self.itemsize
    #    out_layout.volume = self.volume
    #    out_layout.slice_offset = self.slice_offset

    #    # Set new attributes
    #    out_layout.ndim = ndim
    #    _swap(out_layout.shape, new_shape)
    #    _swap(out_layout.strides, new_strides)
    #    return 0

    #cdef int pack_into(StridedLayout self, StridedLayout out_layout, int itemsize, intptr_t data_ptr, bint keep_dim, int axis=-1) except -1 nogil:
    #    
    #    cdef shape_t new_shape
    #    cdef strides_t new_strides
    #    cdef stride_t new_slice_offset = 0
    #    cdef int vec_size = pack_extents(
    #        new_slice_offset,
    #        new_shape,
    #        new_strides,
    #        self.slice_offset,
    #        self.shape,
    #        self.strides,
    #        self.itemsize,
    #        itemsize,
    #        data_ptr,
    #        keep_dim,
    #        axis
    #    )
    #    cdef int64_t new_volume = self.volume // vec_size

    #    if vec_size == 1 and out_layout is self:
    #        return 0

    #    # Reset all memoized properties
    #    out_layout._prop_mask = 0

    #    # Set new attributes
    #    out_layout.itemsize = itemsize
    #    out_layout.volume = new_volume
    #    out_layout.ndim = new_shape.size()
    #    out_layout.slice_offset = new_slice_offset
    #    _swap(out_layout.shape, new_shape)
    #    _swap(out_layout.strides, new_strides)
    #    return vec_size
    #
    #cdef int unpack_into(StridedLayout self, StridedLayout out_layout, int itemsize, int axis=-1) except -1 nogil:
    #    cdef shape_t new_shape
    #    cdef strides_t new_strides
    #    cdef int vec_size = unpack_extents(
    #        new_shape,
    #        new_strides,
    #        self.shape,
    #        self.strides,
    #        self.itemsize,
    #        itemsize,
    #        axis
    #    )
    #    if vec_size == 1 and out_layout is self:
    #        return 0
    #    
    #    cdef int64_t new_volume = overflow_checked_mul(self.volume, vec_size)
    #    cdef int64_t new_slice_offset = overflow_checked_mul(self.slice_offset, vec_size)
    #    
    #    # Reset all memoized properties
    #    out_layout._prop_mask = 0

    #    # Set new attributes
    #    out_layout.itemsize = itemsize
    #    out_layout.volume = new_volume
    #    out_layout.ndim = new_shape.size()
    #    out_layout.slice_offset = new_slice_offset
    #    _swap(out_layout.shape, new_shape)
    #    _swap(out_layout.strides, new_strides)
    #    return vec_size
    #
    #cdef int get_max_compatible_itemsize(StridedLayout self, int max_itemsize, intptr_t data_ptr, int axis=-1) except -1 nogil:
    #    return max_compatible_itemsize(self.slice_offset, self.itemsize, self.shape, self.strides, max_itemsize, axis, data_ptr)
    #
    cdef int slice_into(StridedLayout self, StridedLayout out_layout, tuple slices) except -1:
        cdef BaseLayout sliced
        cdef stride_t slice_offset = slice_extents(sliced, self.base, slices)
        cdef int64_t new_slice_offset = _overflow_checked_sum(self.slice_offset, slice_offset)

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize
        maybe_copy_volume(out_layout, self)
        
        # Set new attributes
        _swap_layout(out_layout.base, sliced)
        out_layout.slice_offset = new_slice_offset
        return 0

cdef inline int maybe_copy_volume(StridedLayout out_layout, StridedLayout in_layout) except -1 nogil:
    if _has_valid_property(out_layout, PROP_VOLUME):
        out_layout._volume = in_layout.get_volume()
        _mark_property_valid(out_layout, PROP_VOLUME)
    return 0


cdef inline int validate_reshaped_shape(BaseLayout& new_shape, int64_t old_volume) except -1 nogil:
    cdef int ndim = new_shape.ndim
    cdef int axis = -1
    cdef extent_t extent
    for i in range(ndim):
        extent = new_shape.shape[i]
        if extent < -1:
            raise ValueError("Extents must be non-negative")
        elif extent == -1:
            if axis == -1:
                axis = i
            else:
                raise ValueError("There can be at most one -1 extent in a shape")
    cdef int64_t new_volume = _c_abs(_volume(new_shape))
    if new_volume == 0 and axis != -1:
        raise ValueError("The -1 extent is ambiguous when the volume is 0")
    if new_volume != old_volume:
        if axis == -1:
            raise ValueError(f"The original volume {old_volume} and the new volume {new_volume} must be equal.")
        extent = old_volume // new_volume
        if extent * new_volume != old_volume:
            raise ValueError(f"The original volume {old_volume} must be divisible by the specified sub-volume {new_volume}.")
        new_shape.shape[axis] = extent
    return 0


cdef inline int flatten_strides_in_c_index_order(BaseLayout& out_layout, BaseLayout& in_layout, axes_mask_t axis_mask) except -1 nogil:
    if in_layout.strides == NULL:
        _init_layout(out_layout, 1)
        out_layout.shape[0] = _volume(in_layout)
        out_layout.strides[0] = 1
        return 1
    cdef int ndim = in_layout.ndim
    _init_layout(out_layout, ndim)
    cdef int group_start = 0
    cdef int group_end = 0
    cdef int64_t group_vol
    cdef int64_t group_stride
    cdef int out_i = 0
    while group_start < ndim:
        group_vol = in_layout.shape[group_start] 
        group_stride = in_layout.strides[group_start]
        group_end = group_start + 1
        while group_end < ndim and (axis_mask & (1 << group_end)) and group_stride == in_layout.strides[group_end] * in_layout.shape[group_end]:
            group_vol = _overflow_checked_mul(group_vol, in_layout.shape[group_end])
            group_stride = in_layout.strides[group_end]
            group_end += 1
        out_layout.shape[out_i] = group_vol
        out_layout.strides[out_i] = group_stride
        out_i += 1
        group_start = group_end
    if out_i != ndim:
        _trim_layout(out_layout, out_i)
    return out_i


cdef inline axes_mask_t flattened_strides_in_c_index_order_mask(BaseLayout& layout) except? -1 nogil:
    if layout.strides == NULL:
        return AXIS_MASK_ALL
    cdef axes_mask_t axis_mask = 0
    cdef int ndim = layout.ndim
    cdef int group_start = 0
    cdef int group_end = 0
    cdef int64_t group_vol
    cdef int64_t group_stride
    while group_start < ndim:
        group_vol = layout.shape[group_start] 
        group_stride = layout.strides[group_start]
        group_end = group_start + 1
        while group_end < ndim and group_stride == layout.strides[group_end] * layout.shape[group_end]:
            group_vol = _overflow_checked_mul(group_vol, layout.shape[group_end])
            group_stride = layout.strides[group_end]
            axis_mask |= (1 << group_end)
            group_end += 1
        group_start = group_end
    return axis_mask


cdef inline bint split_strides_in_c_index_order(BaseLayout& out_layout, BaseLayout& in_layout) except -1 nogil:
    cdef int i = in_layout.ndim - 1
    cdef int new_i = out_layout.ndim - 1
    cdef extent_t extent
    cdef extent_t new_extent
    cdef extent_t group_vol
    cdef stride_t group_stride
    cdef stride_t c_stride = 1
    if out_layout.strides == NULL:
        _zero_strides(out_layout)
    while i >= 0:
        extent = in_layout.shape[i]
        group_vol = 1
        if in_layout.strides == NULL:
            group_stride = c_stride
        else:
            group_stride = in_layout.strides[i]
        while new_i >= 0 and group_vol < extent:
            new_extent = out_layout.shape[new_i]
            if new_extent == 0:
                return False
            group_vol = _overflow_checked_mul(group_vol, new_extent)
            out_layout.strides[new_i] = group_stride
            group_stride = _overflow_checked_mul(group_stride, new_extent)
            new_i -= 1
        if group_vol != extent:
            return False
        c_stride = group_stride
        i -= 1
    return True


cdef inline int _permute_extents(BaseLayout& out_layout, BaseLayout& in_layout, axis_order_t& axis_order) except -1 nogil:
    cdef int ndim = in_layout.ndim
    _init_layout(out_layout, ndim)
    cdef axis_t axis
    cdef axes_mask_t axis_mask
    cdef axes_mask_t axis_order_mask = 0

    for i in range(ndim):
        axis = axis_order[i]
        if not _normalize_axis(axis, ndim):
            raise ValueError(f"Invalid permutation: axis {axis} out of range for {ndim}D tensor")
        axis_mask = 1 << axis
        if axis_order_mask & axis_mask:
            raise ValueError(f"Invalid permutation: axis {axis_order[i]} appears multiple times.")
        axis_order_mask |= axis_mask
        out_layout.shape[i] = in_layout.shape[axis]
        out_layout.strides[i] = in_layout.strides[axis]
    return 0


cdef inline int permute_extents(BaseLayout& out_layout, BaseLayout& in_layout, axis_order_t& axis_order) except -1 nogil:
    if in_layout.strides != NULL:
        return _permute_extents(out_layout, in_layout, axis_order)
    cdef BaseLayout tmp
    _init_layout(tmp, in_layout.ndim)
    for i in range(in_layout.ndim):
        tmp.shape[i] = in_layout.shape[i]
    _dense_strides_c(tmp)
    _permute_extents(out_layout, tmp, axis_order)
    return 0


cdef inline stride_t _slice_extents(BaseLayout& out_layout, BaseLayout& in_layout, tuple slices) except? -1:
    cdef int ndim = in_layout.ndim
    cdef int num_slices = len(slices)
    if num_slices > ndim:
        raise ValueError(f"The number of slices ({num_slices}) is greater than the number of dimensions ({ndim}).")
    _init_layout(out_layout, ndim)
    cdef stride_t slice_offset = 0
    cdef stride_t stride
    cdef extent_t extent
    cdef Py_ssize_t start
    cdef Py_ssize_t stop
    cdef Py_ssize_t step
    cdef extent_t new_extent
    cdef object py_slice
    cdef bint zero_slice = False
    cdef int out_i = 0
    for i in range(num_slices):
        extent = in_layout.shape[i]
        stride = in_layout.strides[i]
        py_slice = slices[i]
        if isinstance(py_slice, int):
            start = py_slice
            if not _normalize_axis(start, extent):
                raise ValueError(f"Invalid index: {start} out of range for axis {i} with extent {extent}")
            # single element index removes extent from the shape,
            # just increase the offset and skip the shape and stride
            slice_offset = _overflow_checked_sum(slice_offset, _overflow_checked_mul(start, stride))
        elif isinstance(py_slice, slice):
            _PySlice_Unpack(<PyObject *>py_slice, &start, &stop, &step)
            new_extent = _PySlice_AdjustIndices(extent, &start, &stop, step)
            if new_extent > 0:
                # out_extent > 0 implies start is in [0, extent - 1] range
                slice_offset = _overflow_checked_sum(slice_offset, _overflow_checked_mul(start, stride))
            else:
                zero_slice = True
            out_layout.shape[out_i] = new_extent
            out_layout.strides[out_i] = _overflow_checked_mul(stride, step)
            out_i += 1
        else:
            raise ValueError(f"Invalid slice: {py_slice}. Expected slice instance or integer.")
    for i in range(num_slices, ndim):
        out_layout.shape[out_i] = in_layout.shape[i]
        out_layout.strides[out_i] = in_layout.strides[i]
        out_i += 1
    if out_i != ndim:
        _trim_layout(out_layout, out_i)
    if zero_slice:
        _zero_strides(out_layout)
    return slice_offset


cdef inline stride_t slice_extents(BaseLayout& out_layout, BaseLayout& in_layout, tuple slices) except? -1:
    if in_layout.strides != NULL:
        return _slice_extents(out_layout, in_layout, slices)
    cdef BaseLayout tmp
    _init_layout(tmp, in_layout.ndim)
    for i in range(in_layout.ndim):
        tmp.shape[i] = in_layout.shape[i]
    _dense_strides_c(tmp)
    return _slice_extents(out_layout, tmp, slices)