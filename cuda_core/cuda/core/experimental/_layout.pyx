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

    # ==============================
    # C API
    # ==============================

# ==============================
# Implementation details - StridedLayout class helpers
# ==============================



# ==============================
# Implementation details - python <-> C conversions
# ==============================



