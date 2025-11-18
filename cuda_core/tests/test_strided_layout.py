# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import random
from enum import Enum

import numpy as np
import pytest
from cuda.core.experimental._layout import StridedLayout

py_rng = random.Random(42)


class StridesKind(Enum):
    C = "C"
    IMPLICIT_C = "implicit_c"
    F = "F"
    PERMUTED = "permuted"
    SLICED_PERMUTED = "sliced_permuted"
    SLICED_BROADCAST_PERMUTED = "sliced_broadcast_permuted"
    SLIDING_WINDOW = "sliding_window"


class ReshapeErr(Enum):
    VOLUME_MISMATCH = "The original volume {old_volume} and the new volume {new_volume} must be equal."
    NEG_EXTENT = "Extents must be non-negative"
    MULTI_NEG_EXTENTS = "There can be at most one -1 extent in a shape"
    AMBIGUOUS_NEG_EXTENT = "The -1 extent is ambiguous when the volume is 0"
    DIVISIBILITY_VIOLATION = (
        "The original volume {old_volume} must be divisible by the specified sub-volume {new_volume}"
    )
    INCOM_STRIDES = "Layout strides are incompatible with the new shape"
    TYPE_ERROR = None


class PermuteErr(Enum):
    INVALID_LEN = (
        "Permutation must have the same length as the number of dimensions, got {perm_len} for {ndim}D tensor."
    )
    OUT_OF_RANGE = "out of range for {ndim}D tensor"
    MULTIPLE_OCCURRENCES = "appears multiple times."
    TYPE_ERROR = None


class SliceErr(Enum):
    ZERO_STEP = "slice step cannot be zer"
    TOO_MANY_SLICES = "is greater than the number of dimensions"
    OUT_OF_RANGE = "out of range for axis"
    TYPE_ERROR = "Expected slice instance or integer."


_ITEMSIZES = [1, 2, 4, 8, 16]


def idfn(val):
    """
    Pytest does not pretty print (repr/str) parameters of custom types.
    """
    if hasattr(val, "pretty_name"):
        return val.pretty_name()
    # use default pytest pretty printing
    return None


class DummySlice:
    def __getitem__(self, value):
        return value


_SL = DummySlice()


class Param:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __bool__(self):
        return bool(self.value)

    def pretty_name(self):
        if isinstance(self.value, Enum):
            value_str = self.value.name
        else:
            value_str = str(self.value)
        return f"{self.name}.{value_str}"


class LayoutSpec:
    def __init__(self, shape, stride_kind, strides, itemsize, np_ref=None):
        self.shape = shape
        self.stride_kind = stride_kind
        self.strides = strides
        self.itemsize = itemsize
        self.np_ref = np_ref

    def pretty_name(self):
        return "-".join(
            [
                f"ndim.{len(self.shape)}",
                f"shape.{self.shape}",
                f"strides.{self.strides}",
                f"stride_kind.{self.stride_kind.value}",
                f"itemsize.{self.itemsize}",
            ]
        )

    def layout(self):
        if self.stride_kind == StridesKind.IMPLICIT_C:
            assert self.strides is None
        return StridedLayout(self.shape, self.strides, self.itemsize)


def gen_permutations(rng, n):
    if n <= 3:
        return [perm for perm in itertools.permutations(range(n))]
    perms = []
    for _ in range(4):
        perm = list(range(n))
        rng.shuffle(perm)
        perms.append(tuple(perm))
    return perms


def permuted_tuple(t, perm):
    return tuple(t[i] for i in perm)


def inv_permutation(perm):
    inv = [None] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def is_id_perm(perm):
    return all(i == j for i, j in enumerate(perm))


def div_strides(strides, itemsize):
    return tuple(s // itemsize for s in strides)


def is_id_slice(slices):
    if isinstance(slices, slice):
        slices = (slices,)
    return all(sl == _SL[:] for sl in slices)


def cmp_layouts(layout, arr, has_no_strides):
    ndim = len(arr.shape)
    assert layout.ndim == ndim
    assert layout.shape == arr.shape
    if has_no_strides:
        assert layout.strides_in_bytes is None
        assert layout.strides is None
        assert arr.flags["C_CONTIGUOUS"]
    elif math.prod(arr.shape) == 0:
        assert layout.strides_in_bytes == tuple(0 for _ in range(ndim))
    else:
        assert layout.strides_in_bytes == arr.strides
    assert layout.volume == math.prod(arr.shape)
    assert layout.itemsize == arr.itemsize

    ref_c_contig = arr.flags["C_CONTIGUOUS"]
    ref_f_contig = arr.flags["F_CONTIGUOUS"]
    assert layout.is_contiguous_c == ref_c_contig
    assert layout.is_contiguous_f == ref_f_contig
    ref_any_contig = ref_c_contig or ref_f_contig or arr.transpose(layout.stride_order).flags["C_CONTIGUOUS"]
    assert layout.is_contiguous_any == ref_any_contig
    assert layout.is_dense == (ref_any_contig and layout.slice_offset == 0)


def random_non_empty_slice(rng, a):
    shape = a.shape
    ndim = len(shape)
    slicable_indicies = [i for i in range(ndim) if shape[i] > 1]
    if not slicable_indicies:
        return []
    sliced_ndim = rng.randint(1, len(slicable_indicies))
    sliced_indicies = rng.sample(slicable_indicies, sliced_ndim)
    slices = [slice(None)] * ndim
    for i in sliced_indicies:
        slice_size = rng.randint(1, shape[i] - 1)
        slice_start = rng.randint(0, shape[i] - slice_size)
        slice_end = slice_start + slice_size
        slices[i] = slice(slice_start, slice_end)
    view = a[tuple(slices)]
    if rng.choice([True, False]):
        neg_slices = [slice(None)] * ndim
        sliced_ndim = rng.randint(1, ndim)
        sliced_indicies = rng.sample(list(range(ndim)), sliced_ndim)
        for i in sliced_indicies:
            neg_slices[i] = slice(None, None, -1)
        view = view[tuple(neg_slices)]
    return [view]


def random_broadcast(rng, a):
    singletion_indicies = [i for i in range(len(a.shape)) if a.shape[i] == 1]
    if not singletion_indicies:
        return []
    broadcast_ndim = rng.randint(1, len(singletion_indicies))
    broadcast_indicies = rng.sample(singletion_indicies, broadcast_ndim)
    new_shape = list(a.shape)
    for i in broadcast_indicies:
        new_shape[i] = rng.randint(2, 100)
    return [np.broadcast_to(a, new_shape)]


def random_sliding_window(rng, a):
    ndim = len(a.shape)
    non_trival_extents = [i for i in range(ndim) if a.shape[i] > 2]
    if not non_trival_extents:
        return []
    sliding_window_ndim = rng.randint(1, len(non_trival_extents))
    sliding_window_indicies = rng.sample(non_trival_extents, sliding_window_ndim)
    window_sizes = tuple(rng.randint(2, a.shape[i] - 1) for i in sliding_window_indicies)
    return [np.lib.stride_tricks.sliding_window_view(a, window_sizes, sliding_window_indicies)]


def dtype_from_itemsize(itemsize):
    if itemsize <= 8:
        return np.dtype(f"int{itemsize * 8}")
    elif itemsize == 16:
        return np.dtype("complex128")
    else:
        raise ValueError(f"Unsupported itemsize: {itemsize}")


def flatten_mask2str(mask, ndim):
    return "".join("1" if mask & (1 << i) else "0" for i in range(ndim))


def gen_layouts(rng, shape, stride_kind, itemsize):
    dtype = dtype_from_itemsize(itemsize)
    vol = math.prod(shape)
    match stride_kind:
        case StridesKind.C:
            a = np.arange(vol, dtype=dtype).reshape(shape)
            return [(a, div_strides(a.strides, itemsize))]
        case StridesKind.IMPLICIT_C:
            a = np.arange(vol, dtype=dtype).reshape(shape)
            return [(a, None)]
        case StridesKind.F:
            a = np.arange(vol, dtype=dtype).reshape(shape, order="F")
            return [(a, div_strides(a.strides, itemsize))]
        case StridesKind.PERMUTED | StridesKind.SLICED_PERMUTED | StridesKind.SLICED_BROADCAST_PERMUTED:
            if len(shape) <= 1:
                return []
            a = np.arange(vol, dtype=dtype).reshape(shape)
            views = [a.transpose(perm) for perm in gen_permutations(rng, len(shape))]
            unique_views = {(v.shape, v.strides): v for v in views}
            views = list(unique_views.values())
            if stride_kind in [
                StridesKind.SLICED_PERMUTED,
                StridesKind.SLICED_BROADCAST_PERMUTED,
            ]:
                views = [v for view in views for v in random_non_empty_slice(rng, view)]
                if stride_kind == StridesKind.SLICED_BROADCAST_PERMUTED:
                    views = [v for view in views for v in random_broadcast(rng, view)]
            return [(view, div_strides(view.strides, itemsize)) for view in views]
        case StridesKind.SLIDING_WINDOW:
            a = np.arange(vol, dtype=dtype).reshape(shape)
            views = random_sliding_window(rng, a)
            return [(view, div_strides(view.strides, itemsize)) for view in views]


@pytest.mark.parametrize(
    ("shape", "permutation"),
    [
        (Param("shape", shape), Param("permutation", permutation))
        for shape in [tuple(), (1,), (2, 3), (5, 6, 7), (5, 1, 7)]
        for permutation in gen_permutations(py_rng, len(shape))
    ],
    ids=idfn,
)
def test_stride_order(shape, permutation):
    shape = shape.value
    permutation = permutation.value
    a = np.arange(math.prod(shape)).reshape(shape)
    v = a.transpose(inv_permutation(permutation))
    layout = StridedLayout(v.shape, v.strides, v.itemsize, divide_strides=True)
    assert layout.stride_order == tuple(permutation), (
        f"layout.stride_order == {layout.stride_order}, permutation == {permutation}"
    )


def test_dense():
    pass


def test_dense_like():
    pass


@pytest.mark.parametrize(
    ("layout_spec",),
    [
        (LayoutSpec(tuple(np_ref.shape), strides_kind, strides, itemsize, np_ref),)
        for base_shape in [
            tuple(),
            (1,),
            (17,),
            (5, 7),
            (1, 7),
            (7, 1),
            (2, 3, 5),
            (2, 1, 5),
            (1, 1, 1, 1),
            (2, 3, 0, 5),
        ]
        for strides_kind in StridesKind
        for itemsize in [1, 4]
        for np_ref, strides in gen_layouts(py_rng, base_shape, strides_kind, itemsize)
    ],
    ids=idfn,
)
def test_flags(layout_spec):
    assert layout_spec.stride_kind != StridesKind.IMPLICIT_C or layout_spec.strides is None
    layout = StridedLayout(layout_spec.shape, layout_spec.strides, layout_spec.itemsize)
    ref = layout_spec.np_ref
    flags = ref.flags
    c_contiguous = flags["C_CONTIGUOUS"]
    f_contiguous = flags["F_CONTIGUOUS"]
    assert layout.is_contiguous_c == c_contiguous, (
        f"layout.is_contiguous_c == {layout.is_contiguous_c}, c_contiguous == {c_contiguous}"
    )
    assert layout.is_contiguous_f == f_contiguous, (
        f"layout.is_contiguous_f == {layout.is_contiguous_f}, f_contiguous == {f_contiguous}"
    )
    a = ref.transpose(layout.stride_order)
    assert layout.is_contiguous_any == a.flags["C_CONTIGUOUS"], (
        f"layout.is_contiguous_any == {layout.is_contiguous_any}, a.flags['C_CONTIGUOUS'] == {a.flags['C_CONTIGUOUS']}"
    )
    volume = math.prod(layout_spec.shape)
    is_unique = volume == 0 or layout_spec.stride_kind not in [
        StridesKind.SLICED_BROADCAST_PERMUTED,
        StridesKind.SLIDING_WINDOW,
    ]
    assert layout.is_unique == is_unique, f"layout.is_unique == {layout.is_unique}, is_unique == {is_unique}"


@pytest.mark.parametrize(
    ("shape", "permutation", "stride_kind", "itemsize"),
    [
        (
            Param("shape", shape),
            Param("permutation", permutation),
            Param("stride_kind", stride_kind),
            Param("itemsize", py_rng.choice(_ITEMSIZES)),
        )
        for shape in [tuple(), (1,), (2, 3), (5, 6, 7), (5, 0, 7), (5, 0, 1, 7)]
        for permutation in gen_permutations(py_rng, len(shape))
        for stride_kind in [StridesKind.C, StridesKind.IMPLICIT_C, StridesKind.F]
    ],
    ids=idfn,
)
def test_permute(shape, permutation, stride_kind, itemsize):
    shape = shape.value
    permutation = permutation.value
    stride_kind = stride_kind.value
    itemsize = itemsize.value
    order = "C" if stride_kind == StridesKind.C else "F" if stride_kind == StridesKind.F else None
    np_ref = np.arange(math.prod(shape), dtype=dtype_from_itemsize(itemsize)).reshape(shape, order=order)

    if stride_kind == StridesKind.IMPLICIT_C:
        layout = StridedLayout(shape, None, itemsize)
    else:
        layout = StridedLayout.dense(shape, itemsize, stride_order=order)

    cmp_layouts(layout, np_ref, stride_kind == StridesKind.IMPLICIT_C)
    layout = layout.permuted(permutation)
    np_ref = np_ref.transpose(permutation)
    cmp_layouts(layout, np_ref, False)


@pytest.mark.parametrize(
    ("shape", "permutation", "error_msg", "stride_kind", "itemsize"),
    [
        (
            Param("shape", shape),
            Param("permutation", permutation),
            Param("error_msg", err_msg),
            Param("stride_kind", stride_kind),
            Param("itemsize", py_rng.choice(_ITEMSIZES)),
        )
        for shape, permutation, err_msg in [
            (tuple(), (0,), PermuteErr.INVALID_LEN),
            ((1, 2, 3), (0, 1), PermuteErr.INVALID_LEN),
            ((1, 2, 3), (1, 253333, 2), PermuteErr.OUT_OF_RANGE),
            ((1, 2, 3), (0, -4, 1), PermuteErr.OUT_OF_RANGE),
            ((1, 2, 3, 4), (0, 1, 0, 2), PermuteErr.MULTIPLE_OCCURRENCES),
            ((1, 2, 3), ("abc",), PermuteErr.TYPE_ERROR),
        ]
        for stride_kind in [StridesKind.C, StridesKind.IMPLICIT_C]
    ],
    ids=idfn,
)
def test_invalid_permute(shape, permutation, error_msg, stride_kind, itemsize):
    shape = shape.value
    permutation = permutation.value
    error_msg = error_msg.value
    stride_kind = stride_kind.value
    itemsize = itemsize.value
    if stride_kind == StridesKind.IMPLICIT_C:
        layout = StridedLayout(shape, None, itemsize)
    else:
        layout = StridedLayout.dense(shape, itemsize)

    if error_msg == PermuteErr.TYPE_ERROR:
        error_cls = TypeError
        match = None
    else:
        error_cls = ValueError
        match = error_msg.value.format(perm_len=len(permutation), ndim=len(shape))
    with pytest.raises(error_cls, match=match):
        layout.permuted(permutation)


@pytest.mark.parametrize(
    (
        "shape",
        "slices",
        "error_msg",
        "stride_kind",
        "itemsize",
    ),
    [
        (
            Param("shape", shape),
            Param("slices", slices),
            Param("error_msg", error_msg),
            Param("stride_kind", stride_kind),
            Param("itemsize", py_rng.choice(_ITEMSIZES)),
        )
        for shape, slices, error_msg in [
            (tuple(), (tuple(),), None),
            ((12,), (_SL[:],), None),
            ((13,), (_SL[::-1],), None),
            ((13,), (_SL[::-1], _SL[::-1]), None),
            ((13,), (_SL[::-1], _SL[1:-1], _SL[::-1]), None),
            ((13,), (_SL[2:-3],), None),
            ((13,), (_SL[2:-3:2],), None),
            ((13,), (_SL[-3:2:-2],), None),
            ((13,), (_SL[-3:2:-2], _SL[1:3]), None),
            ((3, 5), (_SL[:2], _SL[:, 3:]), None),
            ((3, 5), (_SL[5:4],), None),
            ((3, 5), (_SL[:, ::0],), SliceErr.ZERO_STEP),
            ((3, 5), (_SL[:, :-1, :2],), SliceErr.TOO_MANY_SLICES),
            ((11, 12, 3), (_SL[:, 0, :-1],), None),
            ((11, 12, 3), (_SL[0, 1, :-1],), None),
            ((11, 12, 3, 5), (_SL[0], _SL[1]), None),
            ((11, 12, 3, 5), (_SL[:, 1, :-1],), None),
            ((11, 12, 3), (_SL[0, 1, 2],), None),
            ((11, 12, 3), (_SL[0, 1, 5],), SliceErr.OUT_OF_RANGE),
            ((11, 12, 3), (_SL[-2],), None),
            ((11, 12, 3), (_SL[-42],), SliceErr.OUT_OF_RANGE),
            ((11, 12, 3), ("abc",), SliceErr.TYPE_ERROR),
        ]
        for stride_kind in [StridesKind.C, StridesKind.F, StridesKind.IMPLICIT_C]
    ],
    ids=idfn,
)
def test_slice(shape, slices, error_msg, stride_kind, itemsize):
    shape = shape.value
    slices = slices.value
    error_msg = error_msg.value
    stride_kind = stride_kind.value
    itemsize = itemsize.value
    order = "C" if stride_kind == StridesKind.C else "F" if stride_kind == StridesKind.F else None

    if stride_kind == StridesKind.IMPLICIT_C:
        layout = StridedLayout(shape, None, itemsize)
    else:
        layout = StridedLayout.dense(shape, itemsize, stride_order=order)

    if error_msg:
        error_cls = TypeError if error_msg == SliceErr.TYPE_ERROR else ValueError
        with pytest.raises(error_cls, match=error_msg.value):
            for sl in slices:
                layout = layout[sl]
        return

    np_ref = np.arange(math.prod(shape), dtype=dtype_from_itemsize(itemsize)).reshape(shape, order=order)

    cmp_layouts(layout, np_ref, stride_kind == StridesKind.IMPLICIT_C)
    prev_layout = layout
    prev_ref = np_ref
    for sl in slices:
        sliced = prev_layout[sl]
        ref_sliced = prev_ref[sl]
        cmp_layouts(sliced, ref_sliced, False)
        assert sliced.itemsize == itemsize
        # cannot access numpy's scalar data pointer
        if sliced.ndim > 0:
            ref_offset = ref_sliced.ctypes.data - prev_ref.ctypes.data
            layout_offset = sliced.slice_offset_in_bytes - prev_layout.slice_offset_in_bytes
            assert layout_offset == ref_offset
        prev_layout = sliced
        prev_ref = ref_sliced


@pytest.mark.parametrize(
    (
        "shape",
        "stride_kind",
        "slices",
        "new_shape",
        "permutation",
        "error_msg",
        "itemsize",
    ),
    [
        (
            Param("shape", shape),
            Param("stride_kind", base_stride_kind),
            Param("slices", slices),
            Param("new_shape", new_shape),
            Param("permutation", permutation),
            error_msg,
            Param("itemsize", py_rng.choice(_ITEMSIZES)),
        )
        for shape, slices, new_shape, permutation, error_msg in [
            (tuple(), tuple(), tuple(), tuple(), None),
            ((12,), _SL[:], (12,), (0,), None),
            ((12,), _SL[:], (11,), (0,), ReshapeErr.VOLUME_MISMATCH),
            ((12,), _SL[1:], (11,), (0,), None),
            ((0,), _SL[:], (0,), (0,), None),
            ((0,), _SL[:], (1, 3), (0,), ReshapeErr.VOLUME_MISMATCH),
            ((3,), _SL[3:], (3,), (0,), ReshapeErr.VOLUME_MISMATCH),
            ((18,), _SL[:], (0,), (0,), ReshapeErr.VOLUME_MISMATCH),
            ((3,), _SL[3:], (0,), (0,), None),
            ((3, 0, 3), _SL[:], (2, 3, 4, 5, 6, 7, 0, 12), (0, 1, 2), None),
            ((3, 0, 3), _SL[:], (0,), (0, 1, 2), None),
            ((12,), _SL[:], (2, 3, 2), (0,), None),
            ((12,), _SL[:], (2, 6), (0,), None),
            ((12,), _SL[:], (4, 3), (0,), None),
            ((12,), _SL[:], (3, 4), (0,), None),
            ((7, 12), _SL[:, :], (7, 12), (0, 1), None),
            ((7, 12), _SL[:, :], (12, 7), (0, 1), None),
            ((12, 11), _SL[:, :], (2, 3, 2, 11), (0, 1), None),
            ((12, 11), _SL[:, :], (2, 3, 11, 2), (0, 1), None),
            ((12, 11), _SL[:, :], (2, 11, 3, 2), (0, 1), None),
            ((12, 11), _SL[:, :], (11, 2, 3, 2), (0, 1), None),
            ((12, 11), _SL[:, :], (2, 3, 2, -1), (0, 1), None),
            ((12, 11), _SL[:, :], (2, 3, -1, 2), (0, 1), None),
            ((12, 11), _SL[:, :], (2, -1, 3, 2), (0, 1), None),
            ((12, 11), _SL[:, :], (-1, 2, 3, 2), (0, 1), None),
            ((12, 11), _SL[:, :], (2, 3, -1, 11), (0, 1), None),
            ((12, 11), _SL[:, :], (2, 3, 11, -1), (0, 1), None),
            ((12, 11), _SL[:, :], (-1, 11, 3, 2), (0, 1), None),
            ((12, 11), _SL[:, :], (11, 2, -1, 2), (0, 1), None),
            ((5, 12), _SL[:, :], (2, 5, 6), (0, 1), None),
            ((12, 7), _SL[:, :], (4, 3, 7), (0, 1), None),
            ((7, 12), _SL[:, :], (7, 3, 4), (0, 1), None),
            ((2, 3, 2), _SL[:, :, :], (12,), (0, 1, 2), None),
            ((2, 3, 2), _SL[:, :, :], (6, 2), (0, 1, 2), None),
            ((2, 3, 2), _SL[:, :, :], (2, 3, 2), (1, 2, 0), None),
            ((2, 3, 2), _SL[:, :, :], (6, 2), (1, 2, 0), None),
            ((2, 3, 2), _SL[:, :, :], (2, 6), (1, 2, 0), ReshapeErr.INCOM_STRIDES),
            ((2, 3, 2), _SL[:, :, :], (12,), (1, 2, 0), ReshapeErr.INCOM_STRIDES),
            ((2, 3, 2), _SL[:, :, :], (3, 2, 2), (1, 0, 2), None),
            ((10, 10, 10), _SL[::-1, ::-1, :], (10, 10, 10), (0, 1, 2), None),
            ((10, 10, 10), _SL[::-1, ::-1, :], (100, 10), (0, 1, 2), None),
            ((10, 10, 10), _SL[::-1, ::-1, ::-1], (1000,), (0, 1, 2), None),
            ((10, 10, 10), _SL[:, :, ::-1], (100, 10), (0, 1, 2), None),
            (
                (10, 10, 10),
                _SL[:, :, ::-1],
                (10, 100),
                (0, 1, 2),
                ReshapeErr.INCOM_STRIDES,
            ),
            (
                (10, 10, 10),
                _SL[::-1, :, ::-1],
                (1000,),
                (0, 1, 2),
                ReshapeErr.INCOM_STRIDES,
            ),
            (
                (10, 10, 10),
                _SL[::-1, ::-1, :],
                (100, 10),
                (1, 0, 2),
                ReshapeErr.INCOM_STRIDES,
            ),
            (
                (10, 10, 10),
                _SL[::-1, ::-1, :],
                (10, 100),
                (0, 1, 2),
                ReshapeErr.INCOM_STRIDES,
            ),
            ((5, 3), _SL[:-1, :], (12,), (0, 1), None),
            ((13, 3), _SL[1:, :], (6, 6), (0, 1), None),
            ((12, 4), _SL[:, :-1], (6, 2, 3), (0, 1), None),
            ((12, 4), _SL[:, :-1], (6, 6), (0, 1), ReshapeErr.INCOM_STRIDES),
            ((7, 6, 5), _SL[:], (70, -1), (0, 1, 2), None),
            ((7, 6, 5), _SL[:], (-1, 70), (0, 1, 2), None),
            ((7, 6, 5), _SL[:], (71, -1), (0, 1, 2), ReshapeErr.DIVISIBILITY_VIOLATION),
            ((7, 6, 5), _SL[:], (-1, 71), (0, 1, 2), ReshapeErr.DIVISIBILITY_VIOLATION),
            ((7, 6, 5), _SL[:], (71, -2), (0, 1, 2), ReshapeErr.NEG_EXTENT),
            ((7, 6, 5), _SL[:], (-2, 71), (0, 1, 2), ReshapeErr.NEG_EXTENT),
            ((7, 6, 5), _SL[:], (-1, 6, -1), (0, 1, 2), ReshapeErr.MULTI_NEG_EXTENTS),
            ((7, 6, 5), _SL[:], (-2, -1, -1), (0, 1, 2), ReshapeErr.NEG_EXTENT),
            ((7, 6, 5), _SL[:], (-2, -1, -2), (0, 1, 2), ReshapeErr.NEG_EXTENT),
            ((7, 6, 5), _SL[:], (-7, 6, -5), (0, 1, 2), ReshapeErr.NEG_EXTENT),
            ((7, 6, 5), _SL[:], (5, 0, -1), (0, 1, 2), ReshapeErr.AMBIGUOUS_NEG_EXTENT),
            ((7, 0, 5), _SL[:], (5, 0, -1), (0, 1, 2), ReshapeErr.AMBIGUOUS_NEG_EXTENT),
            ((7, 6, 5), _SL[:], map, (0, 1, 2), ReshapeErr.TYPE_ERROR),
        ]
        for base_stride_kind in [StridesKind.C, StridesKind.IMPLICIT_C]
    ],
    ids=idfn,
)
def test_reshape(shape, stride_kind, slices, new_shape, permutation, error_msg, itemsize):
    shape = shape.value
    stride_kind = stride_kind.value
    slices = slices.value
    new_shape = new_shape.value
    permutation = permutation.value
    itemsize = itemsize.value

    if stride_kind == StridesKind.IMPLICIT_C:
        layout = StridedLayout(shape, None, itemsize)
    else:
        assert stride_kind == StridesKind.C
        layout = StridedLayout.dense(shape, itemsize)

    np_ref = np.arange(math.prod(shape), dtype=dtype_from_itemsize(itemsize)).reshape(shape)
    has_id_perm = is_id_perm(permutation)
    has_id_slice = is_id_slice(slices)

    if not has_id_perm:
        layout = layout.permuted(permutation)
        np_ref = np_ref.transpose(permutation)
    if not has_id_slice:
        layout = layout[slices]
        np_ref = np_ref[slices]

    has_no_strides = stride_kind == StridesKind.IMPLICIT_C and has_id_perm and has_id_slice
    cmp_layouts(layout, np_ref, has_no_strides)

    if error_msg:
        if error_msg == ReshapeErr.INCOM_STRIDES:
            with pytest.raises(ValueError):
                np_ref.reshape(new_shape, copy=False)
        error_cls = TypeError if error_msg == ReshapeErr.TYPE_ERROR else ValueError
        if error_msg == ReshapeErr.TYPE_ERROR:
            msg = None
        else:
            vol = math.prod(np_ref.shape)
            new_vol = math.prod(new_shape)
            msg = error_msg.value.format(old_volume=vol, new_volume=abs(new_vol))
        with pytest.raises(error_cls, match=msg):
            layout.reshaped(new_shape)
    else:
        reshaped = layout.reshaped(new_shape)
        reshaped_np = np_ref.reshape(new_shape, copy=False)
        assert reshaped_np.ctypes.data == np_ref.ctypes.data

        cmp_layouts(reshaped, reshaped_np, False)
        assert reshaped.itemsize == itemsize
        assert reshaped.slice_offset == layout.slice_offset


@pytest.mark.parametrize(
    (
        "shape",
        "slices",
        "permutation",
        "expected_shape",
        "expected_strides",
        "expected_axis_mask",
        "stride_kind",
        "itemsize",
    ),
    [
        (
            Param("shape", shape),
            Param("slices", slices),
            Param("permutation", permutation),
            Param("expected_shape", expected_shape),
            Param("expected_strides", expected_strides),
            Param("expected_axis_mask", expected_axis_mask),
            Param("stride_kind", stride_kind),
            Param("itemsize", py_rng.choice(_ITEMSIZES)),
        )
        for shape, slices, permutation, expected_shape, expected_strides, expected_axis_mask in [
            ((12,), _SL[:], None, (12,), (1,), "0"),
            ((1, 2, 3, 4, 5), _SL[:], None, (120,), (1,), "01111"),
            ((1, 2, 3, 0, 5), _SL[:], None, (0,), (0,), "01111"),
            ((5, 1, 2, 4, 3), _SL[:, :, :, :, ::-2], None, (40, 2), (3, -2), "01110"),
            ((5, 2, 4, 3), _SL[:, ::-1, :, :], None, (5, 2, 12), (24, -12, 1), "0001"),
            (
                (5, 7, 4, 3),
                _SL[:, ::-1, ::-1, :],
                None,
                (5, 28, 3),
                (84, -3, 1),
                "0010",
            ),
            ((5, 4, 3, 7), _SL[:], (2, 3, 0, 1), (21, 20), (1, 21), "0101"),
            ((5, 4, 3, 7), _SL[:], (3, 2, 0, 1), (7, 3, 20), (1, 7, 21), "0001"),
        ]
        for stride_kind in [StridesKind.C, StridesKind.IMPLICIT_C]
    ],
    ids=idfn,
)
def test_flatten(
    shape,
    slices,
    permutation,
    expected_shape,
    expected_strides,
    expected_axis_mask,
    stride_kind,
    itemsize,
):
    shape = shape.value
    stride_kind = stride_kind.value
    slices = slices.value
    permutation = permutation.value
    expected_shape = expected_shape.value
    expected_strides = expected_strides.value
    expected_axis_mask = expected_axis_mask.value
    itemsize = itemsize.value

    if stride_kind == StridesKind.IMPLICIT_C:
        layout = StridedLayout(shape, None, itemsize)
    else:
        assert stride_kind == StridesKind.C
        layout = StridedLayout.dense(shape, itemsize)

    if not is_id_slice(slices):
        layout = layout[slices]
    if permutation and not is_id_perm(permutation):
        layout = layout.permuted(permutation)

    mask = flatten_mask2str(layout.flattened_axis_mask(), layout.ndim)
    assert mask == expected_axis_mask

    flattened = layout.flattened()
    assert flattened.shape == expected_shape
    assert flattened.strides == expected_strides
    assert flattened.itemsize == itemsize
    assert flattened.slice_offset == layout.slice_offset

    # cannot be flattened any further
    assert flattened.flattened_axis_mask() == 0


@pytest.mark.parametrize(
    (
        "layout_spec_0",
        "layout_spec_1",
        "expected_layout_spec_0",
        "expected_layout_spec_1",
    ),
    [
        (
            layout_spec_0,
            layout_spec_1,
            expected_layout_spec_0,
            expected_layout_spec_1,
        )
        for layout_spec_0, layout_spec_1, expected_layout_spec_0, expected_layout_spec_1 in [
            (
                LayoutSpec(tuple(), StridesKind.C, tuple(), 2),
                LayoutSpec(tuple(), StridesKind.C, tuple(), 4),
                LayoutSpec(tuple(), StridesKind.C, tuple(), 2),
                LayoutSpec(tuple(), StridesKind.C, tuple(), 4),
            ),
            (
                LayoutSpec((255,), StridesKind.IMPLICIT_C, None, 1),
                LayoutSpec((70,), StridesKind.IMPLICIT_C, None, 1),
                LayoutSpec((255,), StridesKind.C, (1,), 1),
                LayoutSpec((70,), StridesKind.C, (1,), 1),
            ),
            (
                LayoutSpec((2, 7, 13, 5), StridesKind.IMPLICIT_C, None, 8),
                LayoutSpec((3, 5, 11, 1), StridesKind.IMPLICIT_C, None, 4),
                LayoutSpec((910,), StridesKind.C, (1,), 8),
                LayoutSpec((165,), StridesKind.C, (1,), 4),
            ),
            (
                LayoutSpec((2, 7, 13, 5), StridesKind.C, (455, 65, 5, 1), 8),
                LayoutSpec((3, 5, 11, 1), StridesKind.IMPLICIT_C, None, 4),
                LayoutSpec((910,), StridesKind.C, (1,), 8),
                LayoutSpec((165,), StridesKind.C, (1,), 4),
            ),
            (
                LayoutSpec((2, 7, 13, 5), StridesKind.C, (455, 65, 5, 1), 8),
                LayoutSpec((3, 5, 11, 1), StridesKind.C, (55, 11, 1, 1), 4),
                LayoutSpec((910,), StridesKind.C, (1,), 8),
                LayoutSpec((165,), StridesKind.C, (1,), 4),
            ),
            (
                LayoutSpec((5, 7, 13, 2), StridesKind.PERMUTED, (1, 65, 5, 455), 8),
                LayoutSpec((3, 5, 11, 1), StridesKind.IMPLICIT_C, None, 4),
                LayoutSpec((5, 91, 2), StridesKind.PERMUTED, (1, 5, 455), 8),
                LayoutSpec((3, 55, 1), StridesKind.C, (55, 1, 1), 4),
            ),
            (
                LayoutSpec((2, 7, 13, 5), StridesKind.C, (455, 65, 5, 1), 8),
                LayoutSpec((11, 1, 3, 5), StridesKind.PERMUTED, (1, 1, 55, 11), 4),
                LayoutSpec((14, 65), StridesKind.C, (65, 1), 8),
                LayoutSpec((11, 15), StridesKind.PERMUTED, (1, 11), 4),
            ),
            (
                LayoutSpec(
                    (4, 5, 11, 2, 3, 7),
                    StridesKind.PERMUTED,
                    (55, 11, 1, 660, 220, 1320),
                    8,
                ),
                LayoutSpec(
                    (3, 8, 5, 6, 7, 9),
                    StridesKind.PERMUTED,
                    (72, 9, 3024, 504, 72, 1),
                    4,
                ),
                LayoutSpec((20, 11, 6, 7), StridesKind.PERMUTED, (11, 1, 220, 1320), 8),
                LayoutSpec((24, 5, 42, 9), StridesKind.PERMUTED, (9, 3024, 72, 1), 4),
            ),
        ]
    ],
    ids=idfn,
)
def test_flatten_together(
    layout_spec_0,
    layout_spec_1,
    expected_layout_spec_0,
    expected_layout_spec_1,
):
    layout_0 = layout_spec_0.layout()
    layout_1 = layout_spec_1.layout()

    mask_0 = layout_0.flattened_axis_mask()
    mask_1 = layout_1.flattened_axis_mask()
    mask = mask_0 & mask_1

    flattened_0 = layout_0.flattened(mask=mask)
    flattened_1 = layout_1.flattened(mask=mask)
    expected_layout_0 = expected_layout_spec_0.layout()
    expected_layout_1 = expected_layout_spec_1.layout()
    assert flattened_0 == expected_layout_0
    assert flattened_0.shape == expected_layout_0.shape
    assert flattened_0.strides == expected_layout_0.strides
    assert flattened_0.itemsize == expected_layout_0.itemsize
    assert flattened_1 == expected_layout_1
    assert flattened_1.shape == expected_layout_1.shape
    assert flattened_1.strides == expected_layout_1.strides
    assert flattened_1.itemsize == expected_layout_1.itemsize

    for flat_layout, flat_spec in zip([flattened_0, flattened_1], [expected_layout_spec_0, expected_layout_spec_1]):
        should_be_c_contig = flat_spec.stride_kind in [
            StridesKind.C,
            StridesKind.IMPLICIT_C,
        ]
        assert flat_layout.is_contiguous_c == should_be_c_contig


@pytest.mark.parametrize(
    (
        "shape",
        "slices",
        "permutation",
        "stride_kind",
        "itemsize",
    ),
    [
        (
            Param("shape", shape),
            Param("slices", slices),
            Param("permutation", permutation),
            Param("stride_kind", stride_kind),
            Param("itemsize", py_rng.choice(_ITEMSIZES)),
        )
        for shape, slices, permutation in [
            (tuple(), tuple(), tuple()),
            ((12,), _SL[:], None),
            ((1, 5, 4, 3), _SL[:], None),
            ((1, 5, 4, 3), _SL[:, -1:, :], None),
            ((1, 5, 4, 3), _SL[:, -1:, :1, 1:2], None),
            ((7, 5, 3), _SL[::-1, 3:2:-1, :], (2, 0, 1)),
            ((7, 5, 3), _SL[:, 3:2, :], (2, 0, 1)),
        ]
        for stride_kind in [StridesKind.C, StridesKind.IMPLICIT_C, StridesKind.F]
    ],
    ids=idfn,
)
def test_squeezed(shape, slices, permutation, stride_kind, itemsize):
    shape = shape.value
    slices = slices.value
    permutation = permutation.value
    stride_kind = stride_kind.value
    itemsize = itemsize.value

    if stride_kind == StridesKind.IMPLICIT_C:
        layout = StridedLayout(shape, None, itemsize)
        order = "C"
    elif stride_kind == StridesKind.C:
        layout = StridedLayout.dense(shape, itemsize, stride_order="C")
        order = "C"
    elif stride_kind == StridesKind.F:
        layout = StridedLayout.dense(shape, itemsize, stride_order="F")
        order = "F"
    else:
        raise ValueError(f"Invalid stride kind: {stride_kind}")

    np_ref = np.arange(math.prod(shape), dtype=dtype_from_itemsize(itemsize)).reshape(shape, order=order)
    has_id_perm = permutation is None or is_id_perm(permutation)
    has_id_slice = is_id_slice(slices)

    if not has_id_slice:
        layout = layout[slices]
        np_ref = np_ref[slices]
    if not has_id_perm:
        layout = layout.permuted(permutation)
        np_ref = np_ref.transpose(permutation)

    has_no_strides = stride_kind == StridesKind.IMPLICIT_C and has_id_perm and has_id_slice
    cmp_layouts(layout, np_ref, has_no_strides)

    squeezed = layout.squeezed()
    squeezed_ref = np_ref.squeeze()
    if math.prod(np_ref.shape) != 0:
        cmp_layouts(squeezed, squeezed_ref, False)
    else:
        assert squeezed.shape == (0,)
        assert squeezed.strides == (0,)
    assert squeezed.slice_offset == layout.slice_offset


@pytest.mark.parametrize(
    (
        "shape",
        "slices",
        "permutation",
        "stride_kind",
        "itemsize",
        "expected_max_itemsize",
        "new_itemsize",
    ),
    [
        (
            Param("shape", shape),
            Param("slices", slices),
            Param("permutation", permutation),
            Param("stride_kind", stride_kind),
            Param("itemsize", itemsize),
            Param("expected_max_itemsize", expected_max_itemsize),
            Param("new_itemsize", new_itemsize),
        )
        for shape, slices, permutation, stride_kind, itemsize, expected_max_itemsize, new_itemsize in [
            ((12,), _SL[:], None, StridesKind.C, 1, 4, 1),
            ((12,), _SL[:], None, StridesKind.IMPLICIT_C, 1, 4, 1),
            ((12,), _SL[:], None, StridesKind.F, 1, 4, 1),
            ((12,), _SL[:], None, StridesKind.C, 4, 16, 8),
            ((12,), _SL[:], None, StridesKind.IMPLICIT_C, 4, 16, 8),
            ((12,), _SL[:], None, StridesKind.F, 4, 16, 8),
            ((16, 5, 4, 6), _SL[:], None, StridesKind.C, 2, 4, 4),
            ((16, 5, 4, 6), _SL[:], None, StridesKind.IMPLICIT_C, 2, 4, 4),
            ((16, 5, 4, 6), _SL[:], None, StridesKind.F, 2, 16, 4),
            ((11, 5, 9), _SL[:, :, -1:], None, StridesKind.C, 2, 2, 2),
            ((11, 5, 9), _SL[:, :, -1:], None, StridesKind.IMPLICIT_C, 2, 2, 2),
            ((11, 5, 9), _SL[:, :, -1:], None, StridesKind.F, 2, 2, 2),
            ((12, 3, 24), _SL[1:, ::-1, 20:], (1, 2, 0), StridesKind.C, 2, 8, 8),
            ((12, 3, 24), _SL[10:, 1:, ::-1], (1, 2, 0), StridesKind.F, 2, 4, 4),
        ]
    ],
    ids=idfn,
)
def test_packed_unpacked(
    shape,
    slices,
    permutation,
    stride_kind,
    itemsize,
    expected_max_itemsize,
    new_itemsize,
):
    shape = shape.value
    slices = slices.value
    permutation = permutation.value
    stride_kind = stride_kind.value
    itemsize = itemsize.value
    expected_max_itemsize = expected_max_itemsize.value
    new_itemsize = new_itemsize.value

    if stride_kind == StridesKind.IMPLICIT_C:
        layout = StridedLayout(shape, None, itemsize)
        order = "C"
    elif stride_kind == StridesKind.C:
        layout = StridedLayout.dense(shape, itemsize, stride_order="C")
        order = "C"
    elif stride_kind == StridesKind.F:
        layout = StridedLayout.dense(shape, itemsize, stride_order="F")
        order = "F"
    else:
        raise ValueError(f"Invalid stride kind: {stride_kind}")

    np_ref = np.arange(math.prod(shape), dtype=dtype_from_itemsize(itemsize)).reshape(shape, order=order)
    has_id_perm = permutation is None or is_id_perm(permutation)
    has_id_slice = is_id_slice(slices)

    if not has_id_slice:
        layout = layout[slices]
        np_ref = np_ref[slices]
    if not has_id_perm:
        layout = layout.permuted(permutation)
        np_ref = np_ref.transpose(permutation)

    has_no_strides = stride_kind == StridesKind.IMPLICIT_C and has_id_perm and has_id_slice
    cmp_layouts(layout, np_ref, has_no_strides)

    axis = layout.stride_order[-1]
    assert layout.max_compatible_itemsize(axis=axis) == expected_max_itemsize
    packed = layout.repacked(new_itemsize, axis=axis)
    packed_ref = (
        np_ref.transpose(layout.stride_order)
        .view(dtype=dtype_from_itemsize(new_itemsize))
        .transpose(inv_permutation(layout.stride_order))
    )
    cmp_layouts(packed, packed_ref, has_no_strides and itemsize == new_itemsize)
    vec_size = new_itemsize // itemsize
    assert packed.slice_offset * vec_size == layout.slice_offset
    unpacked = packed.repacked(itemsize, axis=axis)
    cmp_layouts(unpacked, np_ref, has_no_strides and itemsize == new_itemsize)
    assert unpacked.slice_offset == layout.slice_offset


@pytest.mark.parametrize(
    (
        "shape",
        "slices",
        "stride_kind",
        "itemsize",
        "axes",
    ),
    [
        (
            Param("shape", shape),
            Param("slices", slices),
            Param("stride_kind", stride_kind),
            Param("itemsize", py_rng.choice(_ITEMSIZES)),
            Param("axes", axes),
        )
        for shape, slices in [
            (tuple(), _SL[:]),
            ((7,), _SL[:]),
            ((4, 5, 7, 11), _SL[1:-1, ::-1, 2:-1, ::3]),
        ]
        for stride_kind in [StridesKind.C, StridesKind.IMPLICIT_C, StridesKind.F]
        for num_axes in range(3)
        for axes in itertools.combinations(list(range(len(shape) + num_axes)), num_axes)
    ],
    ids=idfn,
)
def test_unsqueezed_layout(
    shape,
    slices,
    stride_kind,
    itemsize,
    axes,
):
    shape = shape.value
    slices = slices.value
    stride_kind = stride_kind.value
    itemsize = itemsize.value
    axes = tuple(axes.value)

    order = (
        "C" if stride_kind in [StridesKind.C, StridesKind.IMPLICIT_C] else "F" if stride_kind == StridesKind.F else None
    )

    if stride_kind == StridesKind.IMPLICIT_C:
        layout = StridedLayout(shape, None, itemsize)
    else:
        layout = StridedLayout.dense(shape, itemsize, stride_order=order)

    np_ref = np.arange(math.prod(shape), dtype=dtype_from_itemsize(itemsize)).reshape(shape, order=order)
    has_id_slice = is_id_slice(slices)
    if not is_id_slice(slices):
        layout = layout[slices]
        np_ref = np_ref[slices]

    has_no_strides = stride_kind == StridesKind.IMPLICIT_C and has_id_slice
    cmp_layouts(layout, np_ref, has_no_strides)
    layout = layout.unsqueezed(axes)
    np_ref = np.expand_dims(np_ref, axis=axes)
    cmp_layouts(layout, np_ref, has_no_strides and len(axes) == 0)


@pytest.mark.parametrize(
    (
        "shape",
        "slices",
        "new_shape",
        "stride_kind",
        "itemsize",
    ),
    [
        (
            Param("shape", shape),
            Param("slices", slices),
            Param("new_shape", new_shape),
            Param("stride_kind", stride_kind),
            Param("itemsize", py_rng.choice(_ITEMSIZES)),
        )
        for shape, slices, new_shape in [
            (tuple(), _SL[:], tuple()),
            (tuple(), _SL[:], (1,)),
            (tuple(), _SL[:], (17, 1, 5)),
            ((1,), _SL[:], (5,)),
            ((1,), _SL[:], (3, 5, 2)),
            ((7,), _SL[:], (7,)),
            ((7,), _SL[:], (2, 7)),
            ((5, 11), _SL[1:-1, ::-1], (3, 11)),
            ((5, 11), _SL[1:-1, ::-1], (7, 3, 11)),
            ((5, 11), _SL[::-1, 3:4], (5, 7)),
            ((5, 11), _SL[::-1, 3:4], (5, 30)),
            ((5, 11), _SL[::-1, 3:4], (4, 5, 12)),
            ((5, 11), _SL[-1:,], (4, 13, 11)),
        ]
        for stride_kind in [StridesKind.C, StridesKind.IMPLICIT_C, StridesKind.F]
    ],
    ids=idfn,
)
def test_broadcast_layout(
    shape,
    slices,
    new_shape,
    stride_kind,
    itemsize,
):
    shape = shape.value
    slices = slices.value
    new_shape = new_shape.value
    stride_kind = stride_kind.value
    itemsize = itemsize.value

    order = (
        "C" if stride_kind in [StridesKind.C, StridesKind.IMPLICIT_C] else "F" if stride_kind == StridesKind.F else None
    )

    if stride_kind == StridesKind.IMPLICIT_C:
        layout = StridedLayout(shape, None, itemsize)
    else:
        layout = StridedLayout.dense(shape, itemsize, stride_order=order)

    np_ref = np.arange(math.prod(shape), dtype=dtype_from_itemsize(itemsize)).reshape(shape, order=order)
    has_id_slice = is_id_slice(slices)
    if not is_id_slice(slices):
        layout = layout[slices]
        np_ref = np_ref[slices]

    has_no_strides = stride_kind == StridesKind.IMPLICIT_C and has_id_slice
    cmp_layouts(layout, np_ref, has_no_strides)
    layout = layout.broadcast_to(new_shape)
    np_ref = np.broadcast_to(np_ref, new_shape)
    cmp_layouts(layout, np_ref, False)
