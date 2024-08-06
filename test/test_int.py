# vim:set sw=4 ts=8 et fileencoding=ascii:
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Serguei E. Leontiev (leo@sai.msu.ru)

import math
import sys

import numpy as np
import pytest

import xprec


if sys.version_info >= (3, 9):
    def ulp(x):
        return math.ulp(x)

    smallest_subnormal = math.nextafter(0, math.inf)
else:
    import warnings

    warnings.warn(UserWarning("Python version end-of-life"))

    def ulp(x):
        return sys.float_info.epsilon*x

    smallest_subnormal = sys.float_info.min*sys.float_info.epsilon


np_ints = [np.int8, np.int16, np.int32]

ints_reference = [(npt, *c) for npt in np_ints for c in [
        (0.,    0, 0),
        (0.5,   0, 0),
        (1.,    0, 1),
        (1.5,   0, 1),
        (0.,    1.e-100,    0),
        (2.,    -1.e-100,   1),
        (2.,    1.e-100,    2),
        (np.iinfo(npt).max - 1, 0,          np.iinfo(npt).max - 1),
        (np.iinfo(npt).max - 1, 0.5,        np.iinfo(npt).max - 1),
        (np.iinfo(npt).max,     -1.e-100,   np.iinfo(npt).max - 1),
        (np.iinfo(npt).max,     0,          np.iinfo(npt).max),
        (np.iinfo(npt).max,     1.e-100,    np.iinfo(npt).max),
        (np.iinfo(npt).max,     0.5,        np.iinfo(npt).max),
        (np.iinfo(npt).max + 1, -ulp(np.iinfo(npt).max),
                                            np.iinfo(npt).max),  # noqa: E127
        (np.iinfo(npt).max + 1, -1.e-100,   np.iinfo(npt).max),
        (np.iinfo(npt).max + 1, -smallest_subnormal,
                                            np.iinfo(npt).max),  # noqa: E127
        # Negative only
        (np.iinfo(npt).min,     0,          np.iinfo(npt).min),
        (np.iinfo(npt).min,     -0.5,       np.iinfo(npt).min),
        (np.iinfo(npt).min-1,   +1.e-100,   np.iinfo(npt).min),
        ]]


@pytest.mark.parametrize("npt, a, b, expected", ints_reference)
@pytest.mark.filterwarnings("error")
def test_to_ints(npt, a, b, expected):
    arg = np.float64(a).astype(xprec.ddouble)
    arg += b
    for x, e in [(arg, expected), (-arg, -expected)]:
        if not b:
            assert npt(np.float64(x)) == npt(e)
        assert npt(x) == npt(e)
        if x < 0:
            break


@pytest.mark.parametrize("npt, exp_a, exp_b, int_ref", ints_reference)
@pytest.mark.filterwarnings("error")
def test_from_ints(npt, exp_a, exp_b, int_ref):
    exp = np.float64(exp_a).astype(xprec.ddouble)
    exp += exp_b
    if exp != np.ceil(exp):
        return
    for i, e in [(int_ref, exp), (-int_ref, -exp)]:
        assert e == npt(i).astype(xprec.ddouble)
        if e < 0:
            break


@pytest.mark.skipif(np.lib.NumpyVersion(np.__version__) < '1.24.0',
                    reason="Only NumPy 1.24.0 or higher report "
                    "conversion errors by warning")
@pytest.mark.parametrize("npt, a, b, expected",
                         [(npt, *c) if type(c) is tuple
                          else pytest.param(npt, *(c.values), marks=c.marks)
                          for npt in np_ints for c in
    [                                                           # noqa: E128
        (np.nan, None, None),
        (np.nan, np.nan, None),
        (np.inf, None, None),
        (-np.inf, None, None),
        (np.finfo(np.float64).max, None, None),
        (np.finfo(np.float64).min, None, None),
        (0, xprec.finfo(xprec.ddouble).max, None),
        pytest.param(0, xprec.finfo(xprec.ddouble).min, None,
                     marks=pytest.mark.xfail(reason='xprec.finfo - bug')),
        (np.iinfo(npt).max + 1, None, None),
        (np.iinfo(npt).min - 1, None, None),
        (np.iinfo(np.int32).max + 1, None, None),
        (np.iinfo(np.int32).min - 1, None, None),
        ]])
def test_from_ints_numpy_warning(npt, a, b, expected):
    with pytest.warns(RuntimeWarning) as record:
        arg = np.float64(a).astype(xprec.ddouble)
        if b is not None:
            arg += b
        actual = npt(arg)
        assert expected is None or actual == npt(expected)
    assert len(record) == 1
    assert record[0].message.args[0] == "invalid value encountered in cast"


np_uints = [np.uint8, np.uint16, np.uint32]

uints_reference = [(npt, *c) if type(c) is tuple
                   else pytest.param(npt, *(c.values), marks=c.marks)
                   for npt in np_uints for c in [
        pytest.param(-1, +smallest_subnormal, 0,
                     marks=pytest.mark.xfail(reason='Inaccurate, but '
                                             'otherwise too difficult '
                                             'avoid NumPy warning in this case'
                                             )),
        (-1.,   +np.finfo(np.float64).eps, 0),
        (-0.5,  0, 0),
        (0.,    0, 0),
        (0.5,   0, 0),
        (1.,    0, 1),
        (1.5,   0, 1),
        (0.,    1.e-100,    0),
        (2.,    -1.e-100,   1),
        (2.,    1.e-100,    2),
        (np.iinfo(npt).max - 1, 0,          np.iinfo(npt).max - 1),
        (np.iinfo(npt).max - 1, 0.5,        np.iinfo(npt).max - 1),
        (np.iinfo(npt).max,     -1.e-100,   np.iinfo(npt).max - 1),
        (np.iinfo(npt).max,     0,          np.iinfo(npt).max),
        (np.iinfo(npt).max,     1.e-100,    np.iinfo(npt).max),
        (np.iinfo(npt).max,     0.5,        np.iinfo(npt).max),
        (np.iinfo(npt).max + 1, -smallest_subnormal,
                                            np.iinfo(npt).max),  # noqa: E127
        (np.iinfo(npt).max + 1, -1.e-100,   np.iinfo(npt).max),
        ]]


@pytest.mark.parametrize("npt, a, b, expected", uints_reference)
@pytest.mark.filterwarnings("error")
def test_to_uints(npt, a, b, expected):
    arg = np.float64(a).astype(xprec.ddouble)
    arg += b
    if not b:
        assert npt(np.float64(arg)) == npt(expected)
    assert npt(arg) == npt(expected)


@pytest.mark.parametrize("npt, exp_a, exp_b, uint_ref", uints_reference)
@pytest.mark.filterwarnings("error")
def test_from_uints(npt, exp_a, exp_b, uint_ref):
    exp = np.float64(exp_a).astype(xprec.ddouble)
    exp += exp_b
    if exp != np.ceil(exp):
        return
    assert exp == npt(uint_ref).astype(xprec.ddouble)


@pytest.mark.skipif(np.lib.NumpyVersion(np.__version__) < '1.24.0',
                    reason="Only NumPy 1.24.0 or higher report "
                    "conversion errors by warning")
@pytest.mark.parametrize("npt, a, b, expected",
                         [(npt, *c) for npt in np_uints for c in
    [                                                           # noqa: E128
        (np.nan, None, None),
        (np.nan, np.nan, None),
        (np.inf, None, None),
        (-np.inf, None, None),
        (np.finfo(np.float64).max, None, None),
        (0, xprec.finfo(xprec.ddouble).max, None),
        (-1, None, None),
        (np.iinfo(npt).max + 1, None, None),
        (np.iinfo(np.uint64).max + 1, None, None),
        ]])
def test_from_uints_numpy_warning(npt, a, b, expected):
    with pytest.warns(RuntimeWarning) as record:
        arg = np.float64(a).astype(xprec.ddouble)
        if b is not None:
            arg += b
        actual = npt(arg)
        assert expected is None or actual == npt(expected)
    assert len(record) == 1
    assert record[0].message.args[0] == "invalid value encountered in cast"
