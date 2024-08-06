# vim:set sw=4 ts=8 et fileencoding=ascii:
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Serguei E. Leontiev (leo@sai.msu.ru)

import numpy as np
import pytest

import xprec


if np.lib.NumpyVersion(np.__version__) < '1.27.0':
    smallest_subnormal = (np.finfo(np.float64).tiny *
                          2**-np.finfo(np.float64).nmant)

    import warnings

    warnings.warn(UserWarning("Wow, that is an old NumPy version!"))
else:
    smallest_subnormal = np.finfo(np.float64).smallest_subnormal


int64_reference = [
        (0.,    0, 0),
        (0.5,   0, 0),
        (1.,    0, 1),
        (1.5,   0, 1),
        (0.,    1.e-100,    0),
        (2.,    -1.e-100,   1),
        (2.,    1.e-100,    2),
        (2**53 - 1.5,   0,              2**53 - 2),
        (2**53 - 1,     1.e-100,        2**53 - 1),
        (2**53 - 1,     -1.e-100,       2**53 - 2),
        (2**53 - 1,     0,              2**53 - 1),
        # 2**53 - 1 - limit full precision `a` (ulp == 1)
        (10**16,        0.5,            10**16),
        (10**16,        1.5,            10**16 + 1),
        # 2**63 - 1024 - limit np.int64 conversion of `a` (ulp == 1024)
        (2**63 - 1024,  0,              2**63 - 1024),
        (2**63 - 1024,  1.e-100,        2**63 - 1024),
        (2**63 - 1024,  -1.e-100,       2**63 - 1024 - 1),
        (2**63 - 1024,  -1,             2**63 - 1024 - 1),
        (2**63 - 1024,  1,              2**63 - 1024 + 1),
        (2**63 - 1024,  1.5,            2**63 - 1024 + 1),
        (2**63 - 1024,  511,            2**63 - 1024 + 511),
        (2**63 - 1024,  511.5,          2**63 - 1024 + 511),
        (2**63 - 1024,  512 - 2**-44,   2**63 - 1024 + 511),
        (2**63 - 1024,  512.5,          2**63 - 1024 + 512),
        (2**63,         -1.5,           2**63-2),
        (2**63,         -1.,            2**63-1),
        (2**63 - 1024,  1022,           np.iinfo(np.int64).max - 1),
        (2**63 - 1024,  1022.5,         np.iinfo(np.int64).max - 1),
        (2**63 - 1024,  1023 - 2**-43,  np.iinfo(np.int64).max - 1),
        (2**63 - 1024,  1023,           np.iinfo(np.int64).max),
        (2**63 - 1024,  1023.5,         np.iinfo(np.int64).max),
        (2**63 - 1024,  1024 - 2**-43,  np.iinfo(np.int64).max),
        (2**63,         -smallest_subnormal, np.iinfo(np.int64).max),
        # Negative only
        (-2**63,        0,              np.iinfo(np.int64).min),
        ]


@pytest.mark.parametrize("a, b, expected", int64_reference)
# @pytest.mark.filterwarnings("error")
def test_to_int64(a, b, expected):
    arg = np.float64(a).astype(xprec.ddouble)
    arg += b
    for x, e in [(arg, expected), (-arg, -expected)]:
        if not b:
            assert np.int64(np.float64(x)) == np.int64(e)
        assert np.int64(x) == np.int64(e)
        if x < 0:
            break


@pytest.mark.parametrize("exp_a, exp_b, int64_ref", int64_reference)
# @pytest.mark.filterwarnings("error")
def test_from_int64(exp_a, exp_b, int64_ref):
    exp = np.float64(exp_a).astype(xprec.ddouble)
    exp += exp_b
    if exp != np.ceil(exp):
        return
    for i, e in [(int64_ref, exp), (-int64_ref, -exp)]:
        assert e == np.int64(i).astype(xprec.ddouble)
        if e < 0:
            break


@pytest.mark.skipif(np.lib.NumpyVersion(np.__version__) < '1.24.0', 
                    reason="Only NumPy 1.24.0 or higher report "
                    "conversion errors by warning")
@pytest.mark.parametrize("a, b, expected", [
        (np.nan, None, None),
        (np.nan, np.nan, None),
        (np.inf, None, None),
        (-np.inf, None, None),
        (np.finfo(np.float64).max, None, None),
        (np.finfo(np.float64).min, None, None),
        (0, xprec.finfo(xprec.ddouble).max, None),
        pytest.param(0, xprec.finfo(xprec.ddouble).min, None,
                     marks=pytest.mark.xfail(reason='xprec.finfo - bug')),
        (2**63, None, None),
        (-2**63, -smallest_subnormal, None),
        ])
def test_from_int64_numpy_warning(a, b, expected):
    with pytest.warns(RuntimeWarning) as record:
        arg = np.float64(a).astype(xprec.ddouble)
        if b is not None:
            arg += b
        actual = np.int64(arg)
        assert expected is None or actual == np.int64(expected)
    assert len(record) == 1
    assert record[0].message.args[0] == "invalid value encountered in cast"


# @pytest.mark.filterwarnings("error")
def test_int64_cvt():
    b = [np.iinfo(np.int64).min, -(1 << 62), 0, 1 << 62,
         np.iinfo(np.int64).max]
    t = b + [k ^ ((1 << i) | (1 << j))
             for i in range(63) for j in range(i + 1) for k in b]
    x = np.array(np.int64(t), dtype=xprec.ddouble)
    np.testing.assert_array_equal(np.int64(t), np.int64(x))
    x += 1.
    x -= 1.
    np.testing.assert_array_equal(np.int64(t), np.int64(x))


# @pytest.mark.filterwarnings("error")
def test_int64_border():
    for i in range(-10, 11):
        if i >= 0 and (i & 1):
            assert np.int64(np.float64(np.int64(2**53 + i))) != 2**53 + i
            assert np.int64(np.float64(np.int64(-(2**53) - i))) != -(2**53) - i
        else:
            assert np.int64(np.float64(np.int64(2**53 + i))) == 2**53 + i
            assert np.int64(np.float64(np.int64(-(2**53) - i))) == -(2**53) - i
        assert 2**53 + i == np.int64(np.asarray(np.int64(2**53 + i),
                                                dtype=xprec.ddouble))
        assert -(2**53) - i == np.int64(np.asarray(np.int64(-(2**53) - i),
                                                   dtype=xprec.ddouble))
        assert 2**53 == np.int64(i + np.asarray(np.int64(2**53 - i),
                                                dtype=xprec.ddouble))
        assert -(2**53) == np.int64(i + np.asarray(np.int64(-(2**53) - i),
                                                   dtype=xprec.ddouble))
    pm = np.iinfo(np.int32).max
    mm = np.iinfo(np.int32).min
    for i in range(0, 11):
        assert pm == np.int64(i + np.asarray(np.int64(pm - i),
                                             dtype=xprec.ddouble))
        assert pm - i == np.int64(-i + np.asarray(np.int64(pm),
                                                  dtype=xprec.ddouble))
        assert mm == np.int64(-i + np.asarray(np.int64(mm + i),
                                              dtype=xprec.ddouble))
        assert mm + i == np.int64(i + np.asarray(np.int64(mm),
                                                 dtype=xprec.ddouble))


uint64_reference = [
        (-1,    +smallest_subnormal, 0),
        (-1.,   +np.finfo(np.float64).eps, 0),
        (0.,    0, 0),
        (0.5,   0, 0),
        (1.,    0, 1),
        (1.5,   0, 1),
        (0.,    1.e-100,    0),
        (2.,    -1.e-100,   1),
        (2.,    1.e-100,    2),
        (2**53 - 1.5,   0,              2**53 - 2),
        (2**53 - 1,     1.e-100,        2**53 - 1),
        (2**53 - 1,     -1.e-100,       2**53 - 2),
        (2**53 - 1,     0,              2**53 - 1),
        # 2**53 - 1 - limit full precision `a` (ulp == 1)
        (10**16,        0.5,            10**16),
        (10**16,        1.5,            10**16 + 1),
        # 2**63 - 1024 - limit np.int64 conversion of `a` (ulp == 1024)
        (2**63 - 1024,  0,              2**63 - 1024),
        (2**63 - 1024,  1.e-100,        2**63 - 1024),
        (2**63 - 1024,  -1.e-100,       2**63 - 1024 - 1),
        (2**63 - 1024,  -1,             2**63 - 1024 - 1),
        (2**63 - 1024,  1,              2**63 - 1024 + 1),
        (2**63 - 1024,  1.5,            2**63 - 1024 + 1),
        (2**63 - 1024,  511,            2**63 - 1024 + 511),
        (2**63 - 1024,  511.5,          2**63 - 1024 + 511),
        (2**63 - 1024,  512 - 2**-44,   2**63 - 1024 + 511),
        (2**63 - 1024,  512.5,          2**63 - 1024 + 512),
        (2**63,         -1.5,           2**63-2),
        (2**63,         -1.,            2**63-1),
        (2**63 - 1024,  1022,           np.iinfo(np.int64).max - 1),
        (2**63 - 1024,  1022.5,         np.iinfo(np.int64).max - 1),
        (2**63 - 1024,  1023 - 2**-43,  np.iinfo(np.int64).max - 1),
        (2**63 - 1024,  1023,           np.iinfo(np.int64).max),
        (2**63 - 1024,  1023.5,         np.iinfo(np.int64).max),
        (2**63 - 1024,  1024 - 2**-43,  np.iinfo(np.int64).max),
        (2**63,         -smallest_subnormal, np.iinfo(np.int64).max),
        (2**64 - 2048,  2046,           np.iinfo(np.uint64).max - 1),
        (2**64 - 2048,  2046.5,         np.iinfo(np.uint64).max - 1),
        (2**64 - 2048,  2047 - 2**-42,  np.iinfo(np.uint64).max - 1),
        (2**64 - 2048,  2047,           np.iinfo(np.uint64).max),
        (2**64 - 2048,  2047.5,         np.iinfo(np.uint64).max),
        (2**64 - 2048,  2048 - 2**-42,  np.iinfo(np.uint64).max),
        (2**64,         -smallest_subnormal, np.iinfo(np.uint64).max),
        ]


@pytest.mark.parametrize("a, b, expected", uint64_reference)
# @pytest.mark.filterwarnings("error")
def test_to_uint64(a, b, expected):
    arg = np.float64(a).astype(xprec.ddouble)
    arg += b
    if not b:
        assert np.uint64(np.float64(arg)) == np.uint64(expected)
    assert np.uint64(arg) == np.uint64(expected)


@pytest.mark.parametrize("exp_a, exp_b, uint64_ref", uint64_reference)
# @pytest.mark.filterwarnings("error")
def test_from_uint64(exp_a, exp_b, uint64_ref):
    exp = np.float64(exp_a).astype(xprec.ddouble)
    exp += exp_b
    if exp != np.ceil(exp):
        return
    assert exp == np.uint64(uint64_ref).astype(xprec.ddouble)


@pytest.mark.skipif(np.lib.NumpyVersion(np.__version__) < '1.24.0', 
                    reason="Only NumPy 1.24.0 or higher report "
                    "conversion errors by warning")
@pytest.mark.parametrize("a, b, expected", [
        (np.nan, None, None),
        (np.nan, np.nan, None),
        (np.inf, None, None),
        (-np.inf, None, None),
        (np.finfo(np.float64).max, None, None),
        (0, xprec.finfo(xprec.ddouble).max, None),
        (-1, None, None),
        (2**64, None, None),
        ])
def test_from_uint64_numpy_warning(a, b, expected):
    with pytest.warns(RuntimeWarning) as record:
        arg = np.float64(a).astype(xprec.ddouble)
        if b is not None:
            arg += b
        actual = np.uint64(arg)
        assert expected is None or actual == np.uint64(expected)
    assert len(record) == 1
    assert record[0].message.args[0] == "invalid value encountered in cast"


def test_uint64_cvt():
    b = [0, 1 << 63, np.iinfo(np.uint64).max]
    t = b + [k ^ ((1 << i) | (1 << j))
             for i in range(64) for j in range(i + 1) for k in b]
    x = np.array(np.uint64(t), dtype=xprec.ddouble)
    for te, xe in zip(t, x):
        assert np.uint64(te) == np.uint64(xe)
    x += 1.
    x -= 1.
    np.testing.assert_array_equal(np.uint64(t), np.uint64(x))


def test_uint64_border():
    for i in range(-10, 11):
        if i >= 0 and (i & 1):
            assert np.uint64(np.float64(np.uint64(2**53 + i))) != 2**53 + i
        else:
            assert np.uint64(np.float64(np.uint64(2**53 + i))) == 2**53 + i
        assert 2**53 + i == np.uint64(np.asarray(np.uint64(2**53 + i),
                                                 dtype=xprec.ddouble))
        assert 2**53 == np.uint64(i + np.asarray(np.uint64(2**53 - i),
                                                 dtype=xprec.ddouble))
    pm = np.iinfo(np.int32).max
    for i in range(0, 11):
        assert pm == np.uint64(i + np.asarray(np.uint64(pm - i),
                                              dtype=xprec.ddouble))
        assert pm - i == np.uint64(-i + np.asarray(np.uint64(pm),
                                                   dtype=xprec.ddouble))
