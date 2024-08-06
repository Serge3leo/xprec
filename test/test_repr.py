# vim:set sw=4 ts=8 et fileencoding=ascii:
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Serguei E. Leontiev (leo@sai.msu.ru)

import pytest

import numpy as np

import xprec


def xprec_floats(x):
    hi = np.float64(x)
    lo = np.float64(x - hi)
    return (hi, lo)


def xprec_fromfloats(hi, lo=None):
    if isinstance(hi, tuple) and (lo is None):
        hi, lo = hi
    return np.float64(hi).astype(xprec.ddouble) + np.float64(lo)


def xprec_thex(x):
    hi, lo = xprec_floats(x)
    return f"({hi.hex()},{lo.hex()})"


def xprec_fromthex(s):
    b = s.find('(')
    m = b + s[b:].find('+')
    if m < b:
        m = b + s[b:].find(',')
    e = m + s[m:].find(')')
    return xprec_fromfloats(float.fromhex(s[b + 1:m]),
                            float.fromhex(s[m + 1:e]))


def _unrepr(x):
    s = repr(x)
    b = s.find('(')
    m = b + s[b:].find('+')
    if m < b:
        m = b + s[b:].find(',')
    e = m + s[m:].find(')')
    return xprec_fromfloats(float(s[b + 1:m]),
                            float(s[m + 1:e]))


@pytest.mark.parametrize("h", [
    "(0x1.27ddbf6271dbep-4,-0x1.10535d4bf6190p-58)",
    ])
def test_repr(h):
    x = xprec_fromthex(h)
    assert x == _unrepr(repr(x))


def test_repr_random():
    hi = np.random.uniform(-1., 1., size=1000)
    lo = np.random.uniform(-1., 1., size=hi.size)*np.finfo(np.float64).eps
    xs = hi.astype(xprec.ddouble) + lo
    for x in xs:
        assert x == _unrepr(repr(x))


def test_thex_random():
    hi = np.random.uniform(-1., 1., size=1000)
    lo = np.random.uniform(-1., 1., size=hi.size)*np.finfo(np.float64).eps
    xs = xprec_fromfloats(hi, lo)
    for x in xs:
        assert x == xprec_fromthex(xprec_thex(x))


def test_floats_random():
    hi = np.random.uniform(-1., 1., size=1000)
    lo = np.random.uniform(-1., 1., size=hi.size)*np.finfo(np.float64).eps
    xs = xprec_fromfloats(hi, lo)
    for x in xs:
        assert x == xprec_fromfloats(xprec_floats(x))
