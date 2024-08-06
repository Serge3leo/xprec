# vim:set sw=4 ts=8 et fileencoding=ascii:
# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: 2024 Serguei E. Leontiev (leo@sai.msu.ru)

import sys
import warnings

import pytest

import xprec


@pytest.mark.parametrize("n", ['xprec', 'xprec._dd_linalg',
                                'xprec._dd_ufunc'])
def test_dump(n):
    log = f"{n}: {repr(sys.modules[n])}\n"
    for m in [sys.modules[n], xprec]:
        for a in ['__doc__', '__file__', '__loader__', '__name__',
                  '__package__', '__path__', '__spec__', '__version__']:
            log += f"=={a}: {getattr(m, a, 'Not-Attr')}\n"
        log += "-----\n"
    warnings.warn(UserWarning(log))
