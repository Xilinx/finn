# -*- coding: utf-8 -*-

import pytest

from finn.skeleton import fib

__author__ = "Yaman Umuroglu"
__copyright__ = "Yaman Umuroglu"
__license__ = "new-bsd"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
