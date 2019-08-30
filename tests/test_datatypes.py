# -*- coding: utf-8 -*-

import finn.core.tensor as ten


def test_datatypes():
    assert ten.DataType.BIPOLAR.allowed(-1) is True
    assert ten.DataType.BIPOLAR.allowed(0) is False
    assert ten.DataType.BINARY.allowed(-1) is False
    assert ten.DataType.BINARY.allowed(-1) is True
    assert ten.DataType.UINT2.allowed(2) is True
    assert ten.DataType.UINT2.allowed(10) is False
    assert ten.DataType.UINT3.allowed(5) is True
    assert ten.DataType.UINT3.allowed(-7) is False
    assert ten.DataType.UINT4.allowed(15) is True
    assert ten.DataType.UINT4.allowed(150) is False
    assert ten.DataType.UINT8.allowed(150) is True
    assert ten.DataType.UINT8.allowed(777) is False
    assert ten.DataType.UINT16.allowed(14500) is True
    assert ten.DataType.UINT16.allowed(-1) is False
    assert ten.DataType.UINT32.allowed(2 ** 10) is True
    assert ten.DataType.UINT32.allowed(-1) is False
    assert ten.DataType.INT2.allowed(2) is True
    assert ten.DataType.INT2.allowed(-10) is False
    assert ten.DataType.INT3.allowed(5) is False
    assert ten.DataType.INT3.allowed(-2) is True
    assert ten.DataType.INT4.allowed(15) is False
    assert ten.DataType.INT4.allowed(-5) is True
    assert ten.DataType.INT8.allowed(150) is False
    assert ten.DataType.INT8.allowed(-127) is True
    assert ten.DataType.INT16.allowed(-1.04) is False
    assert ten.DataType.INT16.allowed(-7777) is True
    assert ten.DataType.INT32.allowed(7.77) is False
    assert ten.DataType.INT32.allowed(-5) is True
    assert ten.DataType.INT32.allowed(5) is True
