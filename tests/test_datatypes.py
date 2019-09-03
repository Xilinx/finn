# -*- coding: utf-8 -*-

import finn.core.datatype as dt


def test_datatypes():
    assert dt.DataType.BIPOLAR.allowed(-1)
    assert dt.DataType.BIPOLAR.allowed(0) is False
    assert dt.DataType.BINARY.allowed(-1) is False
    assert dt.DataType.BINARY.allowed(1)
    assert dt.DataType.UINT2.allowed(2)
    assert dt.DataType.UINT2.allowed(10) is False
    assert dt.DataType.UINT3.allowed(5)
    assert dt.DataType.UINT3.allowed(-7) is False
    assert dt.DataType.UINT4.allowed(15)
    assert dt.DataType.UINT4.allowed(150) is False
    assert dt.DataType.UINT8.allowed(150)
    assert dt.DataType.UINT8.allowed(777) is False
    assert dt.DataType.UINT16.allowed(14500)
    assert dt.DataType.UINT16.allowed(-1) is False
    assert dt.DataType.UINT32.allowed(2 ** 10)
    assert dt.DataType.UINT32.allowed(-1) is False
    assert dt.DataType.INT2.allowed(-1)
    assert dt.DataType.INT2.allowed(-10) is False
    assert dt.DataType.INT3.allowed(5) is False
    assert dt.DataType.INT3.allowed(-2)
    assert dt.DataType.INT4.allowed(15) is False
    assert dt.DataType.INT4.allowed(-5)
    assert dt.DataType.INT8.allowed(150) is False
    assert dt.DataType.INT8.allowed(-127)
    assert dt.DataType.INT16.allowed(-1.04) is False
    assert dt.DataType.INT16.allowed(-7777)
    assert dt.DataType.INT32.allowed(7.77) is False
    assert dt.DataType.INT32.allowed(-5)
    assert dt.DataType.INT32.allowed(5)
