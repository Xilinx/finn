# -*- coding: utf-8 -*-

from finn.core.datatype import DataType


def test_datatypes():
    assert DataType.BIPOLAR.allowed(-1)
    assert DataType.BIPOLAR.allowed(0) is False
    assert DataType.BINARY.allowed(-1) is False
    assert DataType.BINARY.allowed(1)
    assert DataType.UINT2.allowed(2)
    assert DataType.UINT2.allowed(10) is False
    assert DataType.UINT3.allowed(5)
    assert DataType.UINT3.allowed(-7) is False
    assert DataType.UINT4.allowed(15)
    assert DataType.UINT4.allowed(150) is False
    assert DataType.UINT8.allowed(150)
    assert DataType.UINT8.allowed(777) is False
    assert DataType.UINT16.allowed(14500)
    assert DataType.UINT16.allowed(-1) is False
    assert DataType.UINT32.allowed(2 ** 10)
    assert DataType.UINT32.allowed(-1) is False
    assert DataType.INT2.allowed(-1)
    assert DataType.INT2.allowed(-10) is False
    assert DataType.INT3.allowed(5) is False
    assert DataType.INT3.allowed(-2)
    assert DataType.INT4.allowed(15) is False
    assert DataType.INT4.allowed(-5)
    assert DataType.INT8.allowed(150) is False
    assert DataType.INT8.allowed(-127)
    assert DataType.INT16.allowed(-1.04) is False
    assert DataType.INT16.allowed(-7777)
    assert DataType.INT32.allowed(7.77) is False
    assert DataType.INT32.allowed(-5)
    assert DataType.INT32.allowed(5)
    assert DataType.BINARY.signed() is False
    assert DataType.FLOAT32.signed()
    assert DataType.BIPOLAR.signed()


def test_smallest_possible():
    assert DataType.get_smallest_possible(1) == DataType.BINARY
    assert DataType.get_smallest_possible(1.1) == DataType.FLOAT32
    assert DataType.get_smallest_possible(-1) == DataType.BIPOLAR
    assert DataType.get_smallest_possible(-3) == DataType.INT3
    assert DataType.get_smallest_possible(-3.2) == DataType.FLOAT32
