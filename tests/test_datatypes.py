# -*- coding: utf-8 -*-

import pytest

import finn.core.tensor as ten

def test_datatypes():
  assert ten.DataType.BIPOLAR.allowed(-1) == True
  assert ten.DataType.BIPOLAR.allowed(0) == False
  assert ten.DataType.BINARY.allowed(-1) == False
  assert ten.DataType.BINARY.allowed(-1) == True
  assert ten.DataType.UINT2.allowed(2) == True
  assert ten.DataType.UINT2.allowed(10) == False
  assert ten.DataType.UINT3.allowed(5) == True
  assert ten.DataType.UINT3.allowed(-7) == False
  assert ten.DataType.UINT4.allowed(15) == True
  assert ten.DataType.UINT4.allowed(150) == False
  assert ten.DataType.UINT8.allowed(150) == True
  assert ten.DataType.UINT8.allowed(777) == False
  assert ten.DataType.UINT16.allowed(14500) == True
  assert ten.DataType.UINT16.allowed(-1) == False
  assert ten.DataType.UINT32.allowed(2 ** 10) == True
  assert ten.DataType.UINT32.allowed(-1) == False
  assert ten.DataType.INT2.allowed(2) == True
  assert ten.DataType.INT2.allowed(-10) == False
  assert ten.DataType.INT3.allowed(5) == False
  assert ten.DataType.INT3.allowed(-2) == True
  assert ten.DataType.INT4.allowed(15) == False
  assert ten.DataType.INT4.allowed(-5) == True
  assert ten.DataType.INT8.allowed(150) == False
  assert ten.DataType.INT8.allowed(-127) == True
  assert ten.DataType.INT16.allowed(-1.04) == False
  assert ten.DataType.INT16.allowed(-7777) == True
  assert ten.DataType.INT32.allowed(7.77) == False
  assert ten.DataType.INT32.allowed(-5) == True
  assert ten.DataType.INT32.allowed(5) == True
