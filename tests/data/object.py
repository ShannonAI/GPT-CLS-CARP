#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/object.py
@time: 2022/12/06 20:03
@desc:
"""


class AbsObject:
    __slots__ = ["name", "gender", "hair_color"]

    def __init__(self):
        self.name = "aabb"


class TestObject(AbsObject):
    def __init__(self):
        pass

    def test_func_one(self):
        print(f"print 1111")


class TestBObject(AbsObject):
    __slots__ = ["flavor", "skin_color"]

    def __init__(self):
        pass

    def test_func_two(self):
        print(f"print 2222")


class TestCObject(AbsObject):
    __slots__ = AbsObject.__slots__ + ["flavor", "skin_color"]

    def __init__(self):
        pass


class TestSubClass(TestObject, TestBObject):
    def __init__(self):
        pass


def test_slots():
    abs_obj = AbsObject()
    print("=*" * 20)
    print(f"ABS slot is {abs_obj.__slots__}")
    # print(f"dict is {abs_obj.__dict__}")

    test_a_obj = TestObject()
    print("=*" * 20)
    print(f"TEST A slot is {test_a_obj.__slots__}")
    # print(f"dict is {test_a_obj.__dict__}")

    test_b_obj = TestBObject()
    print("=*" * 20)
    print(f"TEST B slot is {test_b_obj.__slots__}")
    # print(f"dict is {test_b_obj.__dict__}")

    print("check if we can contain the __slot__ before init the class.")
    print(AbsObject.__slots__)

    print("=*" * 20)
    test_c_obj = TestCObject()
    print(f"TEST C slot is {test_c_obj.__slots__}")


def test_inher_two_father_obj():
    obj = TestSubClass()
    obj.test_func_one()
    obj.test_func_two()


if __name__ == "__main__":
    # test_slots()
    test_inher_two_father_obj()
