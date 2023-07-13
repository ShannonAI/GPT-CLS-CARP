#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/debug/try.py
@time: 2022/12/06 20:03
@desc:
"""


def test_try():
    try:
        a = [1, 2, 3, 4]
    except:
        a = [1, 2]
    finally:
        a = "110"
    print(a)


def test_try_wo_finally():
    try:
        a = [1, 2, 3, 4]
    except:
        a = [1, 2]
    a = "110"
    print(a)


if __name__ == "__main__":
    test_try()
    test_try_wo_finally()
