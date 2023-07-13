#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/time.py
@time: 2022/12/06 20:03
@desc:
"""

import time


def test_sleep():
    print("1", time.time())
    time.sleep(30)
    print("2", time.time())


if __name__ == "__main__":
    test_sleep()
