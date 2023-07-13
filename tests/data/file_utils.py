#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/file_utils.py
@time: 2022/12/06 20:03
@desc:
"""


def test_file_w_mode(file_path: str):
    lst = ["i like cats.", "i like cats.", "i like cats.", "i like cats.", "i like cats.", ]
    with open(file_path, "w") as f:
        for l in lst:
            f.write(f"{l}\n")


def test_file_a_mode(file_path: str):
    lst = ["i like dogs.", "i like dogs.", "i like dogs.", "i like dogs."]
    with open(file_path, "a") as f:
        for l in lst:
            f.write(f"{l}\n")


if __name__ == "__main__":
    file_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/test_file_utils.txt"
    test_file_w_mode(file_path)
    # test_file_a_mode(file_path)
    # test_file_a_mode(file_path)
    # test_file_a_mode(file_path)
