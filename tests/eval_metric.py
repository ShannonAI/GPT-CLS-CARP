#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/eval_metric.py
@time: 2022/12/06 20:03
@desc:
"""

from sklearn.metrics import accuracy_score

def test_str_label_acc_score():
    print("=*"*20)
    print("test_str_label_acc")
    y_pred = ["0", "2", "1", "3" ]
    y_true = ["0", "2", "1", "1"]
    acc_value = accuracy_score(y_true, y_pred)
    print(acc_value)

    y_pred = ["neg", "neg", "neg", "neg"]
    y_true = ["neg", "pos", "neg", "pos"]
    acc_value = accuracy_score(y_true, y_pred)
    print(acc_value)


def test_idx_label_acc_score():
    print("=*"*20)
    print("test_idx_label_acc")
    y_pred = [0, 2, 1, 3]
    y_true = [0, 2, 1, 1]
    acc_value = accuracy_score(y_true, y_pred)
    print(acc_value)

    y_pred = [0, 0, 0, 0]
    y_true = [0, 1, 0, 1]
    acc_value = accuracy_score(y_true, y_pred)
    print(acc_value)


if __name__ == "__main__":
    test_str_label_acc_score()
    test_idx_label_acc_score()