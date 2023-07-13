#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/task/gpt3_text_cls.py
@time: 2022/12/06 20:03
@desc:
"""

from task.gpt3_text_cls import GPT3TextCLS


def test_init_gpt3_text_cls():
    task_config_path = "/data2/lixiaoya/workspace/gpt-text/config_files/gpt3_cls_ZeroShot.json"
    task_obj = GPT3TextCLS(task_config_path)


def test_gpt3_text_cls_step1():
    task_config_path = "/data2/lixiaoya/workspace/gpt-text/config_files/gpt3_cls_ZeroShot.json"
    task_obj = GPT3TextCLS(task_config_path)
    print(">>> test step 1 ...")
    task_obj.step1_prepare_input()


def test_gpt3_text_cls_step2():
    task_config_path = "/Users/xiaoya/PycharmProjects/gpt-text/config_files/gpt3_cls_ZeroShot.json"
    task_obj = GPT3TextCLS(task_config_path)
    prompt_data_path = "/Users/xiaoya/Downloads/data.jsonl"
    print(">>> test step 2 ...")
    task_obj.step2_get_gpt3_results(prompt_data_path)


if __name__ == "__main__":
    # test_init_gpt3_text_cls()
    # test_gpt3_text_cls_step1()
    test_gpt3_text_cls_step2()
