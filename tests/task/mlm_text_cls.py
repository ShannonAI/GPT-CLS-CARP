#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/task/mlm_text_cls.py
@time: 2022/12/06 20:03
@desc:
"""

from task.mlm_text_cls import MaskedLMTextCLS


def test_init_task(config_path):
    init_task = MaskedLMTextCLS(config_path)
    return init_task


def test_roberta_step1():
    config_path = "/data2/lixiaoya/workspace/gpt-text/config_files/roberta_cls_ZeroShot.json"
    roberta_task = test_init_task(config_path)
    roberta_task.step1_prepare_input()


def test_roberta_step2():
    config_path = "/data2/lixiaoya/workspace/gpt-text/config_files/roberta_cls_ZeroShot.json"
    roberta_task = test_init_task(config_path)
    step1_saved_file = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2/roberta_data.jsonl"
    roberta_task.step2_get_model_results(step1_saved_file)


def test_bert_step1():
    config_path = "/data2/lixiaoya/workspace/gpt-text/config_files/bert_cls_ZeroShot.json"
    bert_task = test_init_task(config_path)
    bert_task.step1_prepare_input()


if __name__ == "__main__":
    # test_init_task()
    # test_roberta_step1()
    # test_roberta_step2()

    # test bert
    test_bert_step1()
