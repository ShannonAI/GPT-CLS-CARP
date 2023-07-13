#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/result_utils.py
@time: 2022/12/06 20:03
@desc:
"""

from data.result_utils import ResultAnalysis


def test_result_analysis():
    data_dir_prefix = "/data2/lixiaoya/outputs/gpt_text/sst2_1div4_v2_friday/gpt3_fewshot/mlm_neighbor_sample_dynamic/classify_explain_16nearest_davinci003_run"
    analyzer = ResultAnalysis()
    analyzer.collect_na_label_pred(data_dir_prefix)
    print("=$" * 20)
    data_dir_prefix = "/data2/lixiaoya/outputs/gpt_text/sst2_1div4_v2_friday/gpt3_fewshot/mlm_neighbor_sample_dynamic/explain_classify_16nearest_davinci003_run"
    analyzer.collect_na_label_pred(data_dir_prefix)


if __name__ == "__main__":
    test_result_analysis()
