#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: data/result_utils.py
@time: 2022/12/06 20:03
@desc:
"""

import os
from collections import Counter

from data.file_utils import load_jsonl


class ResultAnalysis(object):
    def __init__(self):
        pass

    def count_length(self, file_path: str, key: str = "gpt_returned_result"):
        length_lst = []
        data_items = load_jsonl(file_path)
        num_data = len(data_items)
        short_response_lst = []
        for idx, item in enumerate(data_items):
            text = item[key]
            tokens = text.split()
            length_lst.append(len(tokens))
            if len(tokens) < 10:
                short_response_lst.append(idx)

        print("=$" * 10)
        print(f"avg_len is {sum(length_lst) / num_data}")
        print(f"max_len is {max(length_lst)}")
        print(f"min_len is {min(length_lst)}")
        print(short_response_lst)
        print("=$" * 10)

    def collect_na_label_pred(self, data_dir_prefix: str, file_name: str = "step3_out_of_scope_prediction.jsonl"):
        overall_na_data_lst = []
        text_counter = Counter()
        for seed in ["2333", "1314", "6624", "8866", "9998"]:
            data_dir = f"{data_dir_prefix}{seed}"
            na_label_file = os.path.join(data_dir, file_name)
            na_data_items = load_jsonl(na_label_file)
            for data_item in na_data_items:
                # data_item -> pred_label, gold_label, prompt_text, gpt_returned_result
                text = data_item["prompt_text"].split("INPUT:")[-1]
                text = text.replace("\n", "")
                overall_na_data_lst.append({"seed": seed,
                                            "pred_label": data_item["pred_label"],
                                            "gold_label": data_item["gold_label"],
                                            "prompt_text": data_item["prompt_text"],
                                            "gpt_returned_result": data_item["gpt_returned_result"],
                                            "text": text})
                text_counter.update([text])

        print(text_counter)
