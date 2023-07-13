#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/data_statistic.py
@time: 2022/12/06 20:03
@desc:
"""

from nltk.tokenize import word_tokenize

from data.file_utils import load_jsonl


def collect_gpt3_response():
    file_path = "/data2/lixiaoya/outputs/gpt_text/sst2_1div10/gpt3_zeroshot/classify_davinci003_random_182subset_seed2333/step2_result.jsonl"
    gpt_results = load_jsonl(file_path)
    num_of_data = len(gpt_results)

    competion_result = set()

    for result_item in gpt_results:
        returned_text = result_item["gpt_returned_result"][0]
        striped_returned_text = returned_text.strip()
        if "\n" in striped_returned_text:
            candidate_lst = striped_returned_text.split("\n")
            striped_returned_text = candidate_lst[-1]
        lowercase_striped_returned_text = striped_returned_text.lower()
        lowercase_tokens = word_tokenize(lowercase_striped_returned_text)
        pred_label = set(lowercase_tokens) & set(["positive", "negative"])
        print(pred_label)
        if len(pred_label) == 0:
            print(">>>", lowercase_striped_returned_text)
        competion_result.update(pred_label)

    print(num_of_data)
    print(len(competion_result))
    print(competion_result)


if __name__ == "__main__":
    collect_gpt3_response()
