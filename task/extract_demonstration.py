#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: task/extract_demonstrations.py
@time: 2022/12/06 20:03
@desc:
"""

import json


def extract_demos(nearest_demo_file: str, save_select_train_file="", demo_clip: int = 16):
    selected_train_instance = []
    set_train_text = set()
    with open(nearest_demo_file, "r") as f:
        datal = [json.loads(item) for item in f.readlines()]

        for data_item in datal:
            train_data_lst = data_item["nearest_neighbors"][:demo_clip]
            train_data_lst = [{"text": item["text"], "label": item["label"]} for item in train_data_lst]
            for train_item in train_data_lst:
                if train_item["text"] not in set_train_text:
                    set_train_text.update([train_item["text"]])
                    selected_train_instance.append(train_item)

    print(len(set_train_text))
    print(len(selected_train_instance))
    # print(set_train_text)
    if save_select_train_file.endswith(".jsonl"):
        with open(save_select_train_file, "w") as f:
            for train_item in selected_train_instance:
                f.write(f"{json.dumps(train_item)}\n")
    elif save_select_train_file.endswith(".tsv"):
        with open(save_select_train_file, "w") as f:
            for train_item in selected_train_instance:
                f.write(f"{train_item['text']}\t{int(train_item['label'])}\n")
    else:
        raise ValueError(save_select_train_file)


def run_r8_ft():
    k_lst = [2, 4, 8, 12, 16, 20, 24]
    for k_item in k_lst:
        nearest_demo_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/r8_nearest_neighbors/bertgcn_test_finetuned_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
        save_select_train_file = f"/data2/lixiaoya/gpt_data_models/data/r8_bertgcn/r8-train-ft{k_item}shot-all-terms.jsonl"
        extract_demos(nearest_demo_file, save_select_train_file, demo_clip=k_item)


def run_r8_simcse():
    k_lst = [2, 4, 8, 12, 16, 20, 24]
    for k_item in k_lst:
        nearest_demo_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/r8_nearest_neighbors/bertgcn_test_sup_simcse_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
        save_select_train_file = f"/data2/lixiaoya/gpt_data_models/data/r8_bertgcn/r8-train-simcse{k_item}shot-all-terms.jsonl"
        extract_demos(nearest_demo_file, save_select_train_file, demo_clip=k_item)


def run_sst2_ft():
    k_lst = [2, 4, 8, 12, 16, 20, 24]
    for k_item in k_lst:
        nearest_demo_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/sst2_nearest_neighbors/test_finetuned_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
        save_select_train_file = f"/data2/lixiaoya/gpt_data_models/data/sst2_original/train-ft{k_item}shot.tsv"
        extract_demos(nearest_demo_file, save_select_train_file, demo_clip=k_item)


def run_sst2_simcse():
    k_lst = [2, 4, 8, 12, 16, 20, 24]
    for k_item in k_lst:
        nearest_demo_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/sst2_nearest_neighbors/test_sup_simcse_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
        save_select_train_file = f"/data2/lixiaoya/gpt_data_models/data/sst2_original/train-simcse{k_item}shot.tsv"
        extract_demos(nearest_demo_file, save_select_train_file, demo_clip=k_item)


if __name__ == "__main__":
    # run_r8_simcse()
    # run_r8_ft()
    run_sst2_ft()
    run_sst2_simcse()
