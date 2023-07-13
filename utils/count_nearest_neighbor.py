#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: count_nearest_neighbor.py

import json
from collections import Counter

from tqdm import tqdm


def count(saved_neighbors_file: str, n_counter: int = 24, clip_idx: int = 16):
    # clip score.
    train_data_cache = {}
    freq_train_data = {}
    test_to_train = {}
    counter_lst = [Counter() for idx in range(n_counter)]
    with open(saved_neighbors_file, "r") as f:
        datalst = [json.loads(item.strip()) for item in f.readlines()]
        for data_item in tqdm(datalst, total=len(datalst)):
            nn_data_lst = data_item["nearest_neighbors"]
            test_to_train[data_item["query_text"]] = nn_data_lst[:clip_idx]
            for idx, nn_data in enumerate(nn_data_lst[:clip_idx]):
                counter_lst[idx].update([nn_data["text"]])
                if nn_data["text"] not in train_data_cache.keys():
                    train_data_cache[nn_data["text"]] = nn_data["label"]
                    freq_train_data[nn_data["text"]] = 1
                else:
                    freq_train_data[nn_data["text"]] += 1
    print("finish counting nearest neighbors")
    # print(counter_lst[0].most_common(24))
    # print(counter_lst[0])
    # print(clip_idx, len(train_data_cache.keys()))
    mean_freq = sum([item for item in freq_train_data.values()]) / float(len(freq_train_data))
    # print(f"mean frequency is {mean_freq}")
    # for clip_freq in [5, 10, 15, 20]:
    #     tmp_freq_lst = [item for item in freq_train_data.values() if item >= clip_freq]
    #     print(clip_freq, len(tmp_freq_lst))
    return train_data_cache, test_to_train, freq_train_data


def run_ft_count():
    # r8
    saved_neighbors_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/r8_nearest_neighbors/bertgcn_test_finetuned_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
    # sst2
    # saved_neighbors_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/sst2_nearest_neighbors/test_finetuned_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
    for clip in [24, 20, 16, 12, 8, 4, 2]:
        print("==" * 20)
        print(clip)
        count(saved_neighbors_file, clip_idx=clip)


def run_simcse_count():
    # r8
    saved_neighbors_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/r8_nearest_neighbors/bertgcn_test_sup_simcse_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
    # sst2
    # saved_neighbors_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/sst2_nearest_neighbors/test_sup_simcse_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
    for clip in [24, 20, 16, 12, 8, 4, 2]:
        print("==" * 20)
        print(clip)
        count(saved_neighbors_file, clip_idx=clip)


def diff_ft_and_simcse():
    # r8
    ft_neighbors_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/r8_nearest_neighbors/bertgcn_test_finetuned_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
    simcse_neighbors_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/r8_nearest_neighbors/bertgcn_test_sup_simcse_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
    # sst2
    # ft_neighbors_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/sst2_nearest_neighbors/test_finetuned_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"
    # simcse_neighbors_file = "/data2/lixiaoya/gpt_data_models/nearest_neighbors/sst2_nearest_neighbors/test_sup_simcse_roberta-large_candtrain_thres0.0_top24_seed9998.jsonl"

    for clip in [24, 20, 16, 12, 8, 4, 2]:
        print("==" * 20)
        print(clip)
        simcse_data, simcse_test_to_train, simcse_freq_train = count(simcse_neighbors_file, clip_idx=clip)
        simcse_data_text = [item for item in simcse_data.keys()]
        ft_data, ft_test_to_train, ft_freq_train = count(ft_neighbors_file, clip_idx=clip)
        ft_data_text = [item for item in ft_data.keys()]
        print(len(set(simcse_data_text) & set(ft_data_text)))
        counter_instance = []
        for simcse_item in simcse_test_to_train.keys():
            simcse_nn_text = [item["text"] for item in simcse_test_to_train[simcse_item]]
            ft_nn_text = [item["text"] for item in ft_test_to_train[simcse_item]]
            assert len(ft_nn_text) == clip
            assert len(simcse_nn_text) == clip
            tmp_same = set(simcse_nn_text) & set(ft_nn_text)
            counter_instance.append(len(tmp_same))
        print(">>> avg", sum(counter_instance) / float(len(counter_instance)))
        print(f">>> max {max(counter_instance)}")
        non_emp = [item for item in counter_instance if item > 0]
        print(f">>> non empty {len(non_emp)}, avg {sum(non_emp) / float(len(non_emp))}")

        for tmp_freq in [1, 5, 10, 15, 20]:
            filter_simcse_freq_train = [key for key, value in simcse_freq_train.items() if value >= tmp_freq]
            filter_ft_freq_train = [key for key, value in ft_freq_train.items() if value >= tmp_freq]
            print(f"$$$ clip {clip} - {tmp_freq} - {len(set(filter_simcse_freq_train) & set(filter_ft_freq_train))}")


if __name__ == "__main__":
    # run_ft_count()
    # run_simcse_count()
    diff_ft_and_simcse()
