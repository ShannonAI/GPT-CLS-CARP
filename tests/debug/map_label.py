#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/debug/map_label.py
@time: 2022/12/06 20:03
@desc:
"""
import os.path


def main():
    origin_dir = "/data2/lixiaoya/gpt_data_models/original_sst2/origin"
    save_dir = "/data2/lixiaoya/gpt_data_models/original_sst2"
    label_map = {"positive": 1, "negative": 0}
    for file in ["train.tsv", "dev.tsv", "test.tsv"]:
        file_path = os.path.join(origin_dir, file)
        with open(file_path, "r") as f:
            datal = [item.strip().split("\t") for item in f.readlines()]

        save_file_path = os.path.join(save_dir, file)
        with open(save_file_path, "w") as f:
            for l in datal:
                f.write(f"{l[0]}\t{label_map[l[1]]}\n")
        print(save_file_path)


if __name__ == "__main__":
    main()
