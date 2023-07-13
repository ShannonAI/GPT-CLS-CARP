#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/merge_result.py
@time: 2022/12/06 20:03
@desc:
"""


def main():
    file_path = "/data2/lixiaoya/hz_data/hz03/data2/data/gpt_text-davinci-003_word_definition_5k5.txt"
    save_path = "/data2/lixiaoya/hz_data/hz03/data2/data/gpt_text-davinci-003_word_definition_5k5_filtered.txt"
    with open(file_path, "r") as f:
        datal = [item.strip().split("\t") for item in f.readlines()]

    filter_datal = []
    two_char_counter = 0
    coll = []
    for data in datal:
        if len(data) == 1:
            print(data)

        if len(data[0]) == 2:
            # print(data)
            two_char_counter += 1
        else:
            coll.append(data[0])
            context = f"{data[0]}\t\t{data[1]}"
            filter_datal.append(context)

    print(two_char_counter)
    with open(save_path, "w") as f:
        for f_data in filter_datal:
            f.write(f"{f_data}\n")
    print(save_path)
    print(len(coll), len(set(coll)))


if __name__ == "__main__":
    main()
