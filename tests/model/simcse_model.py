#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/model/simcse_model.py
@time: 2022/12/06 20:03
@desc:
"""

from simcse import SimCSE
from tqdm import tqdm

from data.dataloader import SST2Dataloader


def test_build_index():
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    print("load simcse model.")
    data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    dataloader = SST2Dataloader(data_dir)
    train_text = [item.text for item in dataloader.load_data_files("train")]
    print(train_text[0])
    print(f"TOTAL NUM IS : {len(train_text)}")
    model.build_index(train_text)
    results = model.search("He plays guitar.")
    print(results)
    results = model.search(
        "a stirring , funny and finally transporting re imagining of beauty and the beast and 1930s horror films")
    print(results)


def test_index_result():
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    print("load simcse model.")
    data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    dataloader = SST2Dataloader(data_dir)
    train_text = [item.text for item in dataloader.load_data_files("train")]
    print(train_text[0])
    print(f"TOTAL NUM IS : {len(train_text)}")
    model.build_index(train_text)

    search_result_lst = []
    counter = 0
    test_text = [item.text for item in dataloader.load_data_files("test_dist_182subset")]
    for text_item in tqdm(test_text):
        search_result = model.search(text_item, top_k=2)
        if len(search_result) < 2:
            counter += 1
            print(text_item)
    # when top-5, 102

    print(counter)


if __name__ == "__main__":
    # test_build_index()
    test_index_result()
