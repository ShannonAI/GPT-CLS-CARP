#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/dataset.py
@time: 2022/12/06 20:03
@desc:
"""

from data.dataloader import SST2Dataloader
from data.dataset import FinetuneMLMDataset, T5Dataset


def test_sst2_finetune_dataset():
    data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    dataloader = SST2Dataloader(data_dir)
    llm_dir = "/data2/lixiaoya/hz_data/hz03/data/models/bert_uncased_base"
    sst2_dataset = FinetuneMLMDataset(dataloader, llm_dir=llm_dir, dataset_name="sst2_1div10", )
    print(sst2_dataset)
    print(len(sst2_dataset))
    for data_idx, data_item in enumerate(sst2_dataset):
        if data_idx > 2:
            break
        print(data_item)


def test_flant5_dataset():
    data_dir = "/data2/lixiaoya/gpt_data_models/original_sst2"
    dataloader = SST2Dataloader(data_dir)
    llm_name_or_dir = "google/flan-t5-small"
    instance_prefix = "SENT:"
    instance_suffix = "Classify the overall sentiment of SENT as Positive or Negative ."
    label_verbalizer = {"Positive": 1, "Negative": 0}
    dataset = T5Dataset(dataloader, llm_name_or_dir, instance_prefix,
                        instance_suffix, label_verbalizer)
    print(len(dataset))
    for data_idx, data_item in enumerate(dataset):
        if data_idx > 2:
            break
        print(data_item)


if __name__ == "__main__":
    # test_sst2_finetune_dataset()
    test_flant5_dataset()
