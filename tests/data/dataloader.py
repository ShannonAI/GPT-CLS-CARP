#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/dataloader.py
@time: 2022/12/06 20:03
@desc:
"""

from data.dataloader import SST2Dataloader, AGNewsDataloader, AbsDataloader, TwentyNewsGroupDataloader


def test_load_sst():
    sst2_data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    dataset = SST2Dataloader()
    train_items, dev_items, test_items = dataset.load_data_files(sst2_data_dir)
    print("=*" * 20)
    print(f"# of train is {len(train_items)}")
    print(f"# of dev is {len(dev_items)}")
    print(f"# of test is {len(test_items)}")
    print("=*" * 20)
    print("data instances should be like :")
    print(test_items[0])


def test_split_sst():
    sst_data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    sst_dataloader = SST2Dataloader(sst_data_dir)
    data_random_subset = sst_dataloader.split_subset_data(data_type="test", sample_ratio=0.1, sample_strategy="random")
    print("=*" * 10)
    print("check RANDOM split subset of full dataset.")
    print(data_random_subset[1])
    data_dist_subset = sst_dataloader.split_subset_data(data_type="test", sample_ratio=0.1, sample_strategy="dist")
    print("=*" * 10)
    print("check DIST split subset of full dataset.")
    print(data_dist_subset[1])


def test_split_and_save_sst():
    sst_data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    sst_dataloader = SST2Dataloader(sst_data_dir)
    data_random_subset = sst_dataloader.split_subset_data(data_type="test", sample_ratio=0.1, sample_strategy="random")
    print("=*" * 10)
    print("check RANDOM split subset of full dataset.")
    print(data_random_subset[1])
    data_dist_subset = sst_dataloader.split_subset_data(data_type="test", sample_ratio=0.1, sample_strategy="dist")
    print("=*" * 10)
    print("check DIST split subset of full dataset.")
    print(data_dist_subset[1])

    # split_and_save_subset
    save_file_path = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2/test_dist_182subset.tsv"
    sst_dataloader.split_and_save_subset(save_file_path, data_type="test", sample_ratio=0.1, sample_strategy="dist")

    # split and save subset randomly
    save_file_path = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2/test_random_182subset.tsv"
    sst_dataloader.split_and_save_subset(save_file_path, data_type="test", sample_ratio=0.1, sample_strategy="random")


def test_agnews_loader():
    data_dir = "/data2/lixiaoya/datasets/agnews"
    data_loader = AGNewsDataloader(data_dir)
    data_items = data_loader.load_data_files("test")
    print(len(data_items))
    for idx, item in enumerate(data_items):
        if idx <= 5:
            print("text, label")
            print(item.text, item.label)
        else:
            break
    print(item)


def test_agnews_counter():
    data_dir = "/data2/lixiaoya/datasets/agnews"
    data_loader = AGNewsDataloader(data_dir)
    data_loader.count_data("train")
    print("$" * 40)
    data_loader.count_data("dev")
    print("$" * 40)
    data_loader.count_data("test")
    print("$" * 40)
    test_items = data_loader.load_data_files("test")
    print(test_items[0])


def test_sst_counter():
    data_dir = "/data2/lixiaoya/gpt_data_models/original_sst2"
    data_loader = SST2Dataloader(data_dir)
    data_loader.count_data("train")
    print("$" * 40)
    data_loader.count_data("dev")
    print("$" * 40)
    data_loader.count_data("test")
    print("$" * 40)


def test_subclass():
    data_dir = "/data2/lixiaoya/datasets/agnews"
    data_loader = AGNewsDataloader(data_dir)
    if isinstance(data_loader, AbsDataloader):
        print("Yes | " * 10)
    else:
        print("No |" * 10)
    info = str(data_loader)
    print(info)


def test_20news_loader():
    data_dir = "/data2/lixiaoya/gpt_data_models/20news_bydate"
    data_loader = TwentyNewsGroupDataloader(data_dir)
    data_items = data_loader.load_data_files("test")
    for idx, item in enumerate(data_items):
        if idx <= 2:
            print(item)

    data_loader.count_data("test")
    save_train_path = "/data2/lixiaoya/gpt_data_models/20news_bydate/train_cleaned.jsonl"
    save_dev_path = "/data2/lixiaoya/gpt_data_models/20news_bydate/dev_cleaned.jsonl"
    data_loader.split_train_and_dev(save_train_path, save_dev_path,
                                    sample_ratio=0.2, file_format="jsonl")


if __name__ == "__main__":
    # test_load_sst()
    # test_split_sst()
    # test_split_and_save_sst()
    # test_agnews_loader()

    # conter
    # test_agnews_counter()
    # test_sst_counter()
    # test_subclass()

    test_sst_counter()
    test_20news_loader()
