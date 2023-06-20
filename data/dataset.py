#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: dataset.py
@time: 2022/12/06 20:03
@desc:
"""

from typing import Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from data.data_utils import Tokenizer
from data.dataloader import AbsDataloader


class FinetuneMLMDataset(Dataset):
    def __init__(self, dataloader: AbsDataloader, llm_dir: str, dataset_name: str = "sst2_1div10",
                 data_type: str = "train",
                 max_length: int = 512, do_lower_case: bool = False):
        super().__init__()
        assert dataset_name.lower() in ["sst2", "agnews", "20news_expire", "r8", "r52", "mr"]
        self.dataloader = dataloader
        self.data_items = self.dataloader.load_data_files(data_type)
        self.tokenizer = Tokenizer(llm_dir, do_lower_case=do_lower_case, max_len=max_length)
        self.max_length = max_length
        self.do_lower_case = do_lower_case
        self.label_lst = self.dataloader.get_labels()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_lst)}

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx: int):
        data_item = self.data_items[idx]
        input_text = data_item.text
        label = data_item.label
        tokenized_result = self.tokenizer.tokenize_input_batch([input_text])
        # convert list to tensor
        input_ids = torch.LongTensor(tokenized_result["input_ids"][0])
        label = torch.LongTensor([int(self.label_to_idx[label])])
        attention_mask = torch.LongTensor(tokenized_result["attention_mask"][0])
        return input_ids, attention_mask, label


class T5Dataset(Dataset):
    def __init__(self, dataloader: AbsDataloader, llm_name_or_dir: str, instance_prefix: str,
                 instance_suffix: str, label_verbalizer: Dict,
                 dataset_name: str = "sst2", data_type: str = "train", max_length: int = 512,
                 do_lower_case: bool = False):
        super(T5Dataset, self).__init__()
        assert dataset_name.lower() in ["sst2", "agnews", "20news_expire", "r8", "r52"]
        self.dataloader = dataloader
        self.data_items = self.dataloader.load_data_files(data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_dir, do_lower_case=do_lower_case, max_len=max_length)
        self.label_lst = self.dataloader.get_labels()
        self.label_verbalizer = label_verbalizer
        self.instance_prefix = instance_prefix
        self.instance_suffix = instance_suffix

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx: int):
        data_item = self.data_items[idx]
        input_text = data_item.text
        label = data_item.label
        full_text = f"{self.instance_prefix} {input_text} {self.instance_suffix}"
        tensored_input = self.tokenizer(full_text, return_tensors="pt")
        return tensored_input, label
