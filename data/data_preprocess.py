#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: data/data_preprocess.py
@time: 2022/12/06 20:03
@desc:
"""

import os
import re
import string
from glob import glob

from tqdm import tqdm

from data.data_utils import clean_header
from data.file_utils import get_subdir_names, save_jsonl, load_jsonl

URL_PATTERN = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
EMAIL_PATTERN = re.compile(
    '(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')


class MRProcessor(object):
    def __init__(self):
        self.label_lst = ["0", "1"]
        self.labeltoken_to_labelid_map = {label_token: str(label_idx) for label_idx, label_token in
                                          enumerate(self.label_lst)}

    def step1_save_data_to_jsonl(self, text_file_path: str, label_file_path: str, save_file_dir: str):
        with open(text_file_path, "r") as f:
            text_item_lst = [item.strip() for item in f.readlines()]

        with open(label_file_path, "r") as f:
            label_item_lst = [item.strip() for item in f.readlines()]

        save_train_file = os.path.join(save_file_dir, "mr-train-dev-all-terms.jsonl")
        train_data_lst = []
        for text_item, label_item in zip(text_item_lst, label_item_lst):
            idx, sign, label = label_item.split("\t")
            if sign == "train":
                train_data_lst.append({"label": self.labeltoken_to_labelid_map[label], "text": text_item})

        save_jsonl(save_train_file, train_data_lst)

        save_test_file = os.path.join(save_file_dir, "mr-test-all-terms.jsonl")
        test_data_lst = []
        for text_item, label_item in zip(text_item_lst, label_item_lst):
            idx, sign, label = label_item.split("\t")
            if sign == "test":
                test_data_lst.append({"label": self.labeltoken_to_labelid_map[label], "text": text_item})

        save_jsonl(save_test_file, test_data_lst)


class R52Processor(object):
    """we follow BERTGCN"""

    def __init__(self):
        self.label_lst = ['jobs', 'interest', 'veg-oil', 'cpu', 'bop', 'rubber', 'lei', 'crude', 'strategic-metal',
                          'tin', 'copper', 'fuel', 'meal-feed', 'money-supply', 'iron-steel', 'carcass', 'instal-debt',
                          'lumber', 'platinum', 'nat-gas', 'gnp', 'potato', 'zinc', 'tea', 'retail', 'heat', 'coffee',
                          'grain', 'gas', 'orange', 'sugar', 'pet-chem', 'lead', 'cotton', 'ship', 'nickel', 'alum',
                          'money-fx', 'trade', 'cocoa', 'ipi', 'wpi', 'reserves', 'earn', 'acq', 'housing', 'livestock',
                          'cpi', 'dlr', 'gold', 'income', 'jet']
        self.labeltoken_to_labelid_map = {label_token: str(label_idx) for label_idx, label_token in
                                          enumerate(self.label_lst)}

    def step1_save_data_to_jsonl(self, text_file_path: str, label_file_path: str, save_file_dir: str):
        with open(text_file_path, "r") as f:
            text_item_lst = [item.strip() for item in f.readlines()]

        with open(label_file_path, "r") as f:
            label_item_lst = [item.strip() for item in f.readlines()]

        save_train_file = os.path.join(save_file_dir, "r52-train-dev-all-terms.jsonl")
        train_data_lst = []
        for text_item, label_item in zip(text_item_lst, label_item_lst):
            idx, sign, label = label_item.split("\t")
            if sign == "train":
                train_data_lst.append({"label": self.labeltoken_to_labelid_map[label], "text": text_item})

        save_jsonl(save_train_file, train_data_lst)

        save_test_file = os.path.join(save_file_dir, "r52-test-all-terms.jsonl")
        test_data_lst = []
        for text_item, label_item in zip(text_item_lst, label_item_lst):
            idx, sign, label = label_item.split("\t")
            if sign == "test":
                test_data_lst.append({"label": self.labeltoken_to_labelid_map[label], "text": text_item})

        save_jsonl(save_test_file, test_data_lst)


class R8Processor(object):
    """In this paper, we follow BERTGCN and use their preprocessed data."""

    def __init__(self):
        self.label_lst = ['money-fx', 'acq', 'trade', 'interest', 'ship', 'earn', 'grain', 'crude']
        self.labeltoken_to_labelid_map = {label_token: str(label_idx) for label_idx, label_token in
                                          enumerate(self.label_lst)}

    def step1_save_data_to_jsonl(self, text_file_path: str, label_file_path: str, save_file_dir: str):
        with open(text_file_path, "r") as f:
            text_item_lst = [item.strip() for item in f.readlines()]

        with open(label_file_path, "r") as f:
            label_item_lst = [item.strip() for item in f.readlines()]

        save_train_file = os.path.join(save_file_dir, "r8-train-dev-all-terms.jsonl")
        train_data_lst = []
        for text_item, label_item in zip(text_item_lst, label_item_lst):
            idx, sign, label = label_item.split("\t")
            if sign == "train":
                train_data_lst.append({"label": self.labeltoken_to_labelid_map[label], "text": text_item})

        save_jsonl(save_train_file, train_data_lst)

        save_test_file = os.path.join(save_file_dir, "r8-test-all-terms.jsonl")
        test_data_lst = []
        for text_item, label_item in zip(text_item_lst, label_item_lst):
            idx, sign, label = label_item.split("\t")
            if sign == "test":
                test_data_lst.append({"label": self.labeltoken_to_labelid_map[label], "text": text_item})

        save_jsonl(save_test_file, test_data_lst)


class TwentyNewsProcessor(object):
    def __init__(self, remove_header: bool = True, lower_case: bool = False, remove_punct: bool = False,
                 remove_useless_punct: bool = True):
        self.label_lst = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                          'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
                          'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
                          'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
                          'talk.politics.misc', 'talk.religion.misc']
        self.labeltoken_to_labelid_map = {label_token: str(label_idx) for label_idx, label_token in
                                          enumerate(self.label_lst)}
        self.remove_header = remove_header
        self.remove_punct = remove_punct
        self.lower_case = lower_case
        self.remove_useless_punct = remove_useless_punct

    def step1_save_text_to_jsonl(self, data_dir_path: str, save_file_path: str):

        subdir_lst = get_subdir_names(data_dir_path)
        print(subdir_lst)

        data_item_lst = []
        # step1: save raw text into file.
        for subdir_name in subdir_lst:
            subdir_path = os.path.join(data_dir_path, subdir_name)
            assert subdir_name in self.label_lst
            label_id = self.labeltoken_to_labelid_map[subdir_name]
            file_path_lst = glob(os.path.join(subdir_path, "*"))
            for file_path in file_path_lst:
                with open(file_path, "r", encoding="ISO-8859-1") as f:
                    # the following error is raised when using utf-8 encoding.
                    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 3697: invalid start byte
                    # according to https://stackoverflow.com/questions/42339876/error-unicodedecodeerror-utf-8-codec-cant-decode-byte-0xff-in-position-0-in
                    # change encoding to ISO-8859-1
                    text_lines = [item.strip() for item in f.readlines() if len(item) != 0]
                    text = "\n".join(text_lines)
                    data_item = {"text": text, "label": label_id, "file": file_path}
                    data_item_lst.append(data_item)

        save_jsonl(save_file_path, data_item_lst)
        print(f"INFO: save file to {save_file_path}")

    def step2_clean_text(self, step1_save_file: str, save_file_path: str):
        data_item_lst = load_jsonl(step1_save_file)

        # step2: preprocess text.
        # 1. clean header
        cleaned_data_item_lst = []
        for data_idx, data_item in tqdm(enumerate(data_item_lst), total=len(data_item_lst)):
            raw_text = data_item["text"]
            cleaned_text = raw_text
            try:
                cleaned_text = cleaned_text.split("\n\n")[1:-1]
                cleaned_text = "\n".join(cleaned_text)
            except:
                pass
            if self.remove_header:
                cleaned_text = clean_header(cleaned_text)
            if self.lower_case:
                cleaned_text = cleaned_text.lower()

            cleaned_text = cleaned_text.strip()
            cleaned_text = re.sub(URL_PATTERN, '', cleaned_text)
            cleaned_text = re.sub(EMAIL_PATTERN, '', cleaned_text)
            cleaned_text = cleaned_text.replace("\n>>", " ")
            cleaned_text = cleaned_text.replace("\n>", " ")
            cleaned_text = cleaned_text.replace("\n:", " ")
            cleaned_text = cleaned_text.replace("\n: |> ", " ")
            cleaned_text = re.sub(r'(\d+)', ' ', cleaned_text)
            cleaned_text = re.sub(r'(\s+)', ' ', cleaned_text)
            if self.remove_punct:
                cleaned_text = re.sub(f'[{re.escape(string.punctuation)}]', '', cleaned_text)
            if self.remove_useless_punct:
                punct_lst = '#$&\'*+-/<=>@[\\]^_`{|}~'
                cleaned_text = re.sub(f'[{re.escape(punct_lst)}]', '', cleaned_text)
            cleaned_text = cleaned_text.replace(r"\s\s+", ' ')
            cleaned_text = cleaned_text.replace("()", "")
            cleaned_text = cleaned_text.replace("( )", "")
            if data_idx <= 400:
                print("|||||" * 20)
                print(cleaned_text)

            data_item.update({"cleaned_text": cleaned_text})
            cleaned_data_item_lst.append(data_item)

        save_jsonl(save_file_path, cleaned_data_item_lst)
