#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: dataloader.py
@time: 2022/12/06 20:03
@desc:
"""

import csv
import os.path
import random
from collections import Counter
from typing import List

from tqdm import tqdm

from data.data_utils import DataItem
from data.file_utils import load_jsonl, save_tsv, save_csv, save_jsonl


class AbsDataloader(object):
    """Abstract class for dataset."""

    def __init__(self):
        pass

    def count_label_dist(self, ):
        raise NotImplementedError

    def load_data_files(self, data_dir_path: str):
        raise NotImplementedError

    @classmethod
    def load_tsv_file(cls, data_file_path: str, delimiter: str = "\t", skip_header: bool = False):
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"FILE {data_file_path} NOT EXISTS ...")

        with open(data_file_path, "r", encoding="utf-8") as f:
            data_items = [tuple(item.replace("\n", "").split(delimiter)) for item in f.readlines()]
            data_items = [DataItem(text=item[0], label=item[1]) for item in data_items]
        if skip_header:
            data_items = data_items[1:]
        return data_items

    @classmethod
    def get_labels(cls):
        raise NotImplementedError

    @classmethod
    def load_csv_file(cls, data_file_path: str, skip_header: bool = False, delimiter: str = ",",
                      concat_text: bool = True):
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"FILE {data_file_path} NOT EXISTS ...")
        data_items = []
        # TODO(xiaoya): change for flex.
        with open(data_file_path, "r", newline='') as csvf:
            file_reader = csv.reader(csvf, delimiter=delimiter, )
            for item in file_reader:
                # label, title, desc.
                if not concat_text:
                    text = [item[1], item[2]]
                else:
                    text = f"{item[1]} {item[2]}"
                data_items.append(DataItem(text=text, label=item[0], title=item[1], desc=item[2]))

                if skip_header:
                    data_items = data_items[1:]
        return data_items

    @classmethod
    def load_jsonl_file(cls, load_file_path: str, text_key: str = "text", label_key: str = "label"):
        data_items = load_jsonl(load_file_path)
        data_items = [DataItem(text=item[text_key], label=item[label_key]) for item in data_items]
        return data_items

    def split_subset_data(self, data_type: str = "test", sample_ratio: float = 0.1, sample_strategy: str = "random",
                          return_left_data: bool = False):
        assert sample_strategy in ["random", "dist", "sample_per_class"]
        data_items = self.load_data_files(data_type=data_type)
        num_of_fullset = len(data_items)
        if sample_ratio > 0 and sample_ratio <= 1:
            num_of_subset = int(num_of_fullset * sample_ratio)
        elif sample_ratio > 1 and sample_strategy == "sample_per_class":
            num_of_subset = int(sample_ratio)
        else:
            raise ValueError(sample_ratio)

        if sample_strategy == "random":
            data_idx_lst = [idx for idx in range(num_of_fullset)]
            random.shuffle(data_idx_lst)
            sampled_data_idx_lst = random.sample(data_idx_lst, k=num_of_subset)
            sampled_data_lst = [data_items[idx] for idx in sampled_data_idx_lst]
            if return_left_data:
                left_data_idx_lst = list(set(data_idx_lst) - set(sampled_data_idx_lst))
                left_data_lst = [data_items[idx] for idx in left_data_idx_lst]
        elif sample_strategy == "dist":
            data_items_by_cate = {}
            for data_idx, data_item in enumerate(data_items):
                if data_item.label not in data_items_by_cate.keys():
                    data_items_by_cate[data_item.label] = [data_idx]
                else:
                    data_items_by_cate[data_item.label].append(data_idx)
            sampled_data_lst = []
            if return_left_data:
                left_data_lst = []
            for data_cate, data_idx_lst_by_cate in data_items_by_cate.items():
                num_sample_cate = int(len(data_idx_lst_by_cate) / float(num_of_fullset) * num_of_subset)
                sampled_cate_idx_lst = random.sample(data_idx_lst_by_cate, k=num_sample_cate)
                sampled_cate_lst = [data_items[idx] for idx in sampled_cate_idx_lst]
                sampled_data_lst.extend(sampled_cate_lst)
                if return_left_data:
                    left_cate_idx_lst = list(set(data_idx_lst_by_cate) - set(sampled_cate_idx_lst))
                    left_cate_lst = [data_items[idx] for idx in left_cate_idx_lst]
                    left_data_lst.extend(left_cate_lst)
        elif sample_strategy == "sample_per_class":
            data_items_by_cate = {}
            for data_idx, data_item in enumerate(data_items):
                if data_item.label not in data_items_by_cate.keys():
                    data_items_by_cate[data_item.label] = [data_idx]
                else:
                    data_items_by_cate[data_item.label].append(data_idx)
            sampled_data_lst = []
            if return_left_data:
                left_data_lst = []
            for data_cate, data_idx_lst_by_cate in data_items_by_cate.items():
                num_sample_cate = num_of_subset
                sampled_cate_idx_lst = random.sample(data_idx_lst_by_cate, k=num_sample_cate)
                sampled_cate_lst = [data_items[idx] for idx in sampled_cate_idx_lst]
                sampled_data_lst.extend(sampled_cate_lst)
                if return_left_data:
                    left_cate_idx_lst = list(set(data_idx_lst_by_cate) - set(sampled_cate_idx_lst))
                    left_cate_lst = [data_items[idx] for idx in left_cate_idx_lst]
                    left_data_lst.extend(left_cate_lst)
        else:
            raise ValueError(f"<sample_strategy> should take the value of [random, dist]")

        if return_left_data:
            return sampled_data_lst, left_data_lst
        return sampled_data_lst

    def split_and_save_subset(self, save_file_path: str, data_type: str = "test", sample_ratio: float = 0.1,
                              sample_strategy: str = "random",
                              file_format: str = "tsv", ):
        sampled_data_lst = self.split_subset_data(data_type=data_type, sample_ratio=sample_ratio,
                                                  sample_strategy=sample_strategy)
        if file_format == "tsv":
            save_tsv(save_file_path, sampled_data_lst)
        elif file_format == "csv":
            save_csv(save_file_path, sampled_data_lst, )
        elif file_format == "jsonl":
            save_jsonl(save_file_path, sampled_data_lst)
        else:
            raise ValueError(file_format)

    def split_train_and_dev(self, save_train_path: str, save_dev_path: str, data_type: str = "train_dev",
                            sample_ratio: float = 0.1, sample_strategy: str = "dist", file_format: str = "tsv"):
        sampled_dev_lst, sampled_train_lst = self.split_subset_data(data_type=data_type,
                                                                    sample_ratio=sample_ratio,
                                                                    sample_strategy=sample_strategy,
                                                                    return_left_data=True)
        if file_format == "tsv":
            save_tsv(save_train_path, sampled_train_lst)
            save_tsv(save_dev_path, sampled_dev_lst)
        elif file_format == "csv":
            save_csv(save_train_path, sampled_train_lst)
            save_csv(save_dev_path, sampled_dev_lst)
        elif file_format == "jsonl":
            save_jsonl(save_train_path, sampled_train_lst)
            save_jsonl(save_dev_path, sampled_dev_lst)
        else:
            raise ValueError(file_format)

    def count_data(self, data_type: str = "test", save_statistic_result_path: str = None):
        data_items = self.load_data_files(data_type)
        total_num = len(data_items)
        length_lst = []
        label_counter = Counter()
        for data_item in tqdm(data_items):
            data_text_len = len(data_item.text.split(" "))
            length_lst.append(data_text_len)
            label_counter.update([data_item.label])
        print("=" * 20)
        print(f"INFO: data-type is {data_type}")
        print("-" * 20)
        print(f"INFO: num-of-data is {total_num}")
        print(f"INFO: length avg is {sum(length_lst) / float(total_num)}")
        print(f"INFO: max len is {max(length_lst)}")
        print(f"INFO: min len is {min(length_lst)}")
        print("-" * 20)
        print(f"INFO: label dist is {label_counter}")
        print("=" * 20)


class SST2Dataloader(AbsDataloader):
    """
    Desc:
        source from the paper 'Parsing With Compositional Vector Grammars'
        domain:
        # data in train/dev/test:
    """

    def __init__(self, data_dir_path: str, filename_suffix: str = "tsv"):
        super(SST2Dataloader, self).__init__()
        self.data_dir_path = data_dir_path
        self.filename_suffix = filename_suffix

    def load_data_files(self, data_type: str = "test", skip_header: bool = False, offset: int = 0) -> List[DataItem]:
        assert data_type in ["train", "dev", "test", "test_1div10", "test_1div4", "train_16_per_class",
                             "train_128_per_class", "train_256_per_class", "train_512_per_class",
                             "train_1024_per_class", "train-ft4shot", "train-ft8shot", "train-ft12shot",
                             "train-ft16shot", "train-ft20shot", "train-ft24shot",
                             "train-simcse4shot", "train-simcse8shot", "train-simcse12shot",
                             "train-simcse16shot", "train-simcse20shot", "train-simcse24shot"]
        data_file = os.path.join(self.data_dir_path, f"{data_type}.{self.filename_suffix}")
        data_items = SST2Dataloader.load_tsv_file(data_file, skip_header=skip_header)
        assert offset >= 0 and offset < len(data_items)
        if offset != 0:
            data_items = data_items[offset:]

        return data_items

    @classmethod
    def get_labels(cls):
        # Original labels are [0, 1] -> ["negative", "positive"]
        # 0 -> negative
        # 1 -> positive
        return ["0", "1"]

    def __str__(self):
        return "Dataloader Object For SST-2."


class TwentyNewsGroupDataloader(AbsDataloader):
    """
    Desc:
        source from the web: https://huggingface.co/datasets/newsgroup/blob/main/newsgroup.py
        refer to BERTGCN, we use By-Date-Version for experiments.
    """

    def __init__(self, data_dir_path: str, filename_suffix: str = "jsonl"):
        super(TwentyNewsGroupDataloader, self).__init__()
        self.data_dir_path = data_dir_path
        self.filename_suffix = filename_suffix

    def load_data_files(self, data_type: str = "test", text_key="cleaned_text", label_key="label",
                        offset: int = 0):
        assert data_type in ["train", "dev", "test", "test_1div10", "train_dev", "train_16_per_class",
                             "train_128_per_class", "train_256_per_class", "train_512_per_class",
                             "train_1024_per_class"]
        data_file = os.path.join(self.data_dir_path, f"{data_type}_cleaned.{self.filename_suffix}")
        data_items = TwentyNewsGroupDataloader.load_jsonl_file(data_file, text_key=text_key, label_key=label_key)
        assert offset >= 0 and offset < len(data_items)
        if offset != 0:
            data_items = data_items[offset:]
        return data_items

    @classmethod
    def get_labels(cls):
        # ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
        #  'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
        #  'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
        #  'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
        #  'talk.politics.misc', 'talk.religion.misc']
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19]

    def __str__(self):
        return "Dataloader Object for 20NewsGroup."


class R8Dataloader(AbsDataloader):
    """
    Desc:
        https://www.cs.umb.edu/˜smimarog/textmining/datasets/
    Data:
        all-terms version, 8 categories.
        5,485 training and 2,189 test documents.
    """

    def __init__(self, data_dir_path: str, filename_suffix: str = "csv"):
        super(R8Dataloader, self).__init__()
        self.data_dir_path = data_dir_path
        self.filename_suffix = filename_suffix

    def load_data_files(self, data_type: str = "test", text_key="text", label_key="label",
                        offset: int = 0):
        assert data_type in ["train", "dev", "test", "test_1div4", "train-dev", "train_16_per_class",
                             "train_128_per_class", "train_256_per_class", "train_512_per_class",
                             "train_1024_per_class", "train-ft4shot", "train-ft8shot", "train-ft12shot",
                             "train-ft16shot", "train-ft20shot", "train-ft24shot",
                             "train-simcse4shot", "train-simcse8shot", "train-simcse12shot",
                             "train-simcse16shot", "train-simcse20shot", "train-simcse24shot"]
        data_file = os.path.join(self.data_dir_path, f"r8-{data_type}-all-terms.jsonl")
        data_items = R8Dataloader.load_jsonl_file(data_file, text_key=text_key, label_key=label_key)
        assert offset >= 0 and offset < len(data_items)
        if offset != 0:
            data_items = data_items[offset:]
        return data_items

    @classmethod
    def get_labels(cls):
        return ["0", "1", "2", "3", "4", "5", "6", "7"]

    def __str__(self):
        return "Dataloader Object For R8."


class R52Dataloader(AbsDataloader):
    """
    Desc:
        https://www.cs.umb.edu/˜smimarog/textmining/datasets/
    Data:
        all-terms version, 52 categories.
        6,532 training and 2,568 test documents.
    """

    def __init__(self, data_dir_path: str, filename_suffix: str = "csv"):
        super(R52Dataloader, self).__init__()
        self.data_dir_path = data_dir_path
        self.filename_suffix = filename_suffix

    def load_data_files(self, data_type: str = "test", text_key="text", label_key="label",
                        offset: int = 0):
        assert data_type in ["train", "dev", "test", "test_1div4", "train-dev", "train_16_per_class",
                             "train_128_per_class", "train_256_per_class", "train_512_per_class",
                             "train_1024_per_class"]
        data_file = os.path.join(self.data_dir_path, f"r52-{data_type}-all-terms.jsonl")
        data_items = R52Dataloader.load_jsonl_file(data_file, text_key=text_key, label_key=label_key)
        assert offset >= 0 and offset < len(data_items)
        if offset != 0:
            data_items = data_items[offset:]
        return data_items

    @classmethod
    def get_labels(cls):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
                '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51']

    def __str__(self):
        return "Dataloader Object for R52."


class AGNewsDataloader(AbsDataloader):
    """
    Desc:
        source from the paper 'Parsing With Compositional Vector Grammars'
        domain:
        # data in train/dev/test:
    """

    def __init__(self, data_dir_path: str, filename_suffix: str = "csv"):
        super(AGNewsDataloader, self).__init__()
        self.data_dir_path = data_dir_path
        self.filename_suffix = filename_suffix

    def load_data_files(self, data_type: str = "test", skip_header: bool = False, delimiter: str = ",",
                        offset: int = 0) -> List[DataItem]:
        assert data_type in ["train", "dev", "test", "test_1div12", "train_16_per_class", "train_128_per_class",
                             "train_256_per_class", "train_512_per_class", "train_1024_per_class"]
        data_file = os.path.join(self.data_dir_path, f"{data_type}.{self.filename_suffix}")
        data_items = AGNewsDataloader.load_csv_file(data_file, skip_header=skip_header, delimiter=delimiter)
        assert offset >= 0 and offset < len(data_items)
        if offset != 0:
            data_items = data_items[offset:]
        return data_items

    @classmethod
    def get_labels(cls):
        # Original labels are [1, 2, 3, 4] -> ['World', 'Sports', 'Business', 'Sci/Tech']
        # 1 -> 'World'
        # 2 -> 'Sports'
        # 3 -> 'Business'
        # 4 -> 'Sci/Tech'
        return ["1", "2", "3", "4"]

    def __str__(self):
        return "Dataloader Object For AGNews."


class MRDataloader(AbsDataloader):
    """
    Desc:
        https://www.cs.umb.edu/˜smimarog/textmining/datasets/
    Data:
        2 categories.
    """

    def __init__(self, data_dir_path: str, filename_suffix: str = "csv"):
        super(MRDataloader, self).__init__()
        self.data_dir_path = data_dir_path
        self.filename_suffix = filename_suffix

    def load_data_files(self, data_type: str = "test", text_key="text", label_key="label",
                        offset: int = 0):
        assert data_type in ["train", "dev", "test", "test_1div4", "train-dev", "train_16_per_class",
                             "train_128_per_class", "train_256_per_class", "train_512_per_class",
                             "train_1024_per_class"]
        data_file = os.path.join(self.data_dir_path, f"mr-{data_type}-all-terms.jsonl")
        data_items = R8Dataloader.load_jsonl_file(data_file, text_key=text_key, label_key=label_key)
        assert offset >= 0 and offset < len(data_items)
        if offset != 0:
            data_items = data_items[offset:]
        return data_items

    @classmethod
    def get_labels(cls):
        return ['0', '1']

    def __str__(self):
        return "Dataloader Object for MR (Movie Review)."
