#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: data/file_utils.py
@time: 2022/12/06 20:03
@desc:
"""
import csv
import json
import os.path
from typing import List, Dict

from tqdm import tqdm

from data.data_utils import DataItem


def get_subdir_names(dir_path: str) -> List[str]:
    """Get a list of immediate subdirectories"""
    return next(os.walk(dir_path))[1]


def save_jsonl(save_file_path: str, data_item_lst: List, resume: bool = False, ):
    check_file_and_mkdir_for_save(save_file_path, resume=resume, file_suffix=".jsonl")
    mode = "w" if not resume else "a"
    with open(save_file_path, mode, encoding='utf-8') as f:
        for data_item in tqdm(data_item_lst, total=len(data_item_lst), ):
            if isinstance(data_item, dict):
                f.write(f"{json.dumps(data_item)}\n")
            elif isinstance(data_item, DataItem):
                # cleaned_text for 20news_expire.
                f.write(f"{json.dumps({'text': data_item.text, 'label': data_item.label})}\n")
            else:
                raise ValueError

    print(f"INFO: SAVE {save_file_path}")


def check_file_and_mkdir_for_save(save_file_path: str, resume: bool = False, file_suffix: str = "jsonl"):
    assert save_file_path.endswith(f"{file_suffix}")
    if os.path.exists(save_file_path) and not resume:
        raise ValueError(f"{save_file_path} already exists ... ...")

    # make the target directory.
    output_dir = os.path.dirname(save_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"INFO: Make directory {output_dir}")


def save_json(save_file_path: str, data_item: Dict, resume: bool = False):
    check_file_and_mkdir_for_save(save_file_path, resume=resume, file_suffix=".json")

    with open(save_file_path, "w", encoding='utf-8') as f:
        json.dump(data_item, f, ensure_ascii=True, indent=2)

    print(f"INFO: save to {save_file_path}")


def load_jsonl(load_file_path: str, offset: int = 0) -> List:
    assert load_file_path.endswith(".jsonl")

    if not os.path.exists(load_file_path):
        raise ValueError(f"{load_file_path} NOT exists ... ...")

    with open(load_file_path, "r", encoding="utf-8") as f:
        data_item_lst = []
        datalines = f.readlines()
        for data_item in tqdm(datalines, total=len(datalines)):
            data_item_lst.append(json.loads(data_item.strip()))

        assert offset >= 0 and offset < len(data_item_lst)
        if offset != 0:
            data_item_lst = data_item_lst[offset:]
        return data_item_lst


def save_tsv(save_file_path: str, data_item_lst: List[DataItem], resume: bool = False, ):
    check_file_and_mkdir_for_save(save_file_path, resume=resume, file_suffix=".tsv")
    mode = "w" if not resume else "a"
    with open(save_file_path, mode, encoding="utf-8") as f:
        for data_idx, data_item in enumerate(data_item_lst):
            if isinstance(data_item, DataItem):
                f.write(f"{data_item.text}\t{data_item.label}\n")
            elif isinstance(data_item, dict):
                if data_idx == 0:
                    data_key_lst = [key_value for key_value in data_item.keys()]
                saved_data_info = "\t".join([data_item[key_value] for key_value in data_key_lst])
                f.write(f"{saved_data_info}\n")
            else:
                raise ValueError

    print(f"INFO: Save to {save_file_path}")


def save_csv(save_file_path: str, data_item_lst: List[DataItem], fieldnames: List[str] = ["label", "title", "desc"],
             writehearder: bool = False, delimiter: str = ",", resume: bool = False):
    check_file_and_mkdir_for_save(save_file_path, resume=resume, file_suffix=".csv")
    mode = "w" if not resume else "a"
    with open(save_file_path, mode, encoding="utf-8", newline="") as f:
        f_writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        if writehearder:
            f_writer.writeheader()
        for item in data_item_lst:
            f_writer.writerow({"title": item.title, "desc": item.desc, "label": str(item.label)})

    print(f"INFO: save to {save_file_path}")


def get_num_lines(file_path: str, strip_line: bool = True) -> int:
    if not os.path.exists(file_path):
        raise ValueError(f"{file_path} NOT exists ... ...")

    with open(file_path, "r") as f:
        datal = f.readlines()

    if strip_line:
        datal = [item.strip() for item in datal]
        datal_filtered = [item for item in datal if len(item) != 0]
        return len(datal_filtered)

    return len(datal)
