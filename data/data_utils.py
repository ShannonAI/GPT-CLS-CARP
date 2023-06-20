#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: data/data_utils.py
@time: 2022/12/06 20:03
@desc:
"""

import hashlib
import os
import re
from collections import namedtuple
from typing import Dict, List

import numpy as np
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

DataItem = namedtuple("DataItem", ["text", "label", "title", "desc"], defaults=(None, None, None, None))


def collate_to_max_length(seq_lst: List[List[int]], filled_value: int = 0) -> List[List[int]]:
    max_len = max([len(seq_item) for seq_item in seq_lst])
    padded_seq_lst = [seq_item + [filled_value] * (max_len - len(seq_item)) for seq_item in seq_lst]
    return padded_seq_lst


def encode_md5hash(input_str: str) -> str:
    """
    Desc:
        get the md5 value of the input string.
    Param:
        input_str:
    Return:
        md5_value(string) of the input string.
    """
    # <class 'str'>, 722fa7fbb768f990a02d9e705a9d2540
    encode_result = hashlib.md5(input_str.encode())
    encode_md5value = encode_result.hexdigest()
    return encode_md5value


def collate_tensors_to_max_length(batch: List[List[torch.Tensor]], max_len: int = None,
                                  fill_values: List[float] = None) -> \
        List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor), which shape is [seq_length]
        max_len: specify max length
        fill_values: specify filled values of each field
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    # [batch, num_fields]
    # assume the last field is <label>
    lengths = np.array([[len(field_data) for field_data in sample] for sample in batch])
    batch_size, num_fields = lengths.shape
    fill_values = fill_values or [0.0] * (num_fields - 1)
    max_lengths = lengths.max(axis=0)
    if max_len:
        assert max_lengths.max() <= max_len
        max_lengths = np.ones_like(max_lengths) * max_len

    output = [torch.full([batch_size, max_lengths[field_idx]],
                         fill_value=fill_values[field_idx],
                         dtype=batch[0][field_idx].dtype)
              for field_idx in range(num_fields - 1)] + [torch.full([batch_size, ],
                                                                    fill_value=0,
                                                                    dtype=batch[0][-1].dtype)]
    for sample_idx in range(batch_size):
        for field_idx in range(num_fields - 1):
            # seq_length
            data = batch[sample_idx][field_idx]
            output[field_idx][sample_idx][: data.shape[0]] = data
        output[-1][sample_idx] = batch[sample_idx][-1]
    return output


class Detokenizer(object):
    def __init__(self):
        self.detokenizer = TreebankWordDetokenizer()

    def detokenize(self, input_text: str, token_delimiter: str = " ") -> str:
        """
        Desc:
            Untokenizing a text undoes the tokenizing operation, restoring
            punctuation and spaces to the places that people expect them to be.
            Ideally, `untokenize(tokenize(text))` should be identical to `text`,
            except for line breaks.
            credit: https://stackoverflow.com/questions/21948019/python-untokenize-a-sentence
        """
        input_token_lst = input_text.split(token_delimiter)
        detokenized_text = self.detokenizer.detokenize(input_token_lst)
        detokenized_text = detokenized_text.strip()
        detokenized_text = detokenized_text.replace("-lrb-", "(")
        detokenized_text = detokenized_text.replace("-rrb-", ")")
        detokenized_text = detokenized_text.replace("`` ", '" ').replace(" ''", ' "').replace(". . .", "...")
        detokenized_text = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", detokenized_text)
        detokenized_text = re.sub(r" ([.,:;?!%]+)$", r"\1", detokenized_text)
        return detokenized_text.strip()

    def __str__(self):
        return "TreebankWordDetokenizer"


class Tokenizer(object):
    def __init__(self, llm_dir: str, do_lower_case: bool = False, max_len: int = 512, pad_to_max_length: bool = False,
                 add_special_tokens: bool = True, return_offsets_mapping: bool = True):
        self.llm_dir = llm_dir
        self.do_lower_case = do_lower_case
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_dir,
                                                       do_lower_case=self.do_lower_case,
                                                       use_fast=True,
                                                       do_basic_tokenize=False)
        self.pad_to_max_length = pad_to_max_length
        self.add_special_tokens = add_special_tokens
        self.return_offsets_mapping = return_offsets_mapping

    def __len__(self) -> int:
        return len(self._get_vocab_idx2token())

    def _get_vocab_idx2token(self) -> dict:
        return {value: key for key, value in self.tokenizer.vocab.items()}

    def _clip_to_maxlen(self, input_batch: List[str]) -> List[str]:
        clipped_input_tokens_batch = [input_item.split(" ")[: self.max_len] for input_item in input_batch]
        clipped_input_batch = [" ".join(clipped_item) for clipped_item in clipped_input_tokens_batch]
        return clipped_input_batch

    def decode(self, idx_batch: List[List[int]]) -> List[str]:
        vocab_idx2token = self._get_vocab_idx2token()
        str_token_batch = [[vocab_idx2token[item] for item in idx_item] for idx_item in idx_batch]
        text_str = [" ".join(str_token_item) for str_token_item in str_token_batch]
        return text_str

    def tokenize_input_batch(self, input_batch: List[str], ) -> Dict:
        # NOTICE: input_batch should be composed by "[CLS] <text-cls-one> [SEP]" or "[CLS] <text-cls-one> [SEP] <text-cls-two>"
        # TODO(xiaoya): more specific for max_tokens, max_subtokens.
        cliped_input_batch = self._clip_to_maxlen(input_batch)
        # tokenize batch
        tokenizer_output = self.tokenizer.batch_encode_plus(cliped_input_batch,
                                                            pad_to_max_length=self.pad_to_max_length,
                                                            return_offsets_mapping=self.return_offsets_mapping,
                                                            add_special_tokens=self.add_special_tokens,
                                                            return_tensors="pt",
                                                            max_length=self.max_len
                                                            )

        return tokenizer_output

    def __str__(self):
        return "AutoTokenizer"


def clean_header(text):
    text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)

    return text


def word_tokenize_with_protection(input_sent: str, vocab_lst: List[str]):
    pass
