#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/model/t5_model.py
@time: 2022/12/06 20:03
@desc:
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration


def test_flan_t5():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    prefix = "SENT:"
    suffix = "Classify the overall sentiment of SENT as Positive or Negative ."
    sent = f"{prefix} If you sometimes like to go to the movies to have fun , Wasabi is a good place to start . {suffix}"
    inputs = tokenizer(sent, return_tensors="pt")
    print("=" * 10)
    print("check inputs")
    print(inputs)
    print("=" * 10)
    outputs = model.generate(**inputs)
    print("%" * 10)
    print("check outputs")
    print(outputs)
    print("%" * 10)
    detoken_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("=" * 10)
    print("detoken_results")
    print(detoken_results)
    print("=" * 10)


def merge_t5_xxl_model_weights():
    ckpt_file_lst = ["/data2/lixiaoya/gpt_data_models/flan-t5-xxl/pytorch_model-00001-of-00005.bin"]
    for ckpt_file in ckpt_file_lst:
        encoder_weight = torch.load(ckpt_file, )
        print(type(encoder_weight))
        exit()


def test_flan_t5_xxl():
    model_dir = "/data2/lixiaoya/gpt_data_models/flan-t5-xxl"
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained(model_dir, )


if __name__ == "__main__":
    # test_flan_t5()
    test_flan_t5_xxl()
    merge_t5_xxl_model_weights()
