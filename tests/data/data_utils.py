#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/data_utils.py
@time: 2022/12/06 20:03
@desc:
"""

from transformers import AutoTokenizer

from data.data_utils import DataItem
from data.data_utils import Detokenizer, Tokenizer
from data.result_utils import ResultAnalysis


def test_detokenizer():
    detokenizer = Detokenizer()
    input_text = "as quiet , patient and tenacious as mr lopez himself , who approaches his difficult , endless work with remarkable serenity and discipline	"
    detok_text = detokenizer.detokenize(input_text)
    print(input_text)
    print(detok_text)

    token_lst = input_text.split(" ")
    detok_text = detokenizer.detokenize(token_lst)
    print(token_lst)
    print(detok_text)


def test_tokenizer():
    llm_dir = "/data2/lixiaoya/hz_data/hz03/data/models/bert_uncased_base"
    tokenizer = Tokenizer(llm_dir, do_lower_case=False, max_len=12)
    print(f"successfully init tokenizer")
    print(tokenizer)
    print("=" * 20)
    text_lst = ["i like apples.", "i like pears.", "I LIKE YOU VERY VERY MUCH AND CATS."]
    tokenized_results = tokenizer.tokenize_input_batch(text_lst)
    print(tokenized_results)
    print("=" * 20)


def count_len():
    analysis = ResultAnalysis()
    file_path = "/data2/lixiaoya/outputs/gpt_text/sst2_1div4_v2/gpt3_fewshot/mlm_neighbor_sample_dynamic/explain_classify_16nearest_davinci003_template2_run2333/step2_result.jsonl"
    analysis.count_length(file_path)

    for seed in ["2333", "6624", "1314", "8866", "9998"]:
        file_path = f"/data2/lixiaoya/outputs/gpt_text/sst2_1div4_v2_friday/gpt3_fewshot/mlm_neighbor_sample_dynamic/explain_classify_16nearest_davinci003_run{seed}/step1_data.jsonl"
        print("#" * 20)
        analysis.count_length(file_path, key="prompt_text")


def test_dataitem():
    instance = DataItem(text="I like cats.", label="0")
    print(instance.text)
    print(instance.label)
    print(instance)


def test_gpt_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "This is a topic classifier.\nFirst, please list key clues and briefly explain the reasoning process for determining the topic of the INPUT sentence (Limit the number of words to 70).\nNext, based on the clues, the reasoning process and the INPUT sentence, classify the topic into one of the eight categories: Money/Foreign Exchange, Acquisitions, Trade, Interest Rates, Shipping, Earnings and Earnings Forecasts, Grain, Crude Oil.\n\nINPUT: yeutter sees u japan trade conflict united states japan brink serious conflict trade especially semiconductors japanese unwillingness public bodies buy u super computers barriers u firms seeking participate eight billion dlr kansai airport project u trade representative clayton yeutter said talking reporters yesterday two day meeting trade ministers review progress made committees set uruguay meeting last september launched new round gatt general agreement tariffs trade talks european community ec commissioner\nClues and the reasoning process: \nTOPIC: Trade"
    token_results = tokenizer(text)
    print(token_results)


if __name__ == "__main__":
    # test_detokenizer()
    # test_tokenizer()
    # count_len()
    # test_dataitem()

    test_gpt_tokenizer()
