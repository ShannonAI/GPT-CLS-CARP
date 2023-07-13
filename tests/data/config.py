#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/config.py
@time: 2022/12/06 20:03
@desc:
"""

from data.config import FridayModelConfig, FLANT5TextCLSConfig
from data.config import GPT3ModelConfig, BaseConfig, GPT3TextCLSTaskConfig, MaskedLMTextCLSTaskConfig, \
    ChatFridayModelConfig


def test_basic_config():
    config = BaseConfig()


def test_chat_friday_config():
    config = ChatFridayModelConfig()
    # print(config)
    file = "/data2/lixiaoya/workspace/gpt-text/tests/file/chat_friday_config.json"
    config = ChatFridayModelConfig.from_json_file(file)
    print(config)


def test_init_gpt3_config():
    key_value_pairs = {"engine_name": "lixiaoya'fake engine",
                       "exponential_base": None,
                       "init_delay": None,
                       "max_retries": None,
                       "max_tokens": 10086,
                       "temperature": 1,
                       "top_p": None
                       }
    gpt_config = GPT3ModelConfig(key_value_pairs)
    print("=*" * 10)
    print("test init gpt3 config with key-value pairs.")
    print(f"{gpt_config}")


def test_gpt3_config():
    gpt_config = GPT3ModelConfig()
    print("=*" * 10)
    print("test gpt3 config file.")
    print(f"{gpt_config}")


def test_load_gpt3_config_file():
    print("=*" * 10)
    print("test load gpt3 config file.")
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/gpt_config.json"
    gpt_config = GPT3ModelConfig.load_from_json(GPT3ModelConfig, config_path)
    print(gpt_config)


def test_save_gpt3_config_file():
    print("=*" * 10)
    print("test save gpt3 config file.")
    gpt_config = GPT3ModelConfig()
    save_config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/saved_gpt_config.json"
    gpt_config.save_to_json(save_config_path)


def test_gpt3_cls_task_config():
    config_path = "/data2/lixiaoya/workspace/gpt-text/config_files/gpt3_cls_ZeroShot.json"
    config = GPT3TextCLSTaskConfig.from_json_file(GPT3TextCLSTaskConfig, config_path)
    print("check task config")
    print(config)


def test_gpt3_cls_fewshot_config():
    config_path = "/data2/lixiaoya/workspace/gpt-text/config_files/sst2_1div4/gpt3_fewshot/mlm_neighbor_sample_dynamic/explain_classify_12nearest_davinci003.json"
    config = GPT3TextCLSTaskConfig.from_json_file(config_path)
    save_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/save_gpt3_config.json"
    config.save_to_json(save_path)
    print(save_path)


def test_roberta_cls_task_config():
    # config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/roberta_config.json"
    # config = RoBERTaTextCLSTaskConfig.load_from_json(config_path)
    # print("test_roberta_cls_task_config")
    # print(config)
    # print(config.vocab_size)

    print("=-" * 20)
    config_path = "/data2/lixiaoya/workspace/gpt-text/config_files/roberta_cls_ZeroShot.json"
    config = MaskedLMTextCLSTaskConfig.from_json_file(config_path)
    print("test_roberta_cls_task_config")
    print(config)
    # print(config.vocab_size)


def test_bert_cls_task():
    print("=-" * 20)
    config_path = "/data2/lixiaoya/workspace/gpt-text/config_files/bert_cls_ZeroShot.json"
    config = MaskedLMTextCLSTaskConfig.from_json_file(config_path)
    print("test_bert_cls_task_config")
    print(config)


def test_friday_config():
    config = FridayModelConfig()
    print(config)


def test_friday_from_file():
    config_file = "/data2/lixiaoya/workspace/gpt-text/tests/file/friday_config.json"
    config = FridayModelConfig.from_json_file(config_file)
    print(config)


def test_flant5_cls_config():
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/flan_t5_sst_config.json"
    config = FLANT5TextCLSConfig.from_json_file(config_path)
    print(config)


if __name__ == "__main__":
    # test_basic_config()

    # test gpt-3 model config
    # test_gpt3_config()
    # test_load_gpt3_config_file()
    # test_save_gpt3_config_file()
    # test_init_gpt3_config()

    # test gpt-3 task config
    # test_gpt3_cls_task_config()

    # test roberta config
    # test_roberta_cls_task_config()

    # test_gpt3_cls_fewshot_config()

    # test_friday_config()
    # test_friday_from_file()

    # test_flant5_cls_config()

    test_chat_friday_config()
