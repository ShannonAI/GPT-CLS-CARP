#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: data/config.py
@time: 2022/12/06 20:03
@desc:
"""

import json
from typing import Dict

from transformers import RobertaConfig, BertConfig

from data.dataloader import SST2Dataloader, AGNewsDataloader, TwentyNewsGroupDataloader, R8Dataloader, R52Dataloader, \
    MRDataloader
from data.file_utils import check_file_and_mkdir_for_save
from data.prompt import GPT3ZeroShotPrompt, MaskedLMZeroShotPrompt, Prompt, GPT3FewShotSamplingPrompt, FLANT5Prompt, \
    ChatGPTFewShotSamplingPrompt


class BaseConfig:
    """Config for OpenAi's GPT model service."""

    __slots__ = []

    def __init__(self, key_value_params: Dict = None):
        if key_value_params is not None:
            for key in self.__slots__:
                if key in key_value_params.keys():
                    value = key_value_params[key]
                else:
                    value = None
                self.__setattr__(key, value)
        else:
            for key in self.__slots__:
                self.__setattr__(key, None)

    @classmethod
    def from_json_file(cls, config_path: str):
        """load config from json files."""
        with open(config_path, "r", encoding="utf-8") as f:
            config_items = json.load(f)
        filtered_configs = {key: value for key, value in config_items.items() if key in cls.__slots__}
        return cls(filtered_configs)

    def save_to_json(self, save_path: str):
        """save config to file."""
        config_pairs = self._get_config()
        check_file_and_mkdir_for_save(save_path, resume=True, file_suffix=".json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_pairs, f, sort_keys=True, indent=2, ensure_ascii=False)
        print(f"SAVE CONFIG TO {save_path}")

    def _get_config(self):
        config_pairs = {}
        for slot_key in self.__slots__:
            try:
                slot_value = self.__getattribute__(slot_key)
            except:
                slot_value = None
            finally:
                if isinstance(slot_value, BaseConfig):
                    slot_value = slot_value._get_config()
                    config_pairs[slot_key] = slot_value
                elif isinstance(slot_value, Prompt):
                    slot_value = slot_value._get_config()
                    config_pairs[slot_key] = slot_value
                elif isinstance(slot_value, RobertaConfig) or isinstance(slot_value, BertConfig):
                    slot_value = json.loads(slot_value.to_json_string())
                    config_pairs[slot_key] = slot_value
                elif isinstance(slot_value, AGNewsDataloader) or isinstance(slot_value, SST2Dataloader) or isinstance(
                        slot_value, TwentyNewsGroupDataloader) or isinstance(slot_value, R8Dataloader) or isinstance(
                    slot_value, R52Dataloader) or isinstance(slot_value, MRDataloader):
                    config_pairs[slot_key] = str(slot_value)
                else:
                    config_pairs[slot_key] = slot_value
        return config_pairs

    def __str__(self):
        """return the string."""
        config_data = self._get_config()
        return json.dumps(config_data, indent=2, sort_keys=True, ensure_ascii=False)

    def update_attribute_value(self, attribute_name: str, update_attribute_value):
        origin_value = self.__getattribute__(attribute_name)
        self.__setattr__(attribute_name, update_attribute_value)
        print(f"INFO: update <{attribute_name}> from {origin_value} to {update_attribute_value}")


class GPT3ModelConfig(BaseConfig):
    """Config for OpenAi's GPT model service.
    Slots:
        - openai_api_key: (str, generated by open.ai)
        - init_delay: (int, 1)
        - exponential_base: (int, 2)
        - max_retries: (int, 6)
        - engine_name: (string, text-curie-001)
        - temperature: (float, 0.0)
        - max_tokens: (int, 5046)
        - top_p: (float, 0.01)
        - frequency_penalty: (float, 0.0), useless in text-cls task.
        - presence_penalty: (float, 0.0), useless in text-cls task.
    """

    __slots__ = ["openai_api_key", "init_delay", "exponential_base", "max_retries", "engine_name", "temperature",
                 "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "rate_limit", "rate_limit_delay",
                 "batch_size", "logprobs", "user"]

    def __init__(self, key_value_params: Dict = None):
        super(GPT3ModelConfig, self).__init__(key_value_params)



class GPT3TextCLSTaskConfig(BaseConfig):
    """Config for finish text classification task with GPT-3 APIs.
    Slots:
        - gpt3_model_config: (GPT3ModelConfig Class,)
        - data_dir_path: (str, )
        - prompt_type: (str, )
        - prompt_config: (Prompt,)
        - save_log_dir: (str, )
    """
    __slots__ = ["gpt3_model_config", "dataset_name", "data_dir_path", "prompt_type", "prompt_config", "save_log_dir",
                 "gpt3_backbone", "dataloader"]

    def __init__(self, key_value_params: Dict = None):
        super(GPT3TextCLSTaskConfig, self).__init__(key_value_params)
        if key_value_params["gpt3_backbone"] == "vanilla":
            self.gpt3_model_config = GPT3ModelConfig(key_value_params["gpt3_model_config"])
        else:
            raise ValueError(key_value_params["gpt3_backbone"])
        self.dataset_name = key_value_params["dataset_name"]
        self.data_dir_path = key_value_params["data_dir_path"]
        self.prompt_type = key_value_params["prompt_type"]
        assert self.prompt_type.startswith("zero-shot") or self.prompt_type.startswith("few-shot")

        if "sst2" in self.dataset_name.lower():
            self.dataloader = SST2Dataloader(self.data_dir_path)
        elif "agnews" in self.dataset_name.lower():
            self.dataloader = AGNewsDataloader(self.data_dir_path)
        elif "20news_expire" in self.dataset_name.lower():
            self.dataloader = TwentyNewsGroupDataloader(self.data_dir_path)
        elif "r8" in self.dataset_name.lower():
            self.dataloader = R8Dataloader(self.data_dir_path)
        elif "r52" in self.dataset_name.lower():
            self.dataloader = R52Dataloader(self.data_dir_path)
        elif "mr" in self.dataset_name.lower():
            self.dataloader = MRDataloader(self.data_dir_path)
        else:
            raise ValueError("Please choose from [sst2*, agnews, 20news_expire]")

        # after init dataloader pass though prompt_config.
        key_value_params["prompt_config"]["dataloader"] = self.dataloader
        if self.prompt_type == "zero-shot":
            self.prompt_config = GPT3ZeroShotPrompt(key_value_params["prompt_config"])
        elif self.prompt_type == "few-shot-fix" or self.prompt_type == "few-shot-dynamic":
            self.prompt_config = GPT3FewShotSamplingPrompt(key_value_params["prompt_config"])
        else:
            raise ValueError(f"NOT IMPLEMENTATION {self.prompt_type} !")

        self.save_log_dir = key_value_params["save_log_dir"]


class MaskedLMTextCLSTaskConfig(BaseConfig):
    """RoBERTa for CLS task config."""
    __slots__ = ["model_config", "model_name", "batch_size",
                 "dataset_name", "data_dir_path", "prompt_type", "prompt_config", "save_log_dir"]

    def __init__(self, key_value_params: Dict = None):
        super(MaskedLMTextCLSTaskConfig, self).__init__(key_value_params)

        if self.model_name.lower() == "roberta":
            self.model_config = RobertaConfig(**key_value_params["model_config"])
        elif self.model_name.lower() == "bert":
            self.model_config = BertConfig(**key_value_params["model_config"])
        else:
            raise ValueError

        if self.prompt_type == "zero-shot":
            self.prompt_config = MaskedLMZeroShotPrompt(key_value_params["prompt_config"])
        else:
            raise ValueError

        # assert
        assert self.model_name.lower() == self.model_config.model_type
        assert self.prompt_type.endswith("-shot")


class FLANT5TextCLSConfig(BaseConfig):
    """Config Object for text-classification task with flan t5.
    Slots:
        - dataset_name: (str, )
        - data_dir_path: (str, )
        - llm_name_or_dir: (str, )
        - save_log_dir: (str, )
        - data_instance_prefix: (str, )
        - data_instance_suffix: (str, )
        - label_verbalizer: (Dict, )
        - dataloader: (AbsDataloader, )
    """
    __slots__ = ["dataset_name", "data_dir_path", "llm_name_or_dir", "save_log_dir",
                 "dataloader", "prompt"]

    def __init__(self, key_value_params: Dict = None):
        super(FLANT5TextCLSConfig, self).__init__(key_value_params)

        if "sst2" in self.dataset_name.lower():
            self.dataloader = SST2Dataloader(self.data_dir_path)
        elif "agnews" in self.dataset_name.lower():
            self.dataloader = AGNewsDataloader(self.data_dir_path)
        elif "20news" in self.dataset_name.lower():
            self.dataloader = TwentyNewsGroupDataloader(self.data_dir_path)
        else:
            raise ValueError("Please choose from [sst2*, agnews, 20news]")

        self.prompt = FLANT5Prompt(key_value_params["prompt"])