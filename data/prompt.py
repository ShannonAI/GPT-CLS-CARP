#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: data/object.py
@time: 2022/12/06 20:03
@desc:
"""

import json
import os
import random
from typing import Dict, List, Union

from nltk.tokenize import word_tokenize, sent_tokenize

from data.data_retriever import FinetunedMLMRetriever, SimCSERetriever
from data.data_utils import Detokenizer, Tokenizer, DataItem, encode_md5hash
from data.dataloader import SST2Dataloader


# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')


class Prompt(object):
    """
    Desc:
        This is an ABS class for input prompt.
    Slots:
        - model_backbone: (str,) should choose value from ["gpt-3", "roberta"].
        - prompt_strategy: (str,) should choose value from ["0-shot", "1-shot",].
        - instance_num: (int,) should choose value in the range of [0, #-train-instance].
        - instance_strategy: (str,) should choose value from ["random", "topk-sentence-score"].
        - gradient_update: (bool,) should choose value from ["False", "True"].
    """

    __slots__ = ["model_backbone", "prompt_strategy", "instance_num", "instance_strategy", "gradient_update"]

    def __init__(self, key_value_params: Dict = None):
        # instance_num: int = 0, instance_strategy: str = "random_train", gradient_update: bool = False):
        if key_value_params is not None:
            for key in self.__slots__:
                if key in key_value_params.keys():
                    self.__setattr__(key, key_value_params[key])
                    key_value_params.pop(key)
                else:
                    self.__setattr__(key, None)
            if len(key_value_params) != 0:
                raise ValueError(key_value_params)

    def get_model_input(self, ):
        raise NotImplementedError

    def map_predicted_verbalizer_to_label(self):
        raise NotImplementedError

    def _get_config(self):
        config_pairs = {}
        for slot_key in self.__slots__:
            if slot_key in ["tokenizer", "data_retriever", "detokenizer"]:
                slot_value = None
            elif slot_key in ["dataloader"]:
                slot_value = str(self.__getattribute__(slot_key))
            else:
                try:
                    slot_value = self.__getattribute__(slot_key)
                except:
                    slot_value = None
            config_pairs[slot_key] = slot_value
        return config_pairs

    def __str__(self):
        """return the string."""
        config_data = self._get_config()
        return json.dumps(config_data, indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, config_path: str):
        """load config from json assets."""
        with open(config_path, "r", encoding="utf-8") as f:
            config_items = json.load(f)
        filtered_configs = {key: value for key, value in config_items.items() if key in cls.__slots__}
        return cls(filtered_configs)

    def save_to_json(self, save_path: str):
        """save config to file."""
        config_pairs = self._get_config()
        if os.path.exists(save_path):
            raise FileExistsError(f"{save_path}")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_pairs, f, sort_keys=True, indent=2, ensure_ascii=False)
        print(f"SAVE CONFIG TO {save_path}")


class MaskedLMPrompt(Prompt):
    """
    Desc:
        prompt for the RoBERTa Model.
    Slots:

    """
    __slots__ = Prompt.__slots__ + ["task_description", "delimiter", "llm_dir", "do_lower_case",
                                    "without_cls_sep_tokens", "max_len", "add_special_tokens", "tokenizer",
                                    "start_token_index", "num_of_token_in_vocab"]

    def __init__(self, key_value_params: Dict = None):
        super(MaskedLMPrompt, self).__init__(key_value_params)
        self.tokenizer = Tokenizer(self.llm_dir, do_lower_case=self.do_lower_case, )
        self.start_token_index = len(self.tokenizer) - 1
        self.without_cls_sep_tokens = self.without_cls_sep_tokens
        self.num_of_token_in_vocab = len(self.tokenizer)

    def get_model_input(self, ):
        raise NotImplementedError


class MaskedLMZeroShotPrompt(MaskedLMPrompt):
    """
    Desc:
        zero-shot prompt for the RoBERTa model.
    Slots:
        - task_description: (str,)
        -
    """
    __slots__ = MaskedLMPrompt.__slots__

    def __init__(self, key_value_params: Dict = None):
        super(MaskedLMZeroShotPrompt, self).__init__(key_value_params)
        # check the config values.
        assert self.prompt_strategy == "zero-shot"
        assert self.instance_num == 0
        assert self.instance_strategy == "NULL"
        assert self.gradient_update is False

    def get_model_input(self, instance_text_lst: List[str]) -> Dict:
        input_batch = [f"{self.task_description}{self.delimiter}{instance_text}" for instance_text in
                       instance_text_lst]
        tokenized_output = self.tokenizer.tokenize_input_batch(input_batch)
        return tokenized_output


class GPT3ZeroShotPrompt(Prompt):
    """
    Desc:
        zero-shot prompt for gpt-3.
    Slots:
        - task_description: (str,)
        - verbalizer: (dict, )
            A function that maps a label to the text (a.k.a. label words).
    """
    __slots__ = Prompt.__slots__ + ["task_description", "delimiter", "verbalizer", "inverse_verbalizer", "detokenizer",
                                    "prompt_pattern", "verbalizer_position_idx", "non_verbalizer", "dataloader",
                                    "feasible_verbalizer", "prompt_suffix"]

    def __init__(self, key_value_params: Dict = None):
        super(GPT3ZeroShotPrompt, self).__init__(key_value_params)
        # check the config values.
        assert self.model_backbone == "gpt-3"
        assert self.prompt_strategy == "zero-shot"
        assert self.instance_num == 0
        assert self.instance_strategy == "NULL"
        assert self.gradient_update is False
        if self.feasible_verbalizer is None:
            self.feasible_verbalizer = {label_id: label_token.lower() for label_id, label_token in
                                        self.verbalizer.items()}
        self.inverse_verbalizer = {}

        for label_symbol, label_word in self.feasible_verbalizer.items():
            # k denote labels, like 1, 2, 3.
            if isinstance(label_word, list):
                for token in label_word:
                    assert not any(element.isupper() for element in token)
                    self.inverse_verbalizer[token] = label_symbol
            elif isinstance(label_word, str):
                assert not any(element.isupper() for element in label_word)
                self.inverse_verbalizer[label_word] = label_symbol
            else:
                raise ValueError(self.inverse_verbalizer)
        self.detokenizer = Detokenizer()

    def get_model_input(self, instance_text: str, need_detokenize: bool = True) -> str:
        """
        Desc:

        Args:
            - instance_text: (str,)
            - need_detokenize:
        """
        # TODO (xiaoya): use template.
        if need_detokenize:
            instance_text = self.detokenizer.detokenize(instance_text)

        model_input_instance = self.prompt_pattern.replace("<TASK-DESC>", self.task_description)
        model_input_instance = model_input_instance.replace("<DELIMITER>", self.delimiter)
        model_input_instance = model_input_instance.replace("<INPUT-TEXT>", instance_text)
        if self.prompt_suffix is not None:
            model_input_instance = model_input_instance + self.prompt_suffix
        return model_input_instance

    def map_predicted_verbalizer_to_label(self, predicted_verbalizer: str) -> str:
        """
        Desc:
            gpt-3 is a text-competition model.
        Args:
            - predicted_verbalizer: (str,)
                e.g., "\n\nThe sentiment in the sentence is Negative."
        """
        # 1. strip&clean to obtain the label text.
        striped_predicted_verbalizer = predicted_verbalizer.strip()
        if "\n" in striped_predicted_verbalizer:
            candidate_lst = striped_predicted_verbalizer.split("\n")
            striped_predicted_verbalizer = candidate_lst[
                self.verbalizer_position_idx]  # should check if all labels are predicted in the last.
        if len(striped_predicted_verbalizer.split(" ")) > 5:
            # e.g.,
            # The sentence is neutral in tone and does not contain any words that would indicate a positive or negative sentiment. Therefore, the sentiment of the sentence is Neutral.
            striped_predicted_verbalizer = sent_tokenize(striped_predicted_verbalizer)[self.verbalizer_position_idx]
        if "," in striped_predicted_verbalizer:
            # e.g.,
            # Since the positive clues outweigh the negative clues, the sentiment of the sentence is Positive.
            striped_predicted_verbalizer = striped_predicted_verbalizer.split(",")[self.verbalizer_position_idx]
        # 2. lower-case (normalization).
        lowercase_striped_returned_text = striped_predicted_verbalizer.lower()
        lowercase_tokens = word_tokenize(lowercase_striped_returned_text)
        # 3. map verbalizer.
        # pred_label_in_text = set(lowercase_tokens) & set(self.inverse_verbalizer.keys())
        ###############
        pred_label_in_text = []
        for key in self.inverse_verbalizer.keys():
            if key in lowercase_striped_returned_text:
                idx = lowercase_striped_returned_text.index(key)
                if idx + len(key) >= len(lowercase_striped_returned_text):
                    pred_label_in_text.append(key)
                elif lowercase_striped_returned_text[idx + len(key)] not in "abcdefjhigklmnopqrstuvwxyz":
                    pred_label_in_text.append(key)
                    print(pred_label_in_text)
                else:
                    pass
        ##################
        if len(pred_label_in_text) == 0 or len(pred_label_in_text) > 1:
            print(predicted_verbalizer)
            raise LookupError
        print(predicted_verbalizer)
        print(pred_label_in_text)
        assert len(pred_label_in_text) == 1 or len(pred_label_in_text) == 0
        # 4. map label tokens to the label.
        pred_label = self.inverse_verbalizer[pred_label_in_text.pop()]
        return pred_label


class GPT3FewShotSamplingPrompt(Prompt):
    """
    Desc:
        Few-Shot prompt for gpt-3.
    Slots:
        - task_description: (str,)
        - demonstration_pattern: (str, )
            e.g. Input:<demonstration>\n\nSentiment:<verbalizer>
        - verbalizer: (dict, )
            A function that maps a label to the text (a.k.a. label words)
        - assemble_demonstration_strategy: (str),
            i.e. ["fill_pattern", ]
        - max_prompt_len: (int, 1024)
    """
    __slots__ = Prompt.__slots__ + ["task_description", "delimiter", "demonstration_pattern", "verbalizer",
                                    "feasible_verbalizer",
                                    "assemble_demonstration_strategy", "max_prompt_len", "inverse_verbalizer",
                                    "detokenizer", "verbalizer_position_idx", "demonstration_subtask_description",
                                    "assemble_demonstration_pattern", "data_retriever", "data_retriever_candidate_dir",
                                    "retriever_name_or_path",
                                    "retriever_ckpt_path", "file_saved_retriever_results", "demonstration_ranking",
                                    "non_verbalizer", "dataloader", "max_instance_len", "max_explain_len",
                                    "model_generate_max_len", "demonstration_subtask_description_pos", "prompt_suffix"]

    def __init__(self, key_value_params: Dict = None):
        super(GPT3FewShotSamplingPrompt, self).__init__(key_value_params)
        # check the config values.
        assert self.model_backbone == "gpt-3"
        assert self.prompt_strategy == "few-shot"
        assert self.instance_num > 0
        assert self.instance_strategy in ["random", "random-k-way-per-class", "finetuned-mlm-nearest-neighbor",
                                          "simcse-nearest-neighbor"]
        assert self.gradient_update is False
        assert "<VERBALIZER-LABEL>" in self.demonstration_pattern
        assert "<TEXT>" in self.demonstration_pattern
        assert self.assemble_demonstration_strategy in ["fill_pattern", "model_generate"]
        assert self.demonstration_ranking in ["random", "score_h2l", "score_l2h"]
        if self.max_instance_len is None:
            self.max_instance_len = 200
        if self.max_explain_len is None:
            self.max_explain_len = 100
        if self.demonstration_subtask_description_pos is None:
            self.demonstration_subtask_description_pos = 0
        if self.feasible_verbalizer is None:
            self.feasible_verbalizer = {label_id: label_token.lower() for label_id, label_token in
                                        self.verbalizer.items()}
        self.inverse_verbalizer = {}
        for label_symbol, label_word in self.feasible_verbalizer.items():
            # k denote labels, like 1, 2, 3.
            if isinstance(label_word, list):
                for token in label_word:
                    assert not any(element.isupper() for element in token)
                    self.inverse_verbalizer[token] = label_symbol
            elif isinstance(label_word, str):
                assert not any(element.isupper() for element in label_word)
                self.inverse_verbalizer[label_word] = label_symbol
            else:
                raise ValueError(self.inverse_verbalizer)
        self.detokenizer = Detokenizer()
        print(self.file_saved_retriever_results)
        if self.instance_strategy == "finetuned-mlm-nearest-neighbor":
            data_retriever_loader = self.dataloader
            self.data_retriever = FinetunedMLMRetriever(self.retriever_name_or_path, self.retriever_ckpt_path,
                                                        saved_nearest_neighbor_file=self.file_saved_retriever_results,
                                                        num_labels=len(data_retriever_loader.get_labels()))
            self.data_retriever.build_index(data_retriever_loader)
        elif self.instance_strategy == "simcse-nearest-neighbor":
            data_retriever_loader = self.dataloader
            self.data_retriever = SimCSERetriever(self.retriever_name_or_path, max_len=256,
                                                  saved_nearest_neighbor_file=self.file_saved_retriever_results, )
            self.data_retriever.build_index(data_retriever_loader)
        elif self.instance_strategy == "random":
            pass
        else:
            raise ValueError(self.instance_strategy)

    def select_demonstration_instances(self, demonstrations_candidates: List[DataItem] = None,
                                       test_instance: Union[List[str,], str] = None,
                                       shuffle: bool = True, ) -> List[DataItem]:

        num_of_candidates = len(demonstrations_candidates)
        if self.instance_strategy == "random":
            data_indices = [idx for idx in range(num_of_candidates)]
            if shuffle:
                random.shuffle(data_indices)
            sampled_demonstration_idx_lst = random.sample(data_indices, k=self.instance_num)
            sampled_demonstration_lst = [demonstrations_candidates[idx] for idx in sampled_demonstration_idx_lst]
        elif self.instance_strategy == "random-k-way-per-class":
            class_to_demonstration_idx = {}
            for demonstration_idx, demonstration_item in enumerate(demonstrations_candidates):
                if demonstration_item.label not in class_to_demonstration_idx.keys():
                    class_to_demonstration_idx[demonstration_item.label] = [demonstration_idx]
                else:
                    class_to_demonstration_idx[demonstration_item.label].append(demonstration_idx)

            sampled_demonstration_idx_lst = []
            for class_label, demo_idx_lst in class_to_demonstration_idx.items():
                assert self.instance_num <= len(demo_idx_lst)
                sampled_subset_demonstration_idx_lst = random.sample(demo_idx_lst, k=self.instance_num)
                sampled_demonstration_idx_lst.extend(sampled_subset_demonstration_idx_lst)
            sampled_demonstration_lst = [demonstrations_candidates[idx] for idx in sampled_demonstration_idx_lst]
            assert len(sampled_demonstration_lst) == self.instance_num * len(class_to_demonstration_idx.keys())
        elif self.instance_strategy == "finetuned-mlm-nearest-neighbor":
            sampled_demonstration_lst = self.data_retriever.search(test_instance, top_k=self.instance_num)
            sampled_demonstration_text_lst = [item[0] for item in sampled_demonstration_lst]
            sampled_demonstration_label_lst = [self.data_retriever.text_md5_to_label[encode_md5hash(text_item)] for
                                               text_item in sampled_demonstration_text_lst]
            sampled_demonstration_lst = []
            for text_item, label_item in zip(sampled_demonstration_text_lst, sampled_demonstration_label_lst):
                sampled_demonstration_lst.append(DataItem(text=text_item, label=label_item))
        elif self.instance_strategy == "simcse-nearest-neighbor":
            sampled_demonstration_lst = self.data_retriever.search(test_instance, top_k=self.instance_num)
            sampled_demonstration_text_lst = [item[0] for item in sampled_demonstration_lst]
            sampled_demonstration_label_lst = [self.data_retriever.text_md5_to_label[encode_md5hash(text_item)] for
                                               text_item in sampled_demonstration_text_lst]
            sampled_demonstration_lst = []
            for text_item, label_item in zip(sampled_demonstration_text_lst, sampled_demonstration_label_lst):
                sampled_demonstration_lst.append(DataItem(text=text_item, label=label_item))
        else:
            raise ValueError("Not Implementation.")

        if self.demonstration_ranking == "random":
            random.shuffle(sampled_demonstration_lst)
        elif self.demonstration_ranking == "score_h2l" and "nearest-neighbor" in self.instance_strategy:
            sampled_demonstration_lst = sampled_demonstration_lst
        elif self.demonstration_ranking == "score_l2h" and "nearest-neighbor" in self.instance_strategy:
            sampled_demonstration_lst.reverse()
        else:
            raise ValueError(self.demonstration_ranking)

        return sampled_demonstration_lst

    def assemble_demonstrations(self, sampled_demonstration_lst: List[DataItem] = None, teacher_model=None,
                                max_len: int = 2048, ) -> str:
        demonstration_info = ""
        if self.assemble_demonstration_strategy == "fill_pattern":
            for sampled_demonstration in sampled_demonstration_lst:
                demonstration_text = self._clip_text_by_space_len(sampled_demonstration.text, self.max_instance_len)
                sampled_info = self.demonstration_pattern.replace("<TEXT>", demonstration_text)
                sampled_info = sampled_info.replace("<VERBALIZER-LABEL>",
                                                    self.verbalizer[str(sampled_demonstration.label)])
                demonstration_info += sampled_info + self.delimiter
        elif self.assemble_demonstration_strategy == "model_generate":
            assert teacher_model is not None
            # 0. prepare text
            sampled_demonstration_text = [self._clip_text_by_space_len(item.text, self.max_instance_len) for item in
                                          sampled_demonstration_lst]
            # 1. generate demonstration prompt
            demonstration_prompt_subtext = [self.demonstration_pattern.replace("<TEXT>", item) for item in
                                            sampled_demonstration_text]
            demonstration_prompt_subtext = [
                item.replace("<VERBALIZER-LABEL>", self.verbalizer[str(sampled_demonstration_lst[idx].label)]) for
                idx, item
                in
                enumerate(demonstration_prompt_subtext)]
            if self.demonstration_subtask_description_pos == 0:
                demonstration_prompt = [
                    self.demonstration_subtask_description + f"{self.delimiter}" + item + f"\n" for item
                    in
                    demonstration_prompt_subtext]
            elif self.demonstration_subtask_description_pos == -1:
                demonstration_prompt = [
                    item + f"{self.delimiter}" + self.demonstration_subtask_description + f"\n" for item
                    in
                    demonstration_prompt_subtext]
            else:
                raise ValueError(self.demonstration_subtask_description_pos)
            # 2. feed demonstration prompt to the model.
            model_generated_info = teacher_model.forward(demonstration_prompt, num_workers=len(demonstration_prompt),
                                                         only_return_text=True,
                                                         update_max_tokens=self.model_generate_max_len)
            model_generated_info = [item.strip().replace("\n\n", "\n") for item in model_generated_info]
            model_generated_info = [self._clip_text_by_space_len(item, self.max_explain_len) for item in
                                    model_generated_info]
            assert len(model_generated_info) == len(sampled_demonstration_text)
            # 3. assemble demonstration
            for demon, text, model_gen in zip(sampled_demonstration_lst, sampled_demonstration_text,
                                              model_generated_info):
                current_demon_info = self.assemble_demonstration_pattern.replace("<TEXT>", text)
                current_demon_info = current_demon_info.replace("<VERBALIZER-LABEL>",
                                                                self.verbalizer[str(demon.label)])
                current_demon_info = current_demon_info.replace("<MODEL-GENERATE>", model_gen)
                demonstration_info += current_demon_info + self.delimiter
        else:
            raise ValueError
        if len(demonstration_info.split(" ")) > max_len:
            demonstration_info = " ".join(demonstration_info.split(" ")[:max_len])
        return demonstration_info

    def assemble_demonstrations_batch(self, sampled_demonstration_lst_batch: List[List[DataItem]] = None,
                                      teacher_model=None,
                                      max_len: int = 2048, ) -> str:

        demonstration_info_batch = []
        if self.assemble_demonstration_strategy == "fill_pattern":
            demonstration_info_batch = [
                self.assemble_demonstrations(sampled_demonstration_lst=sampled_demonstration_lst, max_len=max_len, )
                for sampled_demonstration_lst in sampled_demonstration_lst_batch]

        elif self.assemble_demonstration_strategy == "model_generate":
            assert teacher_model is not None

            demonstration_prompt_batch = []
            for sampled_demonstration_lst in sampled_demonstration_lst_batch:
                # 0. prepare text
                sampled_demonstration_text = [self._clip_text_by_space_len(item.text, self.max_instance_len) for item in
                                              sampled_demonstration_lst]
                # 1. generate demonstration prompt
                demonstration_prompt_subtext = [self.demonstration_pattern.replace("<TEXT>", item) for item in
                                                sampled_demonstration_text]
                demonstration_prompt_subtext = [
                    item.replace("<VERBALIZER-LABEL>", self.verbalizer[str(sampled_demonstration_lst[idx].label)]) for
                    idx, item
                    in
                    enumerate(demonstration_prompt_subtext)]
                if self.demonstration_subtask_description_pos == 0:
                    demonstration_prompt = [self.demonstration_subtask_description + f"{self.delimiter}" + item for item
                                            in
                                            demonstration_prompt_subtext]
                elif self.demonstration_subtask_description_pos == -1:
                    demonstration_prompt = [item + f"{self.delimiter}" + self.demonstration_subtask_description for item
                                            in
                                            demonstration_prompt_subtext]
                else:
                    raise ValueError(self.demonstration_subtask_description_pos)
                # 2. feed demonstration prompt to the model.
                demonstration_prompt_batch.append(demonstration_prompt)
            model_generated_info_batch = teacher_model.forward(demonstration_prompt_batch,
                                                               num_workers=len(demonstration_prompt_batch),
                                                               only_return_text=True,
                                                               update_max_tokens=self.model_generate_max_len)
            assert len(model_generated_info_batch) == len(demonstration_prompt_batch)

            for model_generated_info, sampled_demonstration_lst in zip(model_generated_info_batch,
                                                                       sampled_demonstration_lst_batch):
                demonstration_info = ""
                model_generated_info = [item.strip().replace("\n\n", "\n") for item in model_generated_info]
                model_generated_info = [self._clip_text_by_space_len(item, self.max_explain_len) for item in
                                        model_generated_info]
                assert len(model_generated_info) == len(sampled_demonstration_lst)
                # 3. assemble demonstration
                for demon, model_gen in zip(sampled_demonstration_lst, model_generated_info):
                    current_demon_info = self.assemble_demonstration_pattern.replace("<TEXT>",
                                                                                     self._clip_text_by_space_len(
                                                                                         demon.text,
                                                                                         self.max_instance_len))
                    current_demon_info = current_demon_info.replace("<VERBALIZER-LABEL>",
                                                                    self.verbalizer[str(demon.label)])
                    current_demon_info = current_demon_info.replace("<MODEL-GENERATE>", model_gen)
                    demonstration_info += current_demon_info + self.delimiter
                demonstration_info_batch.append(demonstration_info)
        else:
            raise ValueError

        return demonstration_info_batch

    def get_model_input(self, instance_text: str, demonstrations_candidates: List[DataItem] = None,
                        sampled_demonstration_lst: List[DataItem] = None, teacher_model=None,
                        need_detokenize: bool = True, max_len: int = None) -> str:
        """
        Desc:
            Assemble <Task Description> <Demonstrations> <Test-Instance> into <Prompt>.
        Args:
            - instance_text:
            - demonstrations_candidates: list of data-instance.
        """
        if sampled_demonstration_lst is None and demonstrations_candidates is None:
            raise ValueError("AT LEAST ONE OF <sampled_demonstration_lst> and <demonstrations_candidates> is not NONE.")

        if sampled_demonstration_lst is None:
            sampled_demonstration_lst = self.select_demonstration_instances(demonstrations_candidates,
                                                                            test_instance=instance_text)

        demonstration_info = self.assemble_demonstrations(sampled_demonstration_lst, teacher_model=teacher_model,
                                                          max_len=self.max_prompt_len - 350)

        if need_detokenize:
            instance_text = self.detokenizer.detokenize(instance_text)
        instance_text = self._clip_text_by_space_len(instance_text, self.max_instance_len)
        max_len = self.max_prompt_len if max_len is None else max_len
        if max_len <= len(demonstration_info.split(" ")):
            print(f"WARNING: PROMPT IS TOO LONG.")
            demonstration_info = self._clip_text_by_space_len(demonstration_info, max_len)
        model_input_instance = f"{self.task_description}{self.delimiter}{demonstration_info}INPUT: {instance_text}\n"
        if self.prompt_suffix is not None:
            model_input_instance = model_input_instance + self.prompt_suffix
        return model_input_instance

    def get_model_input_batch(self, instance_text_batch: List[str], demonstrations_candidates: List[DataItem] = None,
                              sampled_demonstration_lst: List[DataItem] = None, teacher_model=None,
                              need_detokenize: bool = True, max_len: int = None) -> List[str]:
        """get_model_input in batch-version"""
        if sampled_demonstration_lst is None and demonstrations_candidates is None:
            raise ValueError("AT LEAST ONE OF <sampled_demonstration_lst> and <demonstrations_candidates> is not NONE.")

        if sampled_demonstration_lst is None:
            sampled_demonstration_batch_lst = [self.select_demonstration_instances(demonstrations_candidates,
                                                                                   test_instance=instance_text) for
                                               instance_text in instance_text_batch]

        demonstration_info_batch = self.assemble_demonstrations_batch(sampled_demonstration_batch_lst,
                                                                      teacher_model=teacher_model,
                                                                      max_len=self.max_prompt_len - 450)

        if need_detokenize:
            instance_text_batch = [self.detokenizer.detokenize(instance_text) for instance_text in instance_text_batch]
        instance_text_batch = [self._clip_text_by_space_len(instance_text, self.max_instance_len) for instance_text in
                               instance_text_batch]

        model_input_instance_batch = [
            f"{self.task_description}{self.delimiter}{demonstration_info}INPUT: {instance}\n" for
            instance, demonstration_info
            in zip(instance_text_batch, demonstration_info_batch)]
        max_len = self.max_prompt_len if max_len is None else max_len
        if self.max_prompt_len <= max(
                [len(model_input_instance.split(" ")) for model_input_instance in model_input_instance_batch]):
            print(f"WARNING: PROMPT IS TOO LONG.")
            model_input_instance_batch = [self._clip_text_by_space_len(model_input, max_len) for model_input in
                                          model_input_instance_batch]
        if self.prompt_suffix is not None:
            model_input_instance_batch = [item + self.prompt_suffix for item in model_input_instance_batch]
        return model_input_instance_batch

    def map_predicted_verbalizer_to_label(self, predicted_verbalizer: str) -> str:
        """
        Desc:
            gpt-3 is a text-competition model.
        Args:
            - predicted_verbalizer: (str,)
                e.g., "\n\nThe sentiment in the sentence is Negative."
        """
        # 1. strip&clean to obtain the label text.
        striped_predicted_verbalizer = predicted_verbalizer.strip()
        if "\n" in striped_predicted_verbalizer:
            candidate_lst = striped_predicted_verbalizer.split("\n")
            striped_predicted_verbalizer = candidate_lst[
                self.verbalizer_position_idx]  # should check if all labels are predicted in the last.
        # 2. lower-case (normalization).
        lowercase_striped_returned_text = striped_predicted_verbalizer.lower()
        lowercase_tokens = word_tokenize(lowercase_striped_returned_text)
        # 3. map verbalizer.
        # pred_label_in_text = set(lowercase_tokens) & set(self.inverse_verbalizer.keys())
        pred_label_in_text = []
        for key in self.inverse_verbalizer.keys():
            if key in lowercase_striped_returned_text:
                idx = lowercase_striped_returned_text.index(key)
                if idx + len(key) >= len(lowercase_striped_returned_text):
                    pred_label_in_text.append(key)
                elif lowercase_striped_returned_text[idx + len(key)] not in "abcdefjhigklmnopqrstuvwxyz":
                    pred_label_in_text.append(key)
                    print(pred_label_in_text)
                else:
                    pass
        if len(pred_label_in_text) == 0 or len(pred_label_in_text) > 1:
            print(predicted_verbalizer)
            print("%" * 30)
            raise LookupError
        assert len(pred_label_in_text) == 1
        # 4. map label tokens to the label.
        pred_label = self.inverse_verbalizer[pred_label_in_text.pop()]
        return pred_label

    def _clip_text_by_space_len(self, input_text: str, max_space_len: int = 200) -> str:
        input_token = input_text.split(" ")
        if len(input_token) <= max_space_len:
            return input_text

        input_token_clipped = input_token[:max_space_len]
        input_text_clip = " ".join(input_token_clipped)
        return input_text_clip


class FLANT5Prompt(Prompt):
    """
    Desc:
        prompt for FLAN-T5.
    Slots:
        - task_description: (str,)
        - demonstration_pattern: (str, )
            e.g. Input:<demonstration>\n\nSentiment:<verbalizer>
        - verbalizer: (dict, )
            A function that maps a label to the text (a.k.a. label words)
        - assemble_demonstration_strategy: (str),
            i.e. ["fill_pattern", ]
        - max_prompt_len: (int, 1024)
    """
    __slots__ = ["inverse_verbalizer", "detokenizer", "non_verbalizer", "verbalizer",
                 "data_instance_prefix", "data_instance_suffix", ]

    def __init__(self, key_value_params: Dict = None):
        super(FLANT5Prompt, self).__init__(key_value_params)
        self.inverse_verbalizer = {}
        for label_symbol, label_word in self.verbalizer.items():
            # k denote labels, like 1, 2, 3.
            if isinstance(label_word, list):
                for token in label_word:
                    assert not any(element.isupper() for element in token)
                    self.inverse_verbalizer[token] = label_symbol
            elif isinstance(label_word, str):
                assert not any(element.isupper() for element in label_word)
                self.inverse_verbalizer[label_word] = label_symbol
            else:
                raise ValueError(self.inverse_verbalizer)

    def get_model_input(self, instance_text: str) -> str:
        instance_prompt = f"{self.data_instance_prefix} {instance_text} {self.data_instance_suffix}"
        return instance_prompt

    def map_predicted_verbalizer_to_label(self, predicted_verbalizer: str) -> str:
        """
        Desc:
            flan-t5 is a seq2seq model.
        Args:
            - predicted_verbalizer: (str,)
                e.g., "\n\nThe sentiment in the sentence is Negative."
        """
        # 1. strip&clean to obtain the label text.
        striped_predicted_verbalizer = predicted_verbalizer.strip()
        if "\n" in striped_predicted_verbalizer:
            candidate_lst = striped_predicted_verbalizer.split("\n")
            striped_predicted_verbalizer = candidate_lst[
                self.verbalizer_position_idx]  # should check if all labels are predicted in the last.
        # 2. lower-case (normalization).
        lowercase_striped_returned_text = striped_predicted_verbalizer.lower()
        lowercase_tokens = word_tokenize(lowercase_striped_returned_text)
        # 3. map verbalizer.
        # pred_label_in_text = set(lowercase_tokens) & set(self.inverse_verbalizer.keys())
        ###############
        pred_label_in_text = []
        for key in self.inverse_verbalizer.keys():
            if key in lowercase_striped_returned_text:
                pred_label_in_text.append(key)
        ##################
        if len(pred_label_in_text) == 0 or len(pred_label_in_text) > 1:
            print(predicted_verbalizer)
            raise LookupError
        assert len(pred_label_in_text) == 1
        # 4. map label tokens to the label.
        pred_label = self.inverse_verbalizer[pred_label_in_text.pop()]
        return pred_label


class ChatGPTFewShotSamplingPrompt(Prompt):
    """
    Desc:
        Few-Shot prompt for gpt-3.
    Slots:
        - task_description: (str,)
        - demonstration_pattern: (str, )
            e.g. Input:<demonstration>\n\nSentiment:<verbalizer>
        - verbalizer: (dict, )
            A function that maps a label to the text (a.k.a. label words)
        - assemble_demonstration_strategy: (str),
            i.e. ["fill_pattern", ]
        - max_prompt_len: (int, 1024)
    """
    __slots__ = Prompt.__slots__ + ["task_description", "delimiter", "demonstration_pattern", "verbalizer",
                                    "feasible_verbalizer",
                                    "assemble_demonstration_strategy", "max_prompt_len", "inverse_verbalizer",
                                    "detokenizer", "verbalizer_position_idx", "demonstration_subtask_description",
                                    "assemble_demonstration_pattern", "data_retriever", "data_retriever_candidate_dir",
                                    "retriever_name_or_path",
                                    "retriever_ckpt_path", "file_saved_retriever_results", "demonstration_ranking",
                                    "non_verbalizer", "dataloader", "max_instance_len", "max_explain_len",
                                    "model_generate_max_len", "demonstration_subtask_description_pos"]

    def __init__(self, key_value_params: Dict = None):
        super(ChatGPTFewShotSamplingPrompt, self).__init__(key_value_params)
        # check the config values.
        assert self.model_backbone == "gpt-3"
        assert self.prompt_strategy == "few-shot"
        assert self.instance_num > 0
        assert self.instance_strategy in ["random", "random-k-way-per-class", "finetuned-mlm-nearest-neighbor"]
        assert self.gradient_update is False
        assert "<VERBALIZER-LABEL>" in self.demonstration_pattern
        assert "<TEXT>" in self.demonstration_pattern
        assert self.assemble_demonstration_strategy in ["fill_pattern", "model_generate"]
        assert self.demonstration_ranking in ["random", "score_h2l", "score_l2h"]
        if self.max_instance_len is None:
            self.max_instance_len = 200
        if self.max_explain_len is None:
            self.max_explain_len = 100
        if self.demonstration_subtask_description_pos is None:
            self.demonstration_subtask_description_pos = 0
        if self.feasible_verbalizer is None:
            self.feasible_verbalizer = {label_id: label_token.lower() for label_id, label_token in
                                        self.verbalizer.items()}
        self.inverse_verbalizer = {}
        for label_symbol, label_word in self.feasible_verbalizer.items():
            # k denote labels, like 1, 2, 3.
            if isinstance(label_word, list):
                for token in label_word:
                    assert not any(element.isupper() for element in token)
                    self.inverse_verbalizer[token] = label_symbol
            elif isinstance(label_word, str):
                assert not any(element.isupper() for element in label_word)
                self.inverse_verbalizer[label_word] = label_symbol
            else:
                raise ValueError(self.inverse_verbalizer)
        self.detokenizer = Detokenizer()
        if self.dataloader is None:
            self.dataloader = SST2Dataloader(self.data_retriever_candidate_dir)
        print(self.file_saved_retriever_results)
        if self.instance_strategy == "finetuned-mlm-nearest-neighbor":
            data_retriever_loader = self.dataloader
            self.data_retriever = FinetunedMLMRetriever(self.retriever_name_or_path, self.retriever_ckpt_path,
                                                        saved_nearest_neighbor_file=self.file_saved_retriever_results,
                                                        num_labels=len(data_retriever_loader.get_labels()))
            self.data_retriever.build_index(data_retriever_loader)
        elif self.instance_strategy == "simcse-nearest-neighbor":
            data_retriever_loader = self.dataloader
            self.data_retriever = SimCSERetriever(self.retriever_name_or_path, max_len=256,
                                                  saved_nearest_neighbor_file=self.file_saved_retriever_results, )
            self.data_retriever.build_index(data_retriever_loader)
        else:
            raise ValueError(self.instance_strategy)

    def select_demonstration_instances(self, demonstrations_candidates: List[DataItem] = None,
                                       test_instance: Union[List[str,], str] = None,
                                       shuffle: bool = True, ) -> List[DataItem]:

        num_of_candidates = len(demonstrations_candidates)
        if self.instance_strategy == "random":
            data_indices = [idx for idx in range(num_of_candidates)]
            if shuffle:
                random.shuffle(data_indices)
            sampled_demonstration_idx_lst = random.sample(data_indices, k=self.instance_num)
            sampled_demonstration_lst = [demonstrations_candidates[idx] for idx in sampled_demonstration_idx_lst]
        elif self.instance_strategy == "random-k-way-per-class":
            class_to_demonstration_idx = {}
            for demonstration_idx, demonstration_item in enumerate(demonstrations_candidates):
                if demonstration_item.label not in class_to_demonstration_idx.keys():
                    class_to_demonstration_idx[demonstration_item.label] = [demonstration_idx]
                else:
                    class_to_demonstration_idx[demonstration_item.label].append(demonstration_idx)

            sampled_demonstration_idx_lst = []
            for class_label, demo_idx_lst in class_to_demonstration_idx.items():
                assert self.instance_num <= len(demo_idx_lst)
                sampled_subset_demonstration_idx_lst = random.sample(demo_idx_lst, k=self.instance_num)
                sampled_demonstration_idx_lst.extend(sampled_subset_demonstration_idx_lst)
            sampled_demonstration_lst = [demonstrations_candidates[idx] for idx in sampled_demonstration_idx_lst]
            assert len(sampled_demonstration_lst) == self.instance_num * len(class_to_demonstration_idx.keys())
        elif self.instance_strategy == "finetuned-mlm-nearest-neighbor":
            sampled_demonstration_lst = self.data_retriever.search(test_instance, top_k=self.instance_num)
            sampled_demonstration_text_lst = [item[0] for item in sampled_demonstration_lst]
            sampled_demonstration_label_lst = [self.data_retriever.text_md5_to_label[encode_md5hash(text_item)] for
                                               text_item in sampled_demonstration_text_lst]
            sampled_demonstration_lst = []
            for text_item, label_item in zip(sampled_demonstration_text_lst, sampled_demonstration_label_lst):
                sampled_demonstration_lst.append(DataItem(text=text_item, label=label_item))
        else:
            raise ValueError("Not Implementation.")

        if self.demonstration_ranking == "random":
            random.shuffle(sampled_demonstration_lst)
        elif self.demonstration_ranking == "score_h2l" and self.instance_strategy == "finetuned-mlm-nearest-neighbor":
            sampled_demonstration_lst = sampled_demonstration_lst
        elif self.demonstration_ranking == "score_l2h" and self.instance_strategy == "finetuned-mlm-nearest-neighbor":
            sampled_demonstration_lst.reverse()
        else:
            raise ValueError(self.demonstration_ranking)

        return sampled_demonstration_lst

    def assemble_demonstrations(self, sampled_demonstration_lst: List[DataItem] = None, teacher_model=None,
                                max_len: int = 2048, ) -> str:
        demonstration_info = ""
        if self.assemble_demonstration_strategy == "fill_pattern":
            for sampled_demonstration in sampled_demonstration_lst:
                demonstration_text = self._clip_text_by_space_len(sampled_demonstration.text, self.max_instance_len)
                sampled_info = self.demonstration_pattern.replace("<TEXT>", demonstration_text)
                sampled_info = sampled_info.replace("<VERBALIZER-LABEL>",
                                                    self.verbalizer[str(sampled_demonstration.label)])
                demonstration_info += sampled_info + self.delimiter
        elif self.assemble_demonstration_strategy == "model_generate":
            assert teacher_model is not None
            # 0. prepare text
            sampled_demonstration_text = [self._clip_text_by_space_len(item.text, self.max_instance_len) for item in
                                          sampled_demonstration_lst]
            # 1. generate demonstration prompt
            demonstration_prompt_subtext = [self.demonstration_pattern.replace("<TEXT>", item) for item in
                                            sampled_demonstration_text]
            demonstration_prompt_subtext = [
                item.replace("<VERBALIZER-LABEL>", self.verbalizer[str(sampled_demonstration_lst[idx].label)]) for
                idx, item
                in
                enumerate(demonstration_prompt_subtext)]
            if self.demonstration_subtask_description_pos == 0:
                demonstration_prompt = [
                    self.demonstration_subtask_description + f"{self.delimiter}" + item + f"\n" for item
                    in
                    demonstration_prompt_subtext]
            elif self.demonstration_subtask_description_pos == -1:
                demonstration_prompt = [
                    item + f"{self.delimiter}" + self.demonstration_subtask_description + f"\n" for item
                    in
                    demonstration_prompt_subtext]
            else:
                raise ValueError(self.demonstration_subtask_description_pos)
            # 2. feed demonstration prompt to the model.
            model_generated_info = teacher_model.forward(demonstration_prompt, num_workers=len(demonstration_prompt),
                                                         only_return_text=True,
                                                         update_max_tokens=self.model_generate_max_len)
            model_generated_info = [item.strip().replace("\n\n", "\n") for item in model_generated_info]
            model_generated_info = [self._clip_text_by_space_len(item, self.max_explain_len) for item in
                                    model_generated_info]
            assert len(model_generated_info) == len(sampled_demonstration_text)
            # 3. assemble demonstration
            for demon, text, model_gen in zip(sampled_demonstration_lst, sampled_demonstration_text,
                                              model_generated_info):
                current_demon_info = self.assemble_demonstration_pattern.replace("<TEXT>", text)
                current_demon_info = current_demon_info.replace("<VERBALIZER-LABEL>",
                                                                self.verbalizer[str(demon.label)])
                current_demon_info = current_demon_info.replace("<MODEL-GENERATE>", model_gen)
                demonstration_info += current_demon_info + self.delimiter
        else:
            raise ValueError
        if len(demonstration_info.split(" ")) > max_len:
            demonstration_info = " ".join(demonstration_info.split(" ")[:max_len])
        return demonstration_info

    def assemble_demonstrations_batch(self, sampled_demonstration_lst_batch: List[List[DataItem]] = None,
                                      teacher_model=None,
                                      max_len: int = 2048, ) -> str:

        demonstration_info_batch = []
        if self.assemble_demonstration_strategy == "fill_pattern":
            demonstration_info_batch = [
                self.assemble_demonstrations(sampled_demonstration_lst=sampled_demonstration_lst, max_len=max_len, )
                for sampled_demonstration_lst in sampled_demonstration_lst_batch]

        elif self.assemble_demonstration_strategy == "model_generate":
            assert teacher_model is not None

            demonstration_prompt_batch = []
            for sampled_demonstration_lst in sampled_demonstration_lst_batch:
                # 0. prepare text
                sampled_demonstration_text = [self._clip_text_by_space_len(item.text, self.max_instance_len) for item in
                                              sampled_demonstration_lst]
                # 1. generate demonstration prompt
                demonstration_prompt_subtext = [self.demonstration_pattern.replace("<TEXT>", item) for item in
                                                sampled_demonstration_text]
                demonstration_prompt_subtext = [
                    item.replace("<VERBALIZER-LABEL>", self.verbalizer[str(sampled_demonstration_lst[idx].label)]) for
                    idx, item
                    in
                    enumerate(demonstration_prompt_subtext)]
                if self.demonstration_subtask_description_pos == 0:
                    demonstration_prompt = [self.demonstration_subtask_description + f"{self.delimiter}" + item for item
                                            in
                                            demonstration_prompt_subtext]
                elif self.demonstration_subtask_description_pos == -1:
                    demonstration_prompt = [item + f"{self.delimiter}" + self.demonstration_subtask_description for item
                                            in
                                            demonstration_prompt_subtext]
                else:
                    raise ValueError(self.demonstration_subtask_description_pos)
                # 2. feed demonstration prompt to the model.
                demonstration_prompt_batch.append(demonstration_prompt)
            model_generated_info_batch = teacher_model.forward(demonstration_prompt_batch,
                                                               num_workers=len(demonstration_prompt_batch),
                                                               only_return_text=True,
                                                               update_max_tokens=self.model_generate_max_len)
            assert len(model_generated_info_batch) == len(demonstration_prompt_batch)

            for model_generated_info, sampled_demonstration_lst in zip(model_generated_info_batch,
                                                                       sampled_demonstration_lst_batch):
                demonstration_info = ""
                model_generated_info = [item.strip().replace("\n\n", "\n") for item in model_generated_info]
                model_generated_info = [self._clip_text_by_space_len(item, self.max_explain_len) for item in
                                        model_generated_info]
                assert len(model_generated_info) == len(sampled_demonstration_lst)
                # 3. assemble demonstration
                for demon, model_gen in zip(sampled_demonstration_lst, model_generated_info):
                    current_demon_info = self.assemble_demonstration_pattern.replace("<TEXT>",
                                                                                     self._clip_text_by_space_len(
                                                                                         demon.text,
                                                                                         self.max_instance_len))
                    current_demon_info = current_demon_info.replace("<VERBALIZER-LABEL>",
                                                                    self.verbalizer[str(demon.label)])
                    current_demon_info = current_demon_info.replace("<MODEL-GENERATE>", model_gen)
                    demonstration_info += current_demon_info + self.delimiter
                demonstration_info_batch.append(demonstration_info)
        else:
            raise ValueError

        return demonstration_info_batch

    def get_model_input(self, instance_text: str, demonstrations_candidates: List[DataItem] = None,
                        sampled_demonstration_lst: List[DataItem] = None, teacher_model=None,
                        need_detokenize: bool = True, max_len: int = None) -> str:
        """
        Desc:
            Assemble <Task Description> <Demonstrations> <Test-Instance> into <Prompt>.
        Args:
            - instance_text:
            - demonstrations_candidates: list of data-instance.
        """
        if sampled_demonstration_lst is None and demonstrations_candidates is None:
            raise ValueError("AT LEAST ONE OF <sampled_demonstration_lst> and <demonstrations_candidates> is not NONE.")

        if sampled_demonstration_lst is None:
            sampled_demonstration_lst = self.select_demonstration_instances(demonstrations_candidates,
                                                                            test_instance=instance_text)

        demonstration_info = self.assemble_demonstrations(sampled_demonstration_lst, teacher_model=teacher_model,
                                                          max_len=self.max_prompt_len - 350)

        if need_detokenize:
            instance_text = self.detokenizer.detokenize(instance_text)
        instance_text = self._clip_text_by_space_len(instance_text, self.max_instance_len)
        max_len = self.max_prompt_len if max_len is None else max_len
        if max_len <= len(demonstration_info.split(" ")):
            print(f"WARNING: PROMPT IS TOO LONG.")
            demonstration_info = self._clip_text_by_space_len(demonstration_info, max_len)
        model_input_instance = f"{self.task_description}{self.delimiter}{demonstration_info}INPUT: {instance_text}\n"
        update_model_input_instance = [
            {
                "role": "system",
                "content": "This is an overall sentiment classifier. ",
            },
            {
                "role": "user",
                "content": model_input_instance
            },
        ]
        return update_model_input_instance

    def get_model_input_batch(self, instance_text_batch: List[str], demonstrations_candidates: List[DataItem] = None,
                              sampled_demonstration_lst: List[DataItem] = None, teacher_model=None,
                              need_detokenize: bool = True, max_len: int = None) -> List[str]:
        """get_model_input in batch-version"""
        if sampled_demonstration_lst is None and demonstrations_candidates is None:
            raise ValueError("AT LEAST ONE OF <sampled_demonstration_lst> and <demonstrations_candidates> is not NONE.")

        if sampled_demonstration_lst is None:
            sampled_demonstration_batch_lst = [self.select_demonstration_instances(demonstrations_candidates,
                                                                                   test_instance=instance_text) for
                                               instance_text in instance_text_batch]

        demonstration_info_batch = self.assemble_demonstrations_batch(sampled_demonstration_batch_lst,
                                                                      teacher_model=teacher_model,
                                                                      max_len=self.max_prompt_len - 450)

        if need_detokenize:
            instance_text_batch = [self.detokenizer.detokenize(instance_text) for instance_text in instance_text_batch]
        instance_text_batch = [self._clip_text_by_space_len(instance_text, self.max_instance_len) for instance_text in
                               instance_text_batch]

        model_input_instance_batch = [
            f"{self.task_description}{self.delimiter}{demonstration_info}INPUT: {instance}\n" for
            instance, demonstration_info
            in zip(instance_text_batch, demonstration_info_batch)]
        max_len = self.max_prompt_len if max_len is None else max_len
        if self.max_prompt_len <= max(
                [len(model_input_instance.split(" ")) for model_input_instance in model_input_instance_batch]):
            print(f"WARNING: PROMPT IS TOO LONG.")
            model_input_instance_batch = [self._clip_text_by_space_len(model_input, max_len) for model_input in
                                          model_input_instance_batch]
        update_model_input_instance = [[
            {
                "role": "system",
                "content": "This is an overall sentiment classifier. ",
            },
            {
                "role": "user",
                "content": item
            },
        ] for item in model_input_instance_batch]
        return update_model_input_instance

    def map_predicted_verbalizer_to_label(self, predicted_verbalizer: str) -> str:
        """
        Desc:
            gpt-3 is a text-competition model.
        Args:
            - predicted_verbalizer: (str,)
                e.g., "\n\nThe sentiment in the sentence is Negative."
        """
        # 1. strip&clean to obtain the label text.
        striped_predicted_verbalizer = predicted_verbalizer.strip()
        if "\n" in striped_predicted_verbalizer:
            candidate_lst = striped_predicted_verbalizer.split("\n")
            striped_predicted_verbalizer = candidate_lst[
                self.verbalizer_position_idx]  # should check if all labels are predicted in the last.
        # 2. lower-case (normalization).
        lowercase_striped_returned_text = striped_predicted_verbalizer.lower()
        lowercase_tokens = word_tokenize(lowercase_striped_returned_text)
        print(lowercase_striped_returned_text)
        print("&" * 20)
        print(self.inverse_verbalizer)
        # 3. map verbalizer.
        # pred_label_in_text = set(lowercase_tokens) & set(self.inverse_verbalizer.keys())
        pred_label_in_text = []
        for key in self.inverse_verbalizer.keys():
            if key in lowercase_striped_returned_text:
                pred_label_in_text.append(key)

        print(pred_label_in_text)
        print("LOOKUP | LOOKUP")
        if len(pred_label_in_text) == 0 or len(pred_label_in_text) > 1:
            print(predicted_verbalizer)
            print("%" * 30)
            raise LookupError
        assert len(pred_label_in_text) == 1
        # 4. map label tokens to the label.
        pred_label = self.inverse_verbalizer[pred_label_in_text[0]]
        return pred_label

    def _clip_text_by_space_len(self, input_text: str, max_space_len: int = 200) -> str:
        input_token = input_text.split(" ")
        if len(input_token) <= max_space_len:
            return input_text

        input_token_clipped = input_token[:max_space_len]
        input_text_clip = " ".join(input_token_clipped)
        return input_text_clip


class FLANT5Prompt(Prompt):
    """
    Desc:
        prompt for FLAN-T5.
    Slots:
        - task_description: (str,)
        - demonstration_pattern: (str, )
            e.g. Input:<demonstration>\n\nSentiment:<verbalizer>
        - verbalizer: (dict, )
            A function that maps a label to the text (a.k.a. label words)
        - assemble_demonstration_strategy: (str),
            i.e. ["fill_pattern", ]
        - max_prompt_len: (int, 1024)
    """
    __slots__ = ["inverse_verbalizer", "detokenizer", "non_verbalizer", "verbalizer",
                 "data_instance_prefix", "data_instance_suffix", ]

    def __init__(self, key_value_params: Dict = None):
        super(FLANT5Prompt, self).__init__(key_value_params)
        self.inverse_verbalizer = {}
        for label_symbol, label_word in self.verbalizer.items():
            # k denote labels, like 1, 2, 3.
            if isinstance(label_word, list):
                for token in label_word:
                    assert not any(element.isupper() for element in token)
                    self.inverse_verbalizer[token] = label_symbol
            elif isinstance(label_word, str):
                assert not any(element.isupper() for element in label_word)
                self.inverse_verbalizer[label_word] = label_symbol
            else:
                raise ValueError(self.inverse_verbalizer)

    def get_model_input(self, instance_text: str) -> str:
        instance_prompt = f"{self.data_instance_prefix} {instance_text} {self.data_instance_suffix}"
        return instance_prompt

    def map_predicted_verbalizer_to_label(self, predicted_verbalizer: str) -> str:
        """
        Desc:
            flan-t5 is a seq2seq model.
        Args:
            - predicted_verbalizer: (str,)
                e.g., "\n\nThe sentiment in the sentence is Negative."
        """
        # 1. strip&clean to obtain the label text.
        striped_predicted_verbalizer = predicted_verbalizer.strip()
        if "\n" in striped_predicted_verbalizer:
            candidate_lst = striped_predicted_verbalizer.split("\n")
            striped_predicted_verbalizer = candidate_lst[
                self.verbalizer_position_idx]  # should check if all labels are predicted in the last.
        # 2. lower-case (normalization).
        lowercase_striped_returned_text = striped_predicted_verbalizer.lower()
        lowercase_tokens = word_tokenize(lowercase_striped_returned_text)
        # 3. map verbalizer.
        # pred_label_in_text = set(lowercase_tokens) & set(self.inverse_verbalizer.keys())
        ###############
        pred_label_in_text = []
        for key in self.inverse_verbalizer.keys():
            if key in lowercase_striped_returned_text:
                pred_label_in_text.append(key)
        ##################
        if len(pred_label_in_text) == 0 or len(pred_label_in_text) > 1:
            print(predicted_verbalizer)
            raise LookupError
        assert len(pred_label_in_text) == 1
        # 4. map label tokens to the label.
        pred_label = self.inverse_verbalizer[pred_label_in_text.pop()]
        return pred_label
