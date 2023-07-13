#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: task/gpt3_text_cls.py
@time: 2022/12/06 20:03
@desc:
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter
from typing import List

from more_itertools import chunked
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data.config import GPT3TextCLSTaskConfig
from data.file_utils import save_jsonl, load_jsonl, save_json, get_num_lines, check_file_and_mkdir_for_save
from model.gpt_model import GPT3TextCompletionModel
from utils.get_logger import get_info_logger
from utils.random_seed import set_basic_random_seed


class GPT3TextCLS(object):
    """TEXT Classification with GPT-3 models."""

    def __init__(self, task_config_path: str, seed: int = 2333, run: int = 1, log_interval: int = 50,
                 save_config: bool = False, fix_uid: bool = False):
        self.config = GPT3TextCLSTaskConfig.from_json_file(task_config_path)
        # upgrade the seed before save config to file.
        # step 0: prepare configs.
        if seed is not None:
            self.config.update_attribute_value("save_log_dir", self.config.save_log_dir + f"_seed{seed}")

        if run is not None:
            self.config.update_attribute_value("save_log_dir", self.config.save_log_dir + f"_run{run}")

        self.logger = get_info_logger("GPT3-Text-Classification",
                                      save_log_file=os.path.join(self.config.save_log_dir, "exp_log.txt"),
                                      print_to_console=True)
        self.prompt = self.config.prompt_config
        # step 1: init model
        if self.config.gpt3_backbone == "vanilla":
            self.model = GPT3TextCompletionModel(self.config.gpt3_model_config, logger=self.logger)
        else:
            raise ValueError(self.config.gpt3_backbone)
        if self.config.prompt_type.startswith("few-shot"):
            self.teacher_model = self.model if self.prompt.assemble_demonstration_strategy == "model_generate" else None
        else:
            self.teacher_model = None

        self.dataloader = self.config.dataloader
        self.logger.info(f">>> PLEASE CHECK THE CONFIG ... ...")
        self.logger.info(f">>> LOG DIR : {self.config.save_log_dir}")
        self.log_interval = log_interval
        # set user
        if fix_uid:
            user_id = str(seed) if seed is not None else str(run)
            self.config.gpt3_model_config.update_attribute_value("user", user_id)
        if save_config:
            save_config_path = os.path.join(self.config.save_log_dir, "task_config.json")
            self.config.save_to_json(save_config_path)

    def step1_prepare_input(self, input_file_name: str = "test_dist_182subset", demonstration_file_name: str = "train",
                            resume: bool = False):
        save_data_path = os.path.join(self.config.save_log_dir, "step1_data.jsonl")
        resume_offset = 0 if not resume else get_num_lines(save_data_path)
        if resume:
            self.logger.info("~~" * 20)
            self.logger.info(f"INFO: STEP-1 RESUME from offset={resume_offset}")
            self.logger.info("~~" * 20)
        if os.path.exists(save_data_path) and not resume:
            raise FileExistsError(f"step1_prepare_input -> {save_data_path}")
        check_file_and_mkdir_for_save(save_data_path, resume=resume, file_suffix=".jsonl")

        self.logger.info("$" * 40)
        self.logger.info("INFO: START STEP-1")
        self.logger.info("$" * 40)
        sleep_counter = 0
        if self.config.prompt_type == "few-shot-fix":
            # type 2.1: few-shot-random-fix
            demonstration_candidates = self.dataloader.load_data_files(demonstration_file_name)

            selected_demonstration_lst = self.prompt.select_demonstration_instances(demonstration_candidates,
                                                                                    shuffle=True)
        if self.config.prompt_type == "few-shot-dynamic":
            demonstration_candidates = self.dataloader.load_data_files(demonstration_file_name)

        test_items = self.dataloader.load_data_files(input_file_name, offset=resume_offset)

        # start save intermediate results.
        writer_mode = "w" if not resume else "a"
        writer_f = open(save_data_path, writer_mode, encoding='utf-8')
        batch_size = self.config.gpt3_model_config.batch_size if self.config.gpt3_backbone != "vanilla" else 1
        for idx, data_item in tqdm(enumerate(chunked(test_items, batch_size)),
                                   total=math.ceil(len(test_items) / batch_size), desc="step-1"):
            if batch_size == 1:
                data_item = data_item[0]
            if idx % self.log_interval == 0:
                self.logger.info(f"$$$ STEP 1 - Currrent progress -> {idx} out-of {len(test_items)}")
            if self.config.prompt_type == "few-shot-fix":
                input_text_with_prompt = self.prompt.get_model_input(data_item.text,
                                                                     sampled_demonstration_lst=selected_demonstration_lst,
                                                                     teacher_model=self.teacher_model)
            elif self.config.prompt_type == "few-shot-dynamic":
                if batch_size == 1:
                    input_text_with_prompt = self.prompt.get_model_input(data_item.text,
                                                                         demonstrations_candidates=demonstration_candidates,
                                                                         teacher_model=self.teacher_model)
                elif batch_size > 1:
                    data_item_text = [item.text for item in data_item]
                    input_text_with_prompt = self.prompt.get_model_input_batch(data_item_text,
                                                                               demonstrations_candidates=demonstration_candidates,
                                                                               teacher_model=self.teacher_model)
                else:
                    raise ValueError(batch_size)
            elif self.config.prompt_type == "zero-shot":
                input_text_with_prompt = []
                for item in data_item:
                    text_with_prompt = self.prompt.get_model_input(item.text, )
                    input_text_with_prompt.append(text_with_prompt)
            else:
                raise ValueError(self.config.prompt_type)

            if self.config.prompt_type.startswith("few-shot"):
                if self.config.prompt_config.assemble_demonstration_strategy == "model_generate":
                    sleep_counter += 1
                    if sleep_counter >= self.config.gpt3_model_config.rate_limit and batch_size == 1:
                        self.logger.info(f"INFO: SLEEP ... ...")
                        time.sleep(self.config.gpt3_model_config.rate_limit_delay)
                        sleep_counter = 0

            if batch_size == 1:
                input_label = data_item.label
                data_item_obj = {"prompt_text": input_text_with_prompt, "gold_label": input_label,
                                 "text": data_item.text}
                writer_f.write(f"{json.dumps(data_item_obj)}\n")
            else:
                for item, prompt in zip(data_item, input_text_with_prompt):
                    data_item_obj = {"prompt_text": prompt, "gold_label": item.label,
                                     "text": item.text}
                    writer_f.write(f"{json.dumps(data_item_obj)}\n")
            if idx <= 10:
                self.logger.info("^=" * 20)
                self.logger.info("INFO: Example of STEP-1 : Prompts")
                self.logger.info(data_item_obj)

        # end saving results.
        writer_f.close()
        self.logger.info(f"STEP 1: save data with prompt to {save_data_path}")
        return save_data_path

    def step2_get_gpt3_results(self, step1_prompt_data_path: str, resume: bool = False, save_logprobs: bool = True):
        """
        Args:
            - prompt_data_path:
        """
        saved_result_path = os.path.join(self.config.save_log_dir, "step2_result.jsonl")
        resume_offset = 0 if not resume else get_num_lines(saved_result_path)
        if resume:
            self.logger.info("~~" * 20)
            self.logger.info(f"INFO: STEP-2 RESUME from offset={resume_offset}")
            self.logger.info("~~" * 20)
        if os.path.exists(saved_result_path) and not resume:
            raise FileExistsError(f"step2_get_gpt3_results -> {saved_result_path}")
        check_file_and_mkdir_for_save(saved_result_path, resume=resume, file_suffix=".jsonl")

        self.logger.info("$" * 40)
        self.logger.info("INFO: START STEP-2")
        self.logger.info("$" * 40)

        data_item_lst = load_jsonl(step1_prompt_data_path, offset=resume_offset)
        sleep_counter = 0

        # start save intermediate results.
        writer_mode = "w" if not resume else "a"
        writer_f = open(saved_result_path, writer_mode, encoding='utf-8')
        num_workers = self.config.prompt_config.instance_num if self.config.prompt_type.startswith("few-shot") and self.config.gpt3_backbone != "vanilla" else 1

        for idx, batch_data_item in tqdm(enumerate(chunked(data_item_lst, num_workers)),
                                         total=math.ceil(len(data_item_lst) / num_workers), desc="step-2"):
            if idx % self.log_interval == 0:
                self.logger.info(f"$$$ STEP 2 - Currrent progress -> {idx} out-of {len(data_item_lst)}")
            batch_prompt_text = [item["prompt_text"] for item in batch_data_item]
            if idx <= 3:
                self.logger.info(">>> PROMPT WITH TEXT")
                self.logger.info(batch_prompt_text[0])
            gpt_returned_results = self.model.forward(batch_prompt_text, num_workers=num_workers,
                                                      only_return_text=False)

            if self.config.gpt3_backbone == "vanilla":
                gpt_returned_text = [result.text for result in gpt_returned_results]
                gpt_returned_logprobs = [result.logprobs for result in gpt_returned_results]
            elif self.config.gpt3_backbone == "chat":
                gpt_returned_text = gpt_returned_results
                gpt_returned_logprobs = len(gpt_returned_results) * [None]
            else:
                raise ValueError(self.config.gpt3_backbone)
            assert len(batch_data_item) == len(gpt_returned_results)
            # assemble results
            for data_item, returned_text, returned_logprobs in zip(batch_data_item, gpt_returned_text,
                                                                   gpt_returned_logprobs):
                save_result_item = {"gpt_returned_result": returned_text, "prompt_text": data_item["prompt_text"],
                                    "gold_label": data_item["gold_label"], "text": data_item["text"],
                                    "gpt_returned_logprobs": returned_logprobs}
                writer_f.write(f"{json.dumps(save_result_item)}\n")

            sleep_counter += 1
            if sleep_counter >= self.config.gpt3_model_config.rate_limit:
                self.logger.info(f"INFO: SLEEP ... ...")
                time.sleep(self.config.gpt3_model_config.rate_limit_delay)
                sleep_counter = 0

        writer_f.close()
        self.logger.info(f"STEP 2: save data with prompt to {saved_result_path}")
        return saved_result_path

    def step3_map_competition_result_to_label(self, step2_gpt_result_path: str):
        """
        step_3
        """
        assert isinstance(self.prompt.non_verbalizer, list)
        assert len(self.prompt.non_verbalizer) >= 1
        saved_result_path_lst = []
        for non_verbalizer_strategy in self.prompt.non_verbalizer:
            feasible = True
            saved_result_path = os.path.join(self.config.save_log_dir, f"step3_result_{non_verbalizer_strategy}.jsonl")
            save_out_of_scope_result_path = os.path.join(self.config.save_log_dir,
                                                         f"step3_out_of_scope_prediction_{non_verbalizer_strategy}.jsonl")
            if os.path.exists(saved_result_path) or os.path.exists(save_out_of_scope_result_path):
                raise FileExistsError(f"step3_map_competition_result_to_label -> {saved_result_path}")
            saved_result_path_lst.append(saved_result_path)

            self.logger.info("$" * 40)
            self.logger.info("INFO: START STEP-3")
            self.logger.info("$" * 40)
            data_item_lst = load_jsonl(step2_gpt_result_path)
            out_of_scope_pred_lst = []
            update_data_item_lst = []
            ood_counter = 0
            for data_item in data_item_lst:
                try:
                    pred_label = self.prompt.map_predicted_verbalizer_to_label(data_item["gpt_returned_result"])
                    pred_label_word_in_verbalizer = True
                except LookupError:
                    pred_label_word_in_verbalizer = False
                    candidate_label_lst = [label for label in self.prompt.verbalizer.keys()]
                    if non_verbalizer_strategy == "wrong":
                        sample_lst = list(set(candidate_label_lst) - set([data_item["gold_label"]]))
                        pred_label = random.choice(sample_lst)
                    elif non_verbalizer_strategy == "random":
                        pred_label = random.choice(candidate_label_lst)
                    elif non_verbalizer_strategy == "retry" or non_verbalizer_strategy == "retry-various":
                        max_retry = 20
                        prompt_text = data_item["prompt_text"]
                        retry_pred_label = None
                        for idx in range(max_retry):
                            if non_verbalizer_strategy == "retry_various":
                                self.config.gpt3_model_config.update_attribute_value("user", str(idx))
                            self.logger.info("$" * 20)
                            self.logger.info(f"INFO: retry {idx}")
                            retry_returned_results = self.model.forward(prompt_text, only_return_text=True)[0]
                            self.logger.info(f"INFO: returned results - {retry_returned_results}")
                            try:
                                retry_pred_label = self.prompt.map_predicted_verbalizer_to_label(retry_returned_results)
                            except:
                                retry_pred_label = None
                            if retry_pred_label is not None:
                                break
                            if retry_pred_label is None and idx == max_retry - 1:
                                ood_counter += 1
                                retry_pred_label = random.choice(candidate_label_lst)
                        self.logger.info("$" * 20)
                        retry_time = idx + 1
                        pred_label = retry_pred_label
                        if retry_time >= max_retry and pred_label is None:
                            feasible = False
                        data_item.update(
                            {"retry_time": retry_time, "retry_gpt_returned_results": retry_returned_results})
                    elif non_verbalizer_strategy == "hard-vote":
                        vote_counter = 20
                        prompt_text = data_item["prompt_text"]
                        pred_labels = []
                        for idx in range(vote_counter):
                            self.config.gpt3_model_config.update_attribute_value("user", str(idx))
                            self.logger.info("$" * 20)
                            self.logger.info(f"INFO: retry {idx}")
                            retry_returned_results = self.model.forward(prompt_text, only_return_text=True)[0]
                            self.logger.info(f"INFO: returned results - {retry_returned_results}")
                            try:
                                current_pred_label = self.prompt.map_predicted_verbalizer_to_label(
                                    retry_returned_results)
                            except:
                                current_pred_label = None
                            if current_pred_label is not None:
                                pred_labels.append(current_pred_label)
                        self.logger.info("$" * 20)
                        pred_label = Counter(pred_labels).most_common(1)[0][0]
                        data_item.update(
                            {"vote_gpt_labels": pred_labels})
                    elif non_verbalizer_strategy == "verbalizer-rank":
                        # only work for classify-explain.
                        pred_tokens = data_item["gpt_returned_logprobs"]["tokens"]
                        verbalizer_token_index = pred_tokens.index("\n") - 1
                        pred_tokens_top = data_item["gpt_returned_logprobs"]["top_logprobs"][verbalizer_token_index]

                        filter_pred_token_top = {k: v for k, v in pred_tokens_top.items() if
                                                 k.strip().lower() in ["negative", "neg", "pos", "positive"]}
                        if len(filter_pred_token_top) > 0:
                            inverse_pred_tokens_top = {v: k for k, v in filter_pred_token_top.items()}
                            pred_tokens_logprobs = [item for item in filter_pred_token_top.values()]
                            pred_tokens_logprobs.sort()
                            top_token_logprobs = pred_tokens_logprobs[-1]
                            top_token = inverse_pred_tokens_top[top_token_logprobs].strip()
                        else:
                            top_token = random.choice([item for item in self.prompt.verbalizer.values()])

                        try:
                            pred_label = self.prompt.map_predicted_verbalizer_to_label(top_token)
                        except LookupError:
                            top_token = random.choice([item for item in self.prompt.verbalizer.values()])
                            pred_label = self.prompt.map_predicted_verbalizer_to_label(top_token)
                    else:
                        raise ValueError(self.prompt.non_verbalizer)
                    out_of_scope_pred_lst.append(data_item)
                data_item.update(
                    {"pred_label": pred_label, "pred_label_word_in_verbalizer": pred_label_word_in_verbalizer,
                     })
                update_data_item_lst.append(data_item)

            # save transformed results to file.
            if feasible:
                save_jsonl(saved_result_path, update_data_item_lst)
                self.logger.info(f"STEP 3: map competition result to label and save to {saved_result_path}")

                # save out-of-scope results
                save_jsonl(save_out_of_scope_result_path, out_of_scope_pred_lst)
                self.logger.info(
                    f"STEP 3: save out-of-scope label to {saved_result_path}; ood_counter is {ood_counter}")
        return saved_result_path_lst

    def step4_evaluate_performance(self, saved_step3_result_path_lst: List[str] = None):
        if saved_step3_result_path_lst is None:
            saved_step3_result_path_lst = [
                os.path.join(self.config.save_log_dir, f"step3_result_{non_verbalizer_strategy}.jsonl")
                for non_verbalizer_strategy in self.prompt.non_verbalizer]
        assert isinstance(saved_step3_result_path_lst, list)
        assert len(saved_step3_result_path_lst) >= 1
        for saved_step3_result_path in saved_step3_result_path_lst:
            eval_performance = {}
            non_verbalizer = saved_step3_result_path.replace(".jsonl", "").split("/")[-1].split("_")[-1]
            assert non_verbalizer in ["wrong", "random", "retry", "vote", "retry-various", "verbalizer-rank",
                                      "hard-vote"]
            save_result_path = os.path.join(self.config.save_log_dir, f"step4_eval_result_{non_verbalizer}.json")
            if os.path.exists(save_result_path):
                raise FileExistsError(f"step4_evaluate_performance -> {save_result_path}")

            self.logger.info("$" * 40)
            self.logger.info("INFO: START STEP-4")
            self.logger.info("$" * 40)
            data_item_lst = load_jsonl(saved_step3_result_path)
            pred_label_lst = [str(item["pred_label"]) for item in data_item_lst]
            gold_label_lst = [str(item["gold_label"]) for item in data_item_lst]
            eval_acc_score = accuracy_score(gold_label_lst, pred_label_lst)
            eval_performance[non_verbalizer] = {"acc_score": eval_acc_score, "total_num": len(pred_label_lst)}
            self.logger.info(f"STEP 4: evaluated ACC score is {eval_acc_score}")
            save_json(save_result_path, eval_performance)
            self.logger.info(f"STEP 4: save ACC score to path {save_result_path}")

    def step6_predict_analysis(self, saved_step3_result_path: str = None, ensemble_pred: bool = False,
                               pred_result_dir_lst: List[str] = None, error_analysis: bool = True):

        if ensemble_pred:
            pred_map = {}
            gold_map = {}
            pred_path_lst = [os.path.join(pred_result, "step3_result.jsonl") for pred_result in pred_result_dir_lst]

            for idx, pred_path in enumerate(pred_path_lst):
                data_items = load_jsonl(pred_path)
                for item in data_items:
                    text = item["prompt_text"].split("INPUT:")[-1]
                    text = text.replace("\n", "")
                    if idx == 0:
                        gold_map[text] = item["gold_label"]
                    if text not in pred_map.keys():
                        pred_map[text] = [item["pred_label"]]
                    else:
                        pred_map[text].append(item["pred_label"])

            ensemble_results = {}
            num_correct = 0
            total_num = len(gold_map)
            for item_key, item_pred in pred_map.items():
                assert len(item_pred) == len(pred_result_dir_lst)
                ensemble_pred = Counter(item_pred)
                ensemble_pred = ensemble_pred.most_common(1)[0][0]
                # most_common -> [('1', 5)]
                ensemble_results[item_key] = ensemble_pred
                if ensemble_pred == gold_map[item_key]:
                    num_correct += 1
            print(f"acc is {num_correct / float(total_num)}")
            print(len(pred_map))

        if error_analysis:
            save_wrong_pred_path = os.path.join(self.config.save_log_dir, "step5_wrong_pred.jsonl")
            if os.path.exists(save_wrong_pred_path):
                raise FileExistsError(f"step4_evaluate_performance -> {save_wrong_pred_path}")
            data_item_lst = load_jsonl(saved_step3_result_path)
            num_of_test_data = len(data_item_lst)
            pred_correct_data_lst, pred_wrong_data_lst = [], []
            for idx, data_item in enumerate(data_item_lst):
                if str(data_item["pred_label"]) == str(data_item["gold_label"]):
                    pred_correct_data_lst.append(data_item)
                else:
                    pred_wrong_data_lst.append(data_item)
            num_correct = len(pred_correct_data_lst)
            num_wrong = len(pred_wrong_data_lst)
            self.logger.info(f"INFO: Total {num_of_test_data}, Correctly {num_correct}, Wrongly {num_wrong}")
            save_jsonl(save_wrong_pred_path, pred_wrong_data_lst)
            self.logger.info(f"INFO: save WRONG data to {save_wrong_pred_path}")


def run():
    parser = argparse.ArgumentParser(description="GPT3-For-TEXT-Classification")
    parser.add_argument("--seed", default=2333, type=int, help="random seed")
    parser.add_argument("--random", action="store_true", default=False)
    parser.add_argument("--config_path", default="../config/", type=str, help="path to the config file.")
    parser.add_argument("--test_file_name", default="test", type=str)
    parser.add_argument("--step_idx", default="1-2-3-4-5", type=str)
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--fix_uid", default=False, action="store_true")
    # TODO (xiaoya): add assert to check.
    args = parser.parse_args()
    if not args.random:
        set_basic_random_seed(args.seed)

    step1_save_path, step2_save_path, step3_save_path, = None, None, None
    step_idx_lst = [int(idx) for idx in args.step_idx.split("-")]
    gpt3_for_text_cls_task = GPT3TextCLS(args.config_path, seed=args.seed if not args.random else None,
                                         run=args.seed if args.random else None,
                                         save_config=True if 1 in step_idx_lst else False,
                                         fix_uid=args.fix_uid)

    if 1 in step_idx_lst:
        resume = True if args.resume and step_idx_lst[0] == 1 else False
        step1_save_path = gpt3_for_text_cls_task.step1_prepare_input(input_file_name=args.test_file_name,
                                                                     resume=resume)
    if 2 in step_idx_lst:
        if step1_save_path is None:
            step1_save_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, "step1_data.jsonl")
        step2_save_path = gpt3_for_text_cls_task.step2_get_gpt3_results(step1_save_path,
                                                                        resume=True if args.resume and step_idx_lst[
                                                                            0] == 2 else False)
    if 3 in step_idx_lst:
        if step2_save_path is None:
            step2_save_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, "step2_result.jsonl")
        step3_save_path = gpt3_for_text_cls_task.step3_map_competition_result_to_label(step2_save_path)
    if 4 in step_idx_lst:
        gpt3_for_text_cls_task.step4_evaluate_performance(step3_save_path)
    if 6 in step_idx_lst:
        if step3_save_path is None:
            step3_save_path = os.path.join(gpt3_for_text_cls_task.config.save_log_dir, "step3_result_wrong.jsonl")
        gpt3_for_text_cls_task.step6_predict_analysis(saved_step3_result_path=step3_save_path)


if __name__ == "__main__":
    run()
