#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: model/gpt_model.py
@time: 2022/12/06 20:03
@desc:
https://beta.openai.com/docs/models/content-filter
"""

import logging
import random
import time
from multiprocessing import Pool
from typing import List, Union, Dict

import numpy as np
import openai
from openai.embeddings_utils import get_embedding

from data.config import GPT3ModelConfig
from utils.get_logger import get_info_logger


class GPT3ModelAPI(object):
    def __init__(self, config: GPT3ModelConfig, openai_key_offset_idx: int = 0, logger: logging = None):
        self.config = config
        self.logger = logger if logger is not None else get_info_logger("GPT-3")
        self.openai_key_idx = openai_key_offset_idx
        self.openai_key_candidates = self.config.openai_api_key if type(self.config.openai_api_key) is list else [
            self.config.openai_api_key]
        assert self.config.engine_name in ["text-ada-002", "text-davinci-002", "text-davinci-003",
                                           "text-embedding-ada-002"]

    def get_openai_response(self, post_prompt: str, key_idx_offset: int = None):
        raise NotImplementedError("get_openai_response")

    def request_api_and_handle_errors(self, post_prompt: str, exponential_base: float = 2,
                                      jitter: bool = True, key_idx_offset: int = None,
                                      only_return_text: bool = False) -> Dict:
        """Keep the prediction func's name the same as previous PyTorch models.
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb 的 Example 3: Manual backoff implementation
        """
        # Initialize variables
        num_retries = 0
        delay = self.config.init_delay
        return_sign = 0
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                response_lst = self.get_openai_response(post_prompt, key_idx_offset=key_idx_offset)
                if self.config.engine_name.startswith("text-embedding"):
                    results = response_lst
                else:
                    results = response_lst.choices
                if only_return_text:
                    results = [item.text for item in results]
                self.logger.info(msg=f"prompt_and_result",
                                 extra={"prompt_list": post_prompt, "results": response_lst})
                time.sleep(delay)
                return_sign = 1
            except openai.error.RateLimitError as limiterror:
                limiterror_info = str(limiterror)
                if limiterror_info == "You exceeded your current quota, please check your plan and billing details.":
                    if self.openai_key_idx < len(self.openai_key_candidates) - 1:
                        self.logger.info("=" * 40)
                        self.logger.info(f"WARNING: {self.openai_key_idx} OUT-Of-Quota ...")
                        self.openai_key_idx += 1
                        self.logger.info(f"WARNING: RESET OPENAI-KEY TO {self.openai_key_idx} ...")
                    elif self.openai_key_idx == len(self.openai_key_candidates) - 1:
                        self.logger.info(f"ReTRY Failed ...")
                        raise Exception(f"WARNING: Out-Of-Quota ({len(self.openai_key_candidates)} accounts).")
                    else:
                        raise ValueError
                # Increment retries
                num_retries += 1
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                self.logger.info("=" * 40)
                self.logger.info(f"WARNING: SLEEP {round(delay, 4)}s for RateLimitError")
                self.logger.info(f"WARNING: {num_retries} RETRY out of {self.config.max_retries}")
                self.logger.info("=" * 40)
                # Check if max retries has been reached
                if num_retries > self.config.max_retries:
                    self.logger.info(f"ReTRY Failed ..")
                    raise Exception(f"WARNING: Maximum number of retries ({self.config.max_retries}) exceeded.")

                # Sleep for the delay
                time.sleep(delay)
            except (openai.error.ServiceUnavailableError, openai.error.APIConnectionError, openai.error.APIError):
                num_retries += 1
                self.logger.info("=" * 40)
                self.logger.info(f"WARNING: SLEEP 70s for ServiceUnavailableError or APIConnectionError")
                self.logger.info(f"WARNING: {num_retries} RETRY out of {self.config.max_retries}")
                self.logger.info("=" * 40)
                time.sleep(70)
                if num_retries > self.config.max_retries:
                    self.logger.info(f"RETRY FAILED")
                    raise Exception(f"WARNING: Maximum number of retries ({self.config.max_retries}) exceeded.")
            # Raise exceptions for any errors not specified
            except openai.error.InvalidRequestError:
                self.logger.info("=" * 40)
                self.logger.info("WARNING: Current Account DO NOT have Quota.")
                self.logger.info("=" * 40)
            except Exception as e:
                self.logger.info(f"OTHER ERRORs")
                self.logger.info(e)
                raise e
            if return_sign == 1:
                return results

    def forward(self, prompt_lst: Union[List[str], str], exponential_base: float = 2,
                jitter: bool = True, num_workers: int = 1, key_idx_offset: int = None,
                only_return_text: bool = False, update_max_tokens: int = None) -> List[Dict]:
        # if num_workers is 1.
        if num_workers == 1 or (len(prompt_lst) == 1 and isinstance(prompt_lst, list)):
            if isinstance(prompt_lst, list):
                prompt_input = prompt_lst[0]
            elif isinstance(prompt_lst, str):
                prompt_input = prompt_lst
            else:
                raise TypeError(prompt_lst)
            results = self.request_api_and_handle_errors(prompt_input, exponential_base=exponential_base, jitter=jitter,
                                                         key_idx_offset=key_idx_offset,
                                                         only_return_text=only_return_text)
            return results
        # if num_workers > 1
        assert isinstance(prompt_lst, list)
        assert len(prompt_lst) <= num_workers
        num_workers = min(num_workers, len(prompt_lst))
        pool_results = []
        pool = Pool(processes=num_workers)
        for worker_id in range(0, num_workers):
            result = pool.apply_async(self.request_api_and_handle_errors,
                                      (prompt_lst[worker_id], exponential_base, jitter, worker_id, only_return_text), )
            pool_results.append(result)

        pool.close()
        results = [item.get()[0] for item in pool_results]
        return results


class GPT3TextCompletionModel(GPT3ModelAPI):
    """Code for OpenAI's GPT-3 model service."""

    def __init__(self, config: GPT3ModelConfig, openai_key_offset_idx: int = 0, logger: logging = None):
        super(GPT3TextCompletionModel, self).__init__(config, openai_key_offset_idx, logger)

    def get_openai_response(self, post_prompt: str, key_idx_offset: int = None):
        # https://beta.openai.com/docs/api-reference/completions/create
        openai_key_idx = self.openai_key_idx if key_idx_offset is None else self.openai_key_idx + key_idx_offset
        openai.api_key = self.openai_key_candidates[openai_key_idx]
        response_lst = openai.Completion.create(
            engine=self.config.engine_name,
            prompt=post_prompt,
            temperature=self.config.temperature,
            max_tokens=min(self.config.max_tokens, 3900 - len(post_prompt.split(" "))),
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            logprobs=self.config.logprobs,
        )
        return response_lst


class GPT3EmbeddingModel(GPT3ModelAPI):
    """Code for OpenAI's GPT-3 model service."""

    def __init__(self, config: GPT3ModelConfig, openai_key_offset_idx: int = 0, logger: logging = None):
        super(GPT3EmbeddingModel, self).__init__(config, openai_key_offset_idx, logger)

    def get_openai_response(self, input_text: str, key_idx_offset: int = None) -> np.array:
        openai_key_idx = self.openai_key_idx if key_idx_offset is None else self.openai_key_idx + key_idx_offset
        openai.api_key = self.openai_key_candidates[openai_key_idx]
        text_embedding = get_embedding(input_text, self.config.engine_name)
        text_embedding = np.array(text_embedding)
        return text_embedding
