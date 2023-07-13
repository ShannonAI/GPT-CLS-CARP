# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2023/2/2
@desc: 这只飞很懒
"""

import json
import logging
import time
from math import ceil
from typing import List, Dict

import requests
from more_itertools import chunked


class ClientWithFriday(object):
    def __init__(self, *, api_key_list, friday_url, completion_inference_parameters, chat_inference_parameters,
                 key_size_per_grou, switch_per_n_seconds, use_proxy, token):
        assert "/api/" not in friday_url
        assert api_key_list
        self.token = token
        self.completion_inference_parameters = completion_inference_parameters
        self.chat_inference_parameters = chat_inference_parameters
        self.friday_url = friday_url
        self.switch_per_n_seconds = switch_per_n_seconds
        self.api_groups = list(chunked(api_key_list, key_size_per_grou))
        self.begin = time.time()
        self.use_proxy = use_proxy

    def choose_api_group(self) -> List[str]:
        """
        根据当前时间戳，选择当前应当使用的api-group，方法是：
        1. 先看一下当前是第几个时间窗口：window_index = (time.time() - self.begin) // self.switch_per_n_seconds
        2. 然后取模：index = window_index % len(self.api_groups)
        """
        window_index = int((time.time() - self.begin)) // self.switch_per_n_seconds
        index = window_index % len(self.api_groups)
        logging.info(f"使用group-{index}")
        return self.api_groups[index]

    def get_multiple_sample_more_than_ten(self, prompt_list):
        # print(prompt_list)
        for_return = []
        bucket_size = 250
        n_slices = ceil(float(len(prompt_list)) / bucket_size)
        for i in range(n_slices):
            print(f"staring slice {i}")
            current_prompts = prompt_list[i * bucket_size:(i + 1) * bucket_size]
            for_return += self.get_multiple_sample(current_prompts)
            print(f"finish slice {i}")
        return for_return

    def get_multiple_sample_with_limited_length(self, prompt_list, max_length):
        for_return = []
        for begin in range(0, len(prompt_list), max_length):
            logging.info(f"staring slice {begin} to {begin + max_length}")
            current_prompts = prompt_list[begin:begin + max_length]
            for_return += self.get_multiple_sample(current_prompts)
            print(f"finish slice {begin} to {begin + max_length}")
        return for_return

    def get_multiple_sample(
            self,
            prompt_list: List[str],
    ):
        """
        直接调用friday获得结果
        """
        # "api_key_list": self.choose_api_group(),
        response = requests.post(
            f"{self.friday_url}/api/generate",
            json={
                "token": self.token,
                "prompt_list": prompt_list,
                "api_key_list": self.choose_api_group(),
                "inference_parameters": self.completion_inference_parameters,
                "max_sleep_time": 100,
                "max_retries": 100,
                "version": "v1.1",
                "use_proxy": self.use_proxy
            }
        )
        print("===" * 10)
        print(response)
        results = response.json()["results_list"]
        logging.info(
            msg=f"prompt_and_result",
            extra={"parameters": self.completion_inference_parameters, "prompt_list": prompt_list, "results": results}
        )
        return results

    def get_chat_result(
            self,
            message_list_list: List[List[Dict]],
    ):
        """
        直接调用friday获得结果
        """
        # "api_key_list": self.choose_api_group(),
        response = requests.post(
            f"{self.friday_url}/api/chat",
            json={
                "token": self.token,
                "message_list_list": message_list_list,
                "api_key_list": self.choose_api_group(),
                "inference_parameters": self.chat_inference_parameters,
                "max_sleep_time": 100,
                "max_retries": 100,
                "version": "v1.1",
                "use_proxy": self.use_proxy
            },
            timeout=600
        )
        results = response.json()["results_list"]
        result_string_list = []
        for r in results:
            if "error" in r:
                result_string_list.append(r["error"])
            else:
                result_string_list.append(r["message"]["content"])
        return result_string_list


def run_test():
    e = ClientWithFriday(
        token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MTE0MTU3OTksInN1YiI6ImthMjAzbnNkaWF3bmRpbnEwMWgiLCJ1aWQiOiJsaXhpYW95YSJ9.gcbTVr9YzdLie3fWBd-6InzLr_Nf0TlkP-ck87vW6c4",
        api_key_list=[
            "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"
        ],
        completion_inference_parameters={
            "request_timeout": 100,
            "engine": "text-davinci-003",
            "temperature": 0.0,
            "max_tokens": 600,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "best_of": 1
        },
        chat_inference_parameters={
            "request_timeout": 100,
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "max_tokens": 600,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        friday_url="http://47.251.43.109:31022",
        key_size_per_grou=1,
        switch_per_n_seconds=60,
        use_proxy=True
    )

    prompt_list = [
        "please write a headline for the ice-cream shop.\n",
    ]
    result = e.get_multiple_sample(prompt_list)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    message_list_list = [
        [
            {
                "role": "system",
                "content": "你是一名学者",
            },
            {
                "role": "user",
                "content": "写一个论文的标题"
            },
        ]
    ]

    # result = e.get_chat_result(message_list_list)
    # print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run_test()
