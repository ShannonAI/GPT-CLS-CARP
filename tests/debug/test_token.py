#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/model/test_conntectiob.py
@time: 2022/12/06 20:03
@desc:
"""

import requests


def test_incompatible():
    """refer to xiaofei's friday."""
    inference_parameters = {
        "request_timeout": 10000,
        "engine": "text-davinci-003",  # 这个是选择engine，也就是到底用什么模型，text-davinci-003是最贵最好的
        "temperature": 0.7,  # 控制随机程度
        "max_tokens": 20,  # 最大token数量
        "top_p": 1,  # 控制多样性，目前我们没有多样性，这个和temperature的相互配合还没有测试
        "frequency_penalty": 1,  # 对于频次的惩罚，这个值调高可以降低重复词
        "presence_penalty": 1,  # 这个是对于词是否已经出现过的惩罚，文档上说这个值调高可以增大谈论新topic的概率，和frequency_penalty的关系还没有测试
        "best_of": 1,  # 这个是说从多少个里选最好的，如果这里是10，就会生成10个然后选最好的，但是这样会更贵,

    }

    response = requests.post(
        "http://47.251.43.109:31023/api/generate_list_list",
        json={
            "prompt_list": ["I like cats"],
            "inference_parameters": inference_parameters,
            "api_key_list": ["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", ],
            "max_sleep_time": 100,
            "max_retries": 100,
            "use_proxy": True,
            "version": "v1.1",
            "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MTE0MTU3OTksInN1YiI6ImthMjAzbnNkaWF3bmRpbnEwMWgiLCJ1aWQiOiJsaXhpYW95YSJ9.gcbTVr9YzdLie3fWBd-6InzLr_Nf0TlkP-ck87vW6c4"

        }
    )

    print(response.json())
    result = response.json()["results_list"]
    print(result)


def test_lucky():
    pass


if __name__ == "__main__":
    test_incompatible()
