#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/model/gpt_model.py
@time: 2022/12/06 20:03
@desc:
"""

from data.config import GPT3ModelConfig
from model.gpt_model import GPT3TextCompletionModel, GPT3EmbeddingModel


def test_gpt_model():
    gpt_config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/gpt_config.json"
    gpt_config = GPT3ModelConfig.from_json_file(gpt_config_path)
    print("=*" * 10)
    print(f"check gpt config {gpt_config}")
    gpt_model = GPT3TextCompletionModel(gpt_config)

    batch_prompt_text = ["Write a tagline for an ice cream shop.",
                         "Q: Who is Batman?\n A: Batman is a fictional comic book character.\n\n\n Q: What is torsalplexity?\n A: "]
    print(gpt_model.get_openai_response(batch_prompt_text))
    # ['\n\n"The best in the city."', "\n\nTorsalplexity is a measure of how much of the forward motion of a boat's motion is due to the torsion of the body. It is a measure of a boat's potential forward motion."]

    print("==" * 20)
    prompt_text = "Write a tagline for an ice cream shop."
    print(gpt_model.get_openai_response(prompt_text))
    # ['\n\n"The best in the city."']


def test_single_processing_gpt_model():
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/gpt_config.json"
    gpt_config = GPT3ModelConfig.from_json_file(config_path)
    gpt_model = GPT3TextCompletionModel(gpt_config)
    prompt_text = ["Write a tagline for an ice cream shop."]
    results = gpt_model.forward(prompt_text[0], num_workers=1)
    print(results)


def test_multiple_processing_gpt_model():
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/gpt_config.json"
    gpt_config = GPT3ModelConfig.from_json_file(config_path)
    gpt_model = GPT3TextCompletionModel(gpt_config)
    prompt_text = ["Write a tagline for an ice cream shop.", "Write a tagline for an ice cream shop."]
    results = gpt_model.forward(prompt_text, num_workers=2)
    print(results)


def test_embedding_model():
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/emb_gpt_config.json"
    gpt_config = GPT3ModelConfig.from_json_file(config_path)
    gpt_model = GPT3EmbeddingModel(gpt_config)
    prompt_text = "Write a tagline for an ice cream shop."
    results = gpt_model.forward(prompt_text, num_workers=1)
    print(results)
    print(type(results))
    prompt_text_lst = ["Write a tagline for an ice cream shop.", "Write a tagline for an ice cream shop."]
    results = gpt_model.forward(prompt_text_lst, num_workers=len(prompt_text_lst))
    print(results)


if __name__ == "__main__":
    # test_single_processing_gpt_model()
    # test_multiple_processing_gpt_model()
    test_embedding_model()
