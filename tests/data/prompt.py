#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/prompt.py
@time: 2022/12/06 20:03
@desc:
"""

from data.config import FridayModelConfig
from data.dataloader import SST2Dataloader
from data.prompt import GPT3ZeroShotPrompt, MaskedLMPrompt, MaskedLMZeroShotPrompt, GPT3FewShotSamplingPrompt, \
    FLANT5Prompt, ChatGPTFewShotSamplingPrompt
from model.friday_model import FridayClient


def test_chat_prompt():
    file = "/data2/lixiaoya/workspace/gpt-text/tests/file/chat_friday_prompt_config.json"
    prompt = ChatGPTFewShotSamplingPrompt.from_json_file(file)
    sent = "If you sometimes like to go to the movies to have fun , Wasabi is a good place to start ."
    demonstration_candidates = prompt.dataloader.load_data_files("train")
    result = prompt.get_model_input(sent, demonstrations_candidates=demonstration_candidates)
    print(result)
    results = prompt.get_model_input_batch([sent, sent], demonstrations_candidates=demonstration_candidates)
    print(results)


def test_gpt3_zeroshot_init_prompt():
    prompt = GPT3ZeroShotPrompt()
    print("init prompt")
    print(prompt)


def test_gpt3_zeroshot_save_prompt():
    prompt = GPT3ZeroShotPrompt()
    save_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/gpt3_zeroshot_prompt.json"
    prompt.save_to_json(save_path)


def test_gpt3_zeroshot_load_prompt():
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/prompt_zeroshot.json"
    prompt = GPT3ZeroShotPrompt.from_json_file(config_path)
    print(prompt)


def test_gpt3_zeroshot_map_predicted_verbalizer_to_label():
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/prompt_zeroshot.json"
    prompt = GPT3ZeroShotPrompt.from_json_file(config_path)
    pred_verbalizer = "\n\nThe sentiment in the sentence is Negative."
    pred_label = prompt.map_predicted_verbalizer_to_label(pred_verbalizer)
    print(f"verbalizer: {pred_verbalizer}")
    print(f"label: {pred_label}")


def test_roberta_basic_prompt():
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/roberta_prompt.json"
    prompt = MaskedLMPrompt.from_json_file(config_path)
    print(prompt)


def test_roberta_zeroshot_prompt():
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/roberta_prompt.json"
    roberta_zero_shot_prompt = MaskedLMZeroShotPrompt.from_json_file(config_path)
    print(roberta_zero_shot_prompt)
    input_batch = ["I like apples", ] * 10
    model_input = roberta_zero_shot_prompt.get_model_input(input_batch)
    print(model_input)

    tokenized_output = [[23156, 50, 8593, 50118, 100, 101, 20150], [23156, 50, 8593, 50118, 100, 101, 20150],
                        [23156, 50, 8593, 50118, 100, 101, 20150], [23156, 50, 8593, 50118, 100, 101, 20150],
                        [23156, 50, 8593, 50118, 100, 101, 20150], [23156, 50, 8593, 50118, 100, 101, 20150],
                        [23156, 50, 8593, 50118, 100, 101, 20150], [23156, 50, 8593, 50118, 100, 101, 20150],
                        [23156, 50, 8593, 50118, 100, 101, 20150], [23156, 50, 8593, 50118, 100, 101, 20150]]
    decoded_results = roberta_zero_shot_prompt.decode(tokenized_output)
    print(decoded_results)
    # ['neg Ġor Ġpos Ċ I Ġlike Ġapples', 'neg Ġor Ġpos Ċ I Ġlike Ġapples', 'neg Ġor Ġpos Ċ I Ġlike Ġapples', 'neg Ġor Ġpos Ċ I Ġlike Ġapples', 'neg Ġor Ġpos Ċ I Ġlike Ġapples', 'neg Ġor Ġpos Ċ I Ġlike Ġapples', 'neg Ġor Ġpos Ċ I Ġlike Ġapples', 'neg Ġor Ġpos Ċ I Ġlike Ġapples', 'neg Ġor Ġpos Ċ I Ġlike Ġapples', 'neg Ġor Ġpos Ċ I Ġlike Ġapples']


def test_roberta_basic_tokenize():
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/roberta_prompt.json"
    basic_prompt = MaskedLMPrompt.from_json_file(config_path)
    input_batch = [
                      "I like Apples. I like Apples. I like Apples. I like Apples. I like Apples. I like Apples. I like Apples. I like Apples. ", ] * 10
    tokenized_output = basic_prompt.tokenizer.tokenize_input_batch(input_batch)
    print(tokenized_output)

    # print(type(tokenized_output))
    # print([item for item in tokenized_output.keys()])
    # <class 'transformers.tokenization_utils_base.BatchEncoding'>
    # ['input_ids', 'attention_mask', 'offset_mapping']

    idx_batch = [[100, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38],
                 [100, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38],
                 [100, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38],
                 [100, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38],
                 [100, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38],
                 [100, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38],
                 [100, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38],
                 [100, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38],
                 [100, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38],
                 [100, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38, 101, 3166, 1634, 4, 38]]
    decode_tokens = basic_prompt.tokenizer.decode(idx_batch)
    # ['I Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI', 'I Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI', 'I Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI', 'I Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI', 'I Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI', 'I Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI', 'I Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI', 'I Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI', 'I Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI', 'I Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI Ġlike ĠApp les . ĠI']
    print(decode_tokens)


def random_sample_gpt3_prompt():
    data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    sst_dataloader = SST2Dataloader(data_dir)

    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/gpt3_fewshot_random.json"
    few_shot_prompt = GPT3FewShotSamplingPrompt.from_json_file(config_path)
    print("=" * 10)
    print("STEP 0: check the prompt configs.")
    print(few_shot_prompt)

    # test sample demos
    demo_candidates = sst_dataloader.load_data_files("train")
    selected_data_lst = few_shot_prompt.select_demonstration_instances(demo_candidates)
    print("=" * 10)
    print("STEP 1: select demos.")
    print(selected_data_lst)

    # test get model input
    instace = "I like it."
    input_example = few_shot_prompt.get_model_input(instace, sampled_demonstration_lst=selected_data_lst)
    print("=" * 10)
    print("STEP 2: Assemble Prompt.")
    print(input_example)

    instace = "I like it."
    input_example_rand = few_shot_prompt.get_model_input(instace, demonstrations_candidates=demo_candidates)
    print("=" * 10)
    print("STEP 3: Assemble Prompt With Random.")
    print(input_example_rand)


def random_kway_sample_gpt3_prompt():
    data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    sst_dataloader = SST2Dataloader(data_dir)

    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/gpt3_fewshot_random_kway.json"
    few_shot_prompt = GPT3FewShotSamplingPrompt.from_json_file(config_path)
    print("=" * 10)
    print("STEP 0: check the prompt configs.")
    print(few_shot_prompt)

    # test sample demos
    demo_candidates = sst_dataloader.load_data_files("train")
    selected_data_lst = few_shot_prompt.select_demonstration_instances(demo_candidates)
    print("=" * 10)
    print("STEP 1: select demos.")
    print(selected_data_lst)

    # test get model input
    instace = "I like it."
    input_example = few_shot_prompt.get_model_input(instace, sampled_demonstration_lst=selected_data_lst)
    print("=" * 10)
    print("STEP 2: Assemble Prompt.")
    print(input_example)

    instace = "I like it."
    input_example_rand = few_shot_prompt.get_model_input(instace, demonstrations_candidates=demo_candidates)
    print("=" * 10)
    print("STEP 3: Assemble Prompt With Random.")
    print(input_example_rand)


def gpt3_finetuned_mlm_neighbor_few_shot_prompt():
    data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    sst_dataloader = SST2Dataloader(data_dir)

    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/gpt3_mlm_prompt.json"
    few_shot_prompt = GPT3FewShotSamplingPrompt.from_json_file(config_path)
    print("=" * 10)
    print("STEP 0: check the prompt configs.")
    print(few_shot_prompt)

    exit()

    # test sample demos
    demo_candidates = sst_dataloader.load_data_files("train")
    selected_data_lst = few_shot_prompt.select_demonstration_instances(demo_candidates)
    print("=" * 10)
    print("STEP 1: select demos.")
    print(selected_data_lst)

    # test get model input
    instace = "I like it."
    input_example = few_shot_prompt.get_model_input(instace, sampled_demonstration_lst=selected_data_lst)
    print("=" * 10)
    print("STEP 2: Assemble Prompt.")
    print(input_example)

    instace = "I like it."
    input_example_rand = few_shot_prompt.get_model_input(instace, demonstrations_candidates=demo_candidates)
    print("=" * 10)
    print("STEP 3: Assemble Prompt With Random.")
    print(input_example_rand)


def gpt3_finetuned_mlm_neighbor_few_shot_prompt_batch():
    data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    sst_dataloader = SST2Dataloader(data_dir)

    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/gpt3_mlm_prompt.json"
    few_shot_prompt = GPT3FewShotSamplingPrompt.from_json_file(config_path)
    print("=" * 10)
    print("STEP 0: check the prompt configs.")
    print(few_shot_prompt)

    # get_model_input_batch(self, instance_text_batch: List[str], demonstrations_candidates: List[DataItem] = None,
    # sampled_demonstration_lst:
    # List[DataItem] = None, teacher_model = None,
    #                                        need_detokenize: bool = True)

    demon_candi = sst_dataloader.load_data_files("train")
    instance_text_batch = sst_dataloader.load_data_files("test")[:3]
    instance_text_batch = [item.text for item in instance_text_batch]
    print(len(instance_text_batch))
    config_file = "/data2/lixiaoya/workspace/gpt-text/tests/file/friday_config.json"
    config = FridayModelConfig.from_json_file(config_file)
    model = FridayClient(config)
    results = few_shot_prompt.get_model_input_batch(instance_text_batch, demonstrations_candidates=demon_candi,
                                                    teacher_model=model)
    print(len(results))
    print(results[0])


def test_flan_t5_prompt():
    prompt = FLANT5Prompt()
    print(prompt)


def test_flan_t5_map_verbalizer():
    json_file = "/data2/lixiaoya/workspace/gpt-text/tests/file/flan_t5_prompt_sst_config.json"
    prompt = FLANT5Prompt.from_json_file(json_file)
    print("=" * 10)
    print("check config ")
    print(prompt.verbalizer)
    print(prompt.inverse_verbalizer)
    print("=" * 10)
    results = prompt.map_predicted_verbalizer_to_label("Positive")
    print(prompt)
    print(results)
    results = prompt.map_predicted_verbalizer_to_label("pos")
    print(prompt)
    print(results)


def test_gpt3_map_label_word():
    config_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/gpt3_mlm_prompt.json"
    few_shot_prompt = GPT3FewShotSamplingPrompt.from_json_file(config_path)
    sent = "CLUES and REASONING: Clues: Heating, Oil, Barge, Price, Decrease, New York Harbor, CTS, Effective, Subsidiary, Reuter.\nReasoning: The input includes multiple references to heating oil, specifically mentioning a price decrease in the New York Harbor, that was effective today. Additionally, it mentions a subsidiary, a barge, CTS, and Reuter, all of which are associated with the heating oil market. These clues indicate that the input is likely related to Heating Oil/Gas Oil.\nTOPIC: Heating Oil/Gas Oil"
    result = few_shot_prompt.map_predicted_verbalizer_to_label(sent)
    print(result)


if __name__ == "__main__":
    # test gpt3 zeroshot
    # test_gpt3_zeroshot_init_prompt()
    # test_gpt3_zeroshot_load_prompt()

    # test roberta prompt
    # test_roberta_basic_prompt()
    # test_roberta_basic_tokenize()
    # test_roberta_zeroshot_prompt()

    # test
    # random_sample_gpt3_prompt()
    # random_kway_sample_gpt3_prompt()

    # test_gpt3_zeroshot_map_predicted_verbalizer_to_label()
    # gpt3_finetuned_mlm_neighbor_few_shot_prompt()
    # gpt3_finetuned_mlm_neighbor_few_shot_prompt_batch()

    # test_flan_t5_prompt()
    # test_flan_t5_map_verbalizer()

    # test_chat_prompt()

    test_gpt3_map_label_word()
