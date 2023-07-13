#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/data/data_retriever.py
@time: 2022/12/06 20:03
@desc:
"""

from data.data_retriever import FinetunedMLMRetriever, SimCSERetriever
from data.dataloader import SST2Dataloader


def test_finetuned_mlm_retriever():
    mlm_name_or_path = "/data2/lixiaoya/hz_data/hz03/data/models/bert_uncased_large"
    encoder_ckpt_path = "/data/lixiaoya/outputs/poc_cls/0102/fold1_robert_domain_comp_v1_5_24_3e-5_0.003_0.01_280_0.2_5_0.25/checkpoint/epoch=3-val_loss=0.2269-val_acc=0.9200.ckpt"
    retriever = FinetunedMLMRetriever(mlm_name_or_path, encoder_ckpt_path)
    # print(retriever.__dict__)
    candidate_sent_file = "/data2/lixiaoya/workspace/gpt-text/tests/file/sent_lst.txt"
    retriever.build_index(candidate_sent_file)
    results = retriever.search("I like cats.")
    print(results)


def test_finetuned_mlm_save_results():
    data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    mlm_name_or_path = "/data2/lixiaoya/hz_data/hz03/data/models/roberta-large"
    encoder_ckpt_path = "/data2/lixiaoya/outputs/gpt-text/sst2_fix/sst2_roberta_large/epoch5_bs36_lr4e-5_weightdecay0.005_warmup0.01_maxlen200_dropout0.2_grad3/checkpoint/epoch=4-val_loss=0.1368-val_acc=0.9588.ckpt"
    retriever = FinetunedMLMRetriever(mlm_name_or_path, encoder_ckpt_path)
    dataloader = SST2Dataloader(data_dir)
    save_file_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/test_find.jsonl"
    retriever.search_nearest_neighbors_and_save_to_file(save_file_path, dataloader=dataloader,
                                                        candidate_type="train", query_type="test",
                                                        search_threshold=0.0, top_k=24,
                                                        ranking_model="finetuned-roberta-large")
    print(f"successfully save to {save_file_path}")


def test_load_saved_nn_results():
    data_dir = "/data2/lixiaoya/datasets/pytorch-sentiment-classification/data/SST2"
    mlm_name_or_path = "/data2/lixiaoya/hz_data/hz03/data/models/roberta-large"
    encoder_ckpt_path = "/data2/lixiaoya/outputs/gpt-text/sst2_fix/sst2_roberta_large/epoch5_bs36_lr4e-5_weightdecay0.005_warmup0.01_maxlen200_dropout0.2_grad3/checkpoint/epoch=4-val_loss=0.1368-val_acc=0.9588.ckpt"
    save_file_path = "/data2/lixiaoya/workspace/gpt-text/tests/file/test_find.jsonl"
    retriever = FinetunedMLMRetriever(mlm_name_or_path, encoder_ckpt_path, saved_nearest_neighbor_file=save_file_path)
    dataloader = SST2Dataloader(data_dir)
    retriever.build_index(dataloader, )
    results = retriever.search(
        "a gob of drivel so sickly sweet , even the eager consumers of moore 's pasteurized ditties will retch it up like rancid cr me br l e",
        threshold=0.0,
        top_k=16)
    print(results)


def test_simcse_retriever():
    mlm_name_or_path = "/data2/lixiaoya/hz_data/hz03/data2/models/sup-simcse-bert-large-uncased"
    retriever = SimCSERetriever(mlm_name_or_path)
    sentence = ["I like you "]
    result = retriever.encode(sentence)

    print(result)


if __name__ == "__main__":
    # test_finetuned_mlm_retriever()
    # test_simcse_retriever()
    # test_finetuned_mlm_save_results()
    test_load_saved_nn_results()
