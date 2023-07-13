#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: save_nearest_neighbors.sh


PROJECT_PATH=/home/lixiaoya/GPT-CLS-CARP
export PYTHONPATH="$PYTHONPATH:$PROJECT_PATH"


DATA_DIR=/data2/lixiaoya/gpt_data_models/data/sst2_original
MLM_DIR=/data2/lixiaoya/hz_data/hz03/data/models/roberta-large
CKPT_PATH=/data2/lixiaoya/outputs/gpt-text/sst2_fix/sst2_roberta_large/original_gpu8_epoch5_bs16_lr1e-5_weightdecay0.1_warmup0.06_maxlen200_dropout0.2_grad1/checkpoint/epoch=4-val_loss=0.0216-val_acc=0.9553.ckpt
CANDI_TYPE=train
THRESHOLD=0.0
TOPK=24
RANKING=finetuned_roberta-large
MAX_LEN=280
# 2333, 8866, 1314, 6624, 9998

SEED=9998
QUERY_TYPE=test
SAVE_PATH=/data2/lixiaoya/gpt_data_models/sst2_nearest_neighbors/test_${RANKING}_cand${CANDI_TYPE}_thres${THRESHOLD}_top${TOPK}_seed${SEED}.jsonl

#
#CUDA_VISIBLE_DEVICES=1 python3 ${PROJECT_PATH}/task/save_nearest_neighbors.sh \
#--seed ${SEED} \
#--data_dir ${DATA_DIR} \
#--mlm_dir ${MLM_DIR} \
#--encoder_ckpt_path ${CKPT_PATH} \
#--candidate_type ${CANDI_TYPE} \
#--query_type ${QUERY_TYPE} \
#--search_threshold ${THRESHOLD} \
#--top_k ${TOPK} \
#--ranking_model ${RANKING} \
#--save_nearest_neighbor_path ${SAVE_PATH} \
#--max_len ${MAX_LEN}

RANKING=sup_simcse_roberta-large
SAVE_PATH=/data2/lixiaoya/gpt_data_models/nearest_neighbors/sst2_nearest_neighbors/test_${RANKING}_cand${CANDI_TYPE}_thres${THRESHOLD}_top${TOPK}_seed${SEED}.jsonl

CUDA_VISIBLE_DEVICES=1 python3 ${PROJECT_PATH}/task/save_nearest_neighbors.py \
--seed ${SEED} \
--data_dir ${DATA_DIR} \
--mlm_dir ${MLM_DIR} \
--encoder_ckpt_path ${CKPT_PATH} \
--candidate_type ${CANDI_TYPE} \
--query_type ${QUERY_TYPE} \
--search_threshold ${THRESHOLD} \
--top_k ${TOPK} \
--ranking_model ${RANKING} \
--save_nearest_neighbor_path ${SAVE_PATH} \
--max_len ${MAX_LEN} \
--retriever_type "simcse"


