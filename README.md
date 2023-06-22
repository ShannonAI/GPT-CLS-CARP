<div align="center">
  <img src="assets/carp_header_v3.jpg" width="800">
</div>

[Paper Link](https://arxiv.org/abs/2305.08377)<br>

If you find this repo helpful, please cite the following:
```latex
@article{sun2023text,
  title={Text Classification via Large Language Models},
  author={Sun, Xiaofei and Li, Xiaoya and Li, Jiwei and Wu, Fei and Guo, Shangwei and Zhang, Tianwei and Wang, Guoyin},
  journal={arXiv preprint arXiv:2305.08377},
  year={2023}
}
```
For any question, please feel free to post Github issues. <br>


## Overview 

In this paper, we introduce Clue And Reasoning Prompting (CARP), which is a progressive reasoning strategy tailored to addressing the complex linguistic phenomena involved in text classification.
CARP first prompts LLMs to find superficial clues (e.g., keywords, tones, semantic relations, references, etc), based on which a diagnostic reasoning process is deduced for final decisions. 
To further address the limited token issue, CARP uses a fine-tuned model on the supervised dataset for kNN demonstration search in the in-context learning, allowing the model to take the advantage of both LLM’s generalization ability and the task-specific evidence provided by the full labeled dataset. <br>
 
Examples of prompts under zero-shot and few-shot (k=1) settings are shown in the following: <br>

<div align="left">
  <img src="assets/carp_prompts.png" width="900">
</div>


### Data and trained models

| Name | Link             |
|------|------------------|
| Fullset | [Google Drive]() |
|Subset|   [Google Drive]() |
|FT Model |   [Google Drive]() |


### Setup Environment

Before running this project, you need to create a conda environment and install required packages. <br>

```bash 
conda create -n gpt-env python=3.7
conda activate gpt-env
pip install torch==1.8.1+cu111  torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
cd GPT-CLS-CARP
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

After that, please execute the following commands in the terminal for downloading NLTK's dependent files.

```bash 
$ conda activate gpt-env
$ python3 
>>> import nltk
>>> nltk.download('punkt')
```

### Supervised RoBERTa

We release code and scripts for fine-tuning RoBERTa-Large on five text classification datasets, including [SST-2](), [AgNews](), [R8](), [R52](), and [MR]().

### Zero-shot in-context learning 

Scripts for reproducing our experimental results can be found in the `./scripts/<dataset_name>/gpt3_zeroshot/` folder, where 
`<dataset_name>` takes value in `[sst2, agnews, r8, r52, mr]`. <br>
Note that you need to change `DATA_DIR`, `OUTPUT_DIR` to your own dataset path, bert model path and log path, respectively.<br>
For example, run `./scripts/sst2/gpt3_zeroshot/carp_davinci003.sh` will start 
prompt gpt-3 in the zero-shot setting and save intermediate log to `$OUTPUT_DIR`.

### Few-shot in-context learning 

Scripts for reproducing our experimental results can be found in the `./scripts/<dataset_name>/<retriever_type>/gpt3_fewshot/` folder,
, where `<dataset_name>` takes value in `[sst2, agnews, r8, r52, mr]` and `<retriever_type>` in `[ft_retriever_knn, simcse_retriever_knn, random_demo]`.  <br>
Note that you need to change `DATA_DIR`, `OUTPUT_DIR` to your own dataset path, bert model path and log path, respectively.<br>
For example, run `./scripts/sst2/gpt3_fewshot/carp_davinci003.sh` will start 
prompt gpt-3 in the zero-shot setting and save intermediate log to `$OUTPUT_DIR`.

### Results 

Experimental results for the supervised baseline `RoBERTa-Large`, the zero-shot setting, and the few-shot setting with the FT-Retriever are shown in the following table. 
More results (e.g., few-shot in-context learning with SimCSE-Retriever) can be found in the [paper](https://arxiv.org/abs/2305.08377).

| Dataset             | SST-2 | AgNews | R8 | R52 | MR | **Average** |
| ------------------- | :--: | :------------: | :--------: | :---: | :---: | :-----: |
| RoBERTa-Large      | 95.99 | 95.55 | 97.76  | 96.42 | 91.16  | 95.38 |
| **Zero-shot**          |  |   |  |  |  | |
|  Vanilla     | 91.55   |  90.72  |  90.19  |  89.06  |  88.69  |  90.04  |  
|   CoT    | 92.11   |   91.25  |   90.48  |   91.24  |   89.37  |   90.89   |  
| CARP     |  93.01  |  92.60 |  91.75 |  91.80 |  89.94 | 91.82 |  
| **Few-shot (FT-Retriever, k=16)** |  |   |  |  |  | |
| Vanilla     | 94.01 | 94.14 | 95.57 | 95.79 | 90.90 | 94.08 |
| CoT     | 95.48  | 94.89  | 95.59  | 95.89  | 90.17  | 94.40 |
| CARP     | 96.80 | 95.99 | 98.29 | 96.82 | 91.90 | 95.97 |

