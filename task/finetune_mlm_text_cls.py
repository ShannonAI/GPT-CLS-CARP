#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: task/finetune_mlm_text_cls.py
@time: 2022/12/06 20:03
@desc:
https://github.com/Lightning-AI/lightning/issues/14445
"""

import argparse
import json
import math
import os
import warnings
from collections import OrderedDict
from functools import partial

import torch
from more_itertools import chunked
from sklearn.metrics import accuracy_score
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
from utils.random_seed import set_train_random_seed

# add these two lines, because pytorch-lightning may throw useless userwarning when training multi-class task.
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2757
warnings.filterwarnings('ignore')

set_train_random_seed(2333)

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModelForSequenceClassification
from torchmetrics import Accuracy
from data.dataloader import SST2Dataloader, AGNewsDataloader, TwentyNewsGroupDataloader, R8Dataloader, R52Dataloader, \
    MRDataloader
from data.dataset import FinetuneMLMDataset
from data.data_utils import collate_tensors_to_max_length, collate_to_max_length
from utils.get_logger import get_info_logger
from utils.argparse_utils import get_finetune_mlm_parser


class TextClassificationTask(pl.LightningModule):
    def __init__(
            self,
            args: argparse.Namespace
    ):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_dir = args.bert_path
        if "sst2" in self.args.dataset_name.lower():
            self.dataloader = SST2Dataloader(self.args.data_dir)
        elif "agnews" in self.args.dataset_name.lower():
            self.dataloader = AGNewsDataloader(self.args.data_dir)
        elif "20news" in self.args.dataset_name.lower():
            self.dataloader = TwentyNewsGroupDataloader(self.args.data_dir)
        elif "r8" in self.args.dataset_name.lower():
            self.dataloader = R8Dataloader(self.args.data_dir)
        elif "r52" in self.args.dataset_name.lower():
            self.dataloader = R52Dataloader(self.args.data_dir)
        elif "mr" in self.args.dataset_name.lower():
            self.dataloader = MRDataloader(self.args.data_dir)
        else:
            raise ValueError("Please choose from [sst2*, agnews]")

        self.labels = self.dataloader.get_labels()
        self.num_labels = len(self.labels)
        self.bert_config = AutoConfig.from_pretrained(self.bert_dir,
                                                      num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.bert_dir, config=self.bert_config)
        self.loss_fn = CrossEntropyLoss()
        # it may throws UserWarning: You have set 15 number of classes if different from predicted (11) and target (9) number of classes
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4135
        self.acc = Accuracy(multiclass=True, mdmc_average="global",
                            num_classes=self.num_labels, dist_sync_on_step=True, )

        self.num_gpus = 1
        self.result_logger = get_info_logger("finetune-mlm",
                                             save_log_file=os.path.join(self.args.save_path, "eval_result_log.txt"),
                                             print_to_console=True)
        self.result_logger.info(self.bert_config)
        self.result_logger.info(self.args)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.args.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        else:
            raise ValueError(self.args.optimizer)
        num_gpus = self.num_gpus
        t_total = (len(self.train_dataloader()) // (
                self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.lr_scheduler == "default":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif self.args.lr_scheduler == "multiple_step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
        else:
            raise ValueError(self.args.lr_schedular)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask):
        """"""
        model = self.model(input_ids, attention_mask=attention_mask)
        return model

    def compute_loss_and_acc(self, batch, return_prob: bool = False, current_type: str = "train"):
        input_ids, attention_mask, labels = batch

        y = labels.view(-1)
        y_hat = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        y_logits = y_hat[0].view(-1, self.num_labels)
        # compute loss
        loss = self.loss_fn(y_logits, y)
        # compute acc
        predict_scores = F.softmax(y_logits, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        if current_type != "train":
            self.acc.update(predict_labels, y)
        acc = 1.0
        if return_prob:
            return loss, acc, predict_scores
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch, current_type="train")
        tf_board_logs = {
            "train_loss": loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.acc.reset()
        loss, acc = self.compute_loss_and_acc(batch, current_type="valid")
        return {'val_loss': loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = self.acc.compute()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        self.result_logger.info(
            f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(f"EVAL INFO -> loss: {avg_loss}, acc: {avg_acc}")
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.args.train_file_name)

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""
        dataset = FinetuneMLMDataset(self.dataloader, self.bert_dir, dataset_name=self.args.dataset_name,
                                     data_type=prefix, max_length=self.args.max_length)
        if prefix == "train":
            # define data_generator will help experiment reproducibility.
            data_generator = torch.Generator()
            data_generator.manual_seed(2333)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            data_sampler = SequentialSampler(dataset)

        # sampler option is mutually exclusive with shuffle
        dataloader = DataLoader(dataset=dataset,
                                sampler=data_sampler,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.workers,
                                collate_fn=partial(collate_tensors_to_max_length,
                                                   fill_values=[self.bert_config.pad_token_id, 0]),
                                drop_last=False)

        return dataloader

    def test_dataloader(self):
        return self.get_dataloader(self.args.test_file_name)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.acc.reset()
        loss, acc, probs = self.compute_loss_and_acc(batch, return_prob=True, current_type="test")
        probs = probs.detach().cpu().numpy().tolist()
        pred_label = [0 if prob[0] > prob[1] else 1 for prob in probs]
        return {'test_loss': loss, "test_acc": acc, "test_pred": pred_label}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        save_file_path = os.path.join(self.args.save_path, "pred_test.txt")
        with open(f"{save_file_path}", "w") as f:
            for x in outputs:
                label_batch = x['test_pred']
                for label in label_batch:
                    f.write(f"{label}\n")

        test_acc = self.acc.compute()
        tensorboard_logs = {'test_loss': test_loss, 'test_acc': test_acc}
        self.result_logger.info(f"INFO: Using Best Checkpoint and Test ...")
        self.result_logger.info(f"EVAL INFO -> loss: {test_loss}, acc: {test_acc}")
        return {'test_loss': test_loss, 'log': tensorboard_logs}


def run():
    """main"""

    parser = get_finetune_mlm_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = TextClassificationTask(args)
    if args.pretrain_checkpoint:
        checkpoint = torch.load(args.pretrain_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, 'checkpoint', '{epoch}-{val_loss:.4f}-{val_acc:.4f}'),
        save_top_k=args.save_topk,
        save_last=True,
        monitor="val_acc",
        mode="max",
    )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log'
    )

    # save args
    with open(os.path.join(args.save_path, 'checkpoint', "args.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         logger=logger,
                                         deterministic=True)

    if not args.only_eval:
        trainer.fit(model)
        best_checkpoint_path = checkpoint_callback.best_model_path
        model.result_logger.info("=" * 40)
        model.result_logger.info(f"INFO: BEST CKPT PATH IS {best_checkpoint_path}")
        model.result_logger.info("=" * 40)
    else:
        best_checkpoint_path = args.eval_ckpt_path

    checkpoint = torch.load(best_checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    trainer.test(model)

    print("$=" * 30)
    print("MODEL EVAL")
    evaluate(best_checkpoint_path, mlm_dir=args.bert_path, data_dir=args.data_dir,
             test_file_name=args.test_file_name, dataset_name=args.dataset_name, max_len=args.max_length)

    print("$=" * 30)
    print("FILE EVAL")
    save_file_path = os.path.join(args.save_path, "pred_test.txt")
    file_evaluate(args.data_dir, save_file_path, test_file_name=args.test_file_name, dataset_name=args.dataset_name)


def evaluate(eval_ckpt_path: str, prefix: str = "model", mlm_dir: str = "", data_dir: str = "",
             test_file_name: str = "test", dataset_name: str = "sst2", max_len: int = 200):
    if dataset_name == "sst2":
        dataloader = SST2Dataloader(data_dir)
    elif dataset_name == "agnews":
        dataloader = AGNewsDataloader(data_dir)
    elif dataset_name == "r52":
        dataloader = R52Dataloader(data_dir)
    elif dataset_name == "r8":
        dataloader = R8Dataloader(data_dir)
    elif dataset_name == "mr":
        dataloader = MRDataloader(data_dir)
    else:
        raise ValueError(dataset_name)
    num_labels = len(dataloader.get_labels())

    encoder_weight = torch.load(eval_ckpt_path)['state_dict']
    checkpoint_weight = OrderedDict(
        {key.replace(f"{prefix}.", ""): value.clone()
         for key, value in encoder_weight.items()})

    config = AutoConfig.from_pretrained(mlm_dir, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(mlm_dir, config=config)
    model.load_state_dict(checkpoint_weight, strict=True)
    model.eval()

    dataset = FinetuneMLMDataset(dataloader, mlm_dir, dataset_name=dataset_name,
                                 data_type=test_file_name, max_length=max_len)

    pred_label_lst = []
    gold_label_lst = []
    print("=" * 20)
    print("Label to IDX ")
    print(dataset.label_to_idx)
    print("=" * 20)
    for data_idx, data_item in tqdm(enumerate(chunked(dataset, 12)), total=math.ceil(len(dataset) / 12.0), desc="test"):
        batch_input_ids = [item[0].numpy().tolist() for item in data_item]
        batch_input_ids = collate_to_max_length(batch_input_ids, filled_value=config.pad_token_id)
        batch_attention_mask = [item[1].numpy().tolist() for item in data_item]
        batch_attention_mask = collate_to_max_length(batch_attention_mask, filled_value=0)
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch_label = [item[-1].numpy().tolist()[0] for item in data_item]
        # input_ids, label = data_item
        results = model.forward(batch_input_ids, attention_mask=batch_attention_mask)
        pred_score = F.softmax(results.logits, dim=1)
        pred_labels = torch.argmax(pred_score, dim=-1).numpy().tolist()

        pred_label_lst.extend(pred_labels)
        gold_label_lst.extend(batch_label)

    assert len(pred_label_lst) == len(gold_label_lst)
    print(len(pred_label_lst))
    print(eval_ckpt_path)
    acc_score = accuracy_score(gold_label_lst, pred_label_lst)
    print(f">>> Model ACC is {acc_score}")


def file_evaluate(data_dir: str, save_file_path: str, test_file_name: str = "test", dataset_name: str = "sst2"):
    if dataset_name == "sst2":
        dataloader = SST2Dataloader(data_dir)
    elif dataset_name == "agnews":
        dataloader = AGNewsDataloader(data_dir)
    elif dataset_name == "r8":
        dataloader = R8Dataloader(data_dir)
    elif dataset_name == "r52":
        dataloader = R52Dataloader(data_dir)
    elif dataset_name == "mr":
        dataloader = MRDataloader(data_dir)
    else:
        raise ValueError(dataset_name)

    data_items = dataloader.load_data_files(test_file_name)
    gold_labels = [str(item.label) for item in data_items]
    with open(f"{save_file_path}", "r") as f:
        pred_labels = [item.strip() for item in f.readlines()]

    assert len(gold_labels) == len(pred_labels)
    acc_score = accuracy_score(gold_labels, pred_labels)
    print(f">>> File ACC is {acc_score}")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    run()
