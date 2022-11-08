import random
import logging
from numpy.lib.function_base import average

import torch
import numpy as np

from scipy.stats import pearsonr, spearmanr
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

from src import KoBertTokenizer, HanBertTokenizer
from transformers import (
    BertConfig,
    ElectraConfig,
    ElectraTokenizer,
    ElectraModel,
    BertForSequenceClassification,
    ElectraForSequenceClassification,
    BertForTokenClassification,
    ElectraForTokenClassification,
)

CONFIG_CLASSES = {
    "kobert": BertConfig,
    "koelectra-base": ElectraConfig,
}

TOKENIZER_CLASSES = {
    "kobert": KoBertTokenizer,
    "koelectra-base": ElectraTokenizer,
}

MODEL_FOR_SEQUENCE_CLASSIFICATION = {
    "kobert": BertForSequenceClassification,
    "koelectra-base": ElectraForSequenceClassification,
}

MODEL_FOR_REGRESSION = {
    "koelectra-base": ElectraModel
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
def simple_accuracy(labels, preds):
    return (labels == preds).mean()

def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }

def pearson_and_spearman(labels, preds):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def f1_pre_rec(labels, preds, is_ner=True):
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds, suffix=True),
            "recall": seqeval_metrics.recall_score(labels, preds, suffix=True),
            "f1": seqeval_metrics.f1_score(labels, preds, suffix=True),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }


def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds, suffix=True)


def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    if task_name == 'wellness':
        return acc_score(labels, preds)
    else:
        raise KeyError(task_name)