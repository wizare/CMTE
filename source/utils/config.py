import argparse
import os
import random
import torch
import numpy as np
import importlib
from source.utils.misc import load_yaml


def load_config(config_file):
    # model_config

    filename = config_file
    config = load_yaml(filename)
    return config


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('Setting random seed as {}'.format(seed))


def load_corpus(config):
    # Data Definition

    ip_module = importlib.import_module(config.corpus_module_path)
    Corpus = getattr(ip_module, config.corpus_class_name)
    corpus = Corpus(config)

    corpus.load(rebuild=config.rebuild)
    
    return corpus


def load_model(corpus, config):
    # Model Definition

    ip_module = importlib.import_module(config.model_module_path)
    Seq2Seq = getattr(ip_module, config.model_class_name)

    model = Seq2Seq(corpus, config)

    return model
