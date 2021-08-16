#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
"""
File: network.py
"""

import os
import sys
import json
import shutil
import logging
import torch
import numpy as np
from datetime import datetime

from source.utils.engine import Trainer
from source.utils.generator import TopKGenerator
from source.utils.engine import evaluate
from source.utils.evaluation import evaluate_generation, evaluate_ppl
from source.utils.config import load_config, load_corpus, load_model, seed_everything

import argparse


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="./config/seq2seq.yaml")
    # TODO More For Future

    parser.add_argument('--GPU', type=int, default=None)

    args = parser.parse_args()
    return args

def main():
    """
    main
    """

    args = getArgs()
    config_file = args.config_file

    if not os.path.exists(config_file):
        print("\nLost config file!")
        exit()

    config = load_config(config_file)

    use_gpu_flag = torch.cuda.is_available() and config.gpu >= 0
    config.add(use_gpu=use_gpu_flag)
    device = config.gpu
    torch.cuda.set_device(device)
    if config.random_seed is not None:
        seed_everything(config.random_seed)
    # Data definition
    corpus = load_corpus(config)

    # Iteration definition
    train_iter = corpus.create_batches(
        config.batch_size, "train", shuffle=True, device=device)
    valid_iter = corpus.create_batches(
        config.batch_size, "valid", shuffle=False, device=device)
    test_iter = corpus.create_batches(
        config.batch_size, "test", shuffle=False, device=device)

    # Model definition
    model = load_model(corpus, config)

    model_name = model.__class__.__name__
    # Generator definition
    generator = TopKGenerator(model=model, k=config.beam_size,
                              src_field=corpus.SRC, tgt_field=corpus.OUTPUT,
                              max_length=config.max_dec_len, ignore_unk=config.ignore_unk,
                              length_average=config.length_average, use_gpu=config.use_gpu)

    # Testing
    if config.test and config.ckpt:
        print(model)
        model.load(config.ckpt)
        model.eval()
        
        print("Testing ...")
        corpus.reload()
        test_iter = corpus.create_batches(
            config.batch_size, "test", shuffle=False, device=device)
        # metrics, scores = evaluate(model, test_iter)
        # print(metrics.report_cum())
        print("Generating ...")
        evaluate_generation(generator, test_iter, save_file=config.gen_file, verbos=True)
        # evaluate_ppl(model, test_iter)

    # Training
    else:
        # Load word embeddings
        if config.use_embed and config.embed_file is not None:
            model.encoder.embedder.load_embeddings(
                corpus.SRC.embeddings, scale=0.03)
            model.decoder.embedder.load_embeddings(
                corpus.TGT.embeddings, scale=0.03)
            model.topic_embedder.load_embeddings(
                corpus.TOPIC.embeddings, scale=0.03)

        # Load Topic Distribution
        if config.topic_prepare_matrix is not None:
            topic_distribution_matrix = np.load(config.topic_prepare_matrix)
            model.ntm.topics.init_matrix(topic_distribution_matrix)

        # Optimizer definition
        optimizer = getattr(torch.optim, config.optimizer)(
            model.parameters(), lr=config.lr)
        
        # Learning rate scheduler
        if config.lr_decay is not None and 0 < config.lr_decay < 1.0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                      factor=config.lr_decay, patience=1, verbose=True,
                                                                      min_lr=1e-5)
        else:
            lr_scheduler = None
        
        # Save directory
        date_str, time_str = datetime.now().strftime("%Y%m%d-%H%M%S").split("-")
        save_dir = "{}-{}-{}-{}".format(config.save_dir_prefix, model_name, date_str, time_str)
        config.add(save_dir=save_dir)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        
        # Logger definition
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        fh = logging.FileHandler(os.path.join(config.save_dir, "train.log"))
        logger.addHandler(fh)
        
        # Save config
        params_file = os.path.join(config.save_dir, "params.json")
        with open(params_file, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        print("Saved params to '{}'".format(params_file))
        logger.info(model)
        
        # Train
        logger.info("Training starts ...")
        generator.k = 1 # use greed search
        trainer = Trainer(model=model, optimizer=optimizer, train_iter=train_iter,
                          valid_iter=valid_iter, logger=logger, generator=generator,
                          valid_metric_name="-loss", num_epochs=config.num_epochs,
                          save_dir=config.save_dir, log_steps=config.log_steps,
                          valid_steps=config.valid_steps, grad_clip=config.grad_clip,
                          lr_scheduler=lr_scheduler, save_summary=False, save_mode=config.save_mode)
        if config.ckpt is not None:
            trainer.load(ckpt=config.ckpt)
        trainer.train()
        logger.info("Training done!")

        # Test
        logger.info("")
        best_model_file = os.path.join(config.save_dir, "best.model")
        model.load(best_model_file)
        model.eval()
        # Generator Re-Definition
        generator.k = config.beam_size
        logger.info("Generation starts ...")
        test_gen_file = os.path.join(config.save_dir, config.gen_file)
        evaluate_generation(generator, test_iter, save_file=test_gen_file, verbos=True)
        # evaluate_ppl(model, test_iter)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
