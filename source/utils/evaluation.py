#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#

import numpy as np
import torch

from source.utils.metrics import bleu, distinct, renda_perplexity as perplexity
from source.utils.metrics import EmbeddingMetrics



def evaluate_generation(generator,
                        data_iter,
                        save_file=None,
                        num_batches=None,
                        verbos=False,
                        embMetric=False,):
    """
    evaluate_generation
    """
    results = generator.generate(batch_iter=data_iter,
                                 num_batches=num_batches)

    refs = [result.tgt.split(" ") for result in results]
    hyps = [result.preds[0].split(" ") for result in results]

    report_message = []

    avg_len = np.average([len(s) for s in hyps])
    report_message.append("Avg_Len-{:.3f}".format(avg_len))

    bleu_1, bleu_2 = bleu(hyps, refs)
    report_message.append("Bleu-{:.4f}/{:.4f}".format(bleu_1, bleu_2))

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(hyps)
    report_message.append("Inter_Dist-{:.4f}/{:.4f}".format(inter_dist1, inter_dist2))
    report_message.append("Intra_Dist-{:.4f}/{:.4f}".format(intra_dist1, intra_dist2))


    # embedding metrics 
    # embed_metric = EmbeddingMetrics(field=generator.tgt_field)
    # ext_sim, avg_sim, greedy_sim = embed_metric.embed_sim(
    #     hyp_texts=[' '.join(ws) for ws in hyps],
    #     ref_texts=[' '.join(ws) for ws in refs])
    # report_message.append(
    #     f"Embed(E/A/G)-{ext_sim:.4f}/{avg_sim:.4f}/{greedy_sim:.4f}")

    report_message = "   ".join(report_message)

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(refs)
    avg_len = np.average([len(s) for s in refs])
    target_message = "Target:   AVG_LEN-{:.3f}   ".format(avg_len) + \
                     "Inter_Dist-{:.4f}/{:.4f}".format(inter_dist1, inter_dist2) + " Intra_Dist-{:.4f}/{:.4f}".format(
        intra_dist1, intra_dist2)

    message = report_message + "\n" + target_message

    if save_file is not None:
        write_results(results, save_file)
        print("Saved generation results to '{}'".format(save_file))
    if verbos:
        print(message)
    else:
        return message


def write_results(results, results_file):
    """
    write_results
    """
    with open(results_file, "w", encoding="utf-8") as f:
        for result in results:
            """
            f.write("Source : {}\n".format(result.src))
            f.write("Target : {}\n".format(result.tgt))
            if "cue" in result.keys():
                f.write("Cue : {}\n".format(result.cue))
            if "prior_attn" in result.keys():
                f.write("Prior Attn: {}\n".format(' '.join([str(value) for value in result.prior_attn.data.tolist()])))
            if "posterior_attn" in result.keys():
                f.write("Posterior Attn: {}\n".format(' '.join([str(value) for value in result.posterior_attn.data.tolist()])))
            if "gumbel_attn" in result.keys():
                f.write("Gumbel Attn: {}\n".format(' '.join([str(value) for value in result.gumbel_attn.data.tolist()])))
            if "indexs" in result.keys():
                f.write("Indexs : {}\n".format(result.indexs))
            if "weights" in result.keys():
                f.write("Weights : {}\n".format(result.weights))
            """
            for pred, score in zip(result.preds, result.scores):
                # f.write("Predict: {} ({:.3f})\n".format(pred, score))
                # f.write("{}\t{:.3f}\n".format(pred, score))
                f.write("{}\n".format(pred))
            # f.write("\n")



def evaluate_ppl(model, data_iter, verbos=True):
    toatl_ppl = 0
    toatl_nums = 0
    padding_idx = model.padding_idx

    for batch_id, inputs in enumerate(data_iter, 1):

        enc_inputs = inputs
        dec_inputs = inputs.tgt[0][:, :-1], inputs.tgt[1] - 1
        probs = model(enc_inputs, dec_inputs).logits
        tgt = inputs.tgt[0][:, 1:]
        batch_ppl = perplexity(logits=probs, targets=tgt, )

        toatl_ppl += batch_ppl  
        toatl_nums += 1

    ppl = toatl_ppl / toatl_nums
    message = f"PPL: {ppl:.4f}"
    if verbos:
        print(message)
    else:
        return message

