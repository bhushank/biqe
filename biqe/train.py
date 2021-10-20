# coding=utf-8
#   File:     train.py
#   Authors: Bhushan Kotnis bhushan.kotnis@neclab.eu
#             Carolin Lawrence carolin.lawrence@neclab.eu
#             Mathias Niepert mathias.niepert@neclab.eu
#
# NEC Laboratories Europe GmbH, Copyright (c) 2020, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging, torch, os
import numpy as np
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange
import pandas as pd


from biqe.predict import predict, get_eval_dataloader
from biqe.model_helper import save_model, get_train_dataloader, prepare_optimizer

LOGGER = logging.getLogger(__name__)


def get_loss(biqe_args, model, batch):
    """
    Given a batch, gets the loss for the chosen BERT model.

    :param biqe_args: an instance of :py:class:NspArguments
    :param model: a BERT model
    :param batch: the current batch
    :return: the loss of the current batch for the chosen model.
    """
    input_ids, input_mask, segment_ids, position_ids, loss_mask, gen_label_ids, \
    classify_id_cls, classify_id_tokens, _ = batch
    # Get loss
    # Note: the not used dimensions (labels_tok, masked_lm_labels, labels_cls) may not be None
    # but this is fine because the model remembers with which options it was instantiated)
    outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, position_ids=position_ids,
                    loss_mask=loss_mask,labels_tok=classify_id_tokens, masked_lm_labels=gen_label_ids,
                    labels_cls=classify_id_cls)
    #outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
    #                labels_tok=classify_id_tokens[0], masked_lm_labels=gen_label_ids[0],
    #                labels_cls=classify_id_cls[0])
    loss = outputs[0]
    return loss


def train(biqe_args, data_handler, data_handler_predict, model, masker, tokenizer, device, n_gpu):
    """
    Runs training for a model.

    :param biqe_args: Instance of :py:class:NspArguments
    :param data_handler: instance or subclass of :py:class:Bitext, for training
    :param data_handler_predict: instance or subclass of :py:class:Bitext, for validation
    :param model: the model that will be trained
    :param masker: subclass instance of :py:class:Masking
    :param tokenizer: instance of BertTokenzier
    :param device: the device to run the computation on
    :param n_gpu: number of gpus used
    :return: 0 on success
    """
    train_examples = data_handler.examples
    num_train_steps = \
        int(len(train_examples) / biqe_args.train_batch_size /
            biqe_args.gradient_accumulation_steps * biqe_args.num_train_epochs)


    optimizer, scheduler = prepare_optimizer(biqe_args, model, num_train_steps)

    if biqe_args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=biqe_args.fp16_opt_level)

    get_train_dataloader(biqe_args, masker, data_handler, tokenizer)

    best_valid_score = 0.0  # if validation is run during training, keep track of best
    avg_loss = 0.0

    n_params = sum([p.nelement() for p in model.parameters()])
    LOGGER.info("Number of parameters: %d", n_params)

    tokenizer.save_pretrained(biqe_args.output_dir)

    global_step = 0
    epoch_step = 0
    model.zero_grad()

    #TODO add incremental loading of batches for large datasets where not everything fits in RAM
    for epoch in trange(int(biqe_args.num_train_epochs), desc="Epoch"):
        LOGGER.info("Starting Epoch %s:", epoch)
        #if epoch>0:
            #LOGGER.info("Recreating Targets")
            #get_train_dataloader(biqe_args, masker, data_handler, tokenizer)

        # some masking changes at every epoch, thus reload if necessary
        if biqe_args.masking_strategy is not None and epoch != 0:  # already done for first epoch
            LOGGER.info("Recreating masks")
            get_train_dataloader(biqe_args, masker, data_handler, tokenizer)

        for step, batch in enumerate(tqdm(data_handler.train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            loss = get_loss(biqe_args, model, batch)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if biqe_args.gradient_accumulation_steps > 1:
                loss = loss / biqe_args.gradient_accumulation_steps

            if biqe_args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(optimizer), biqe_args.max_grad_norm)
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), biqe_args.max_grad_norm)

            if (step + 1) % biqe_args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses,
                # only needed for fp16, else handled in optimizer.py
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                avg_loss += loss.item()
                epoch_step += 1

        # Validate on the dev set if desired.
        if biqe_args.valid_every_epoch:
            avg_loss = avg_loss / epoch_step
            LOGGER.info('Average loss since last validation: %0.15f', avg_loss)
            avg_loss = 0
            LOGGER.info('Current learning rate and number of updates performed: %0.15f, %d',
                        scheduler.get_lr()[0], global_step)
            best_valid_score = validate(best_valid_score, biqe_args, data_handler_predict, masker,
                                        tokenizer, model, device, epoch)

    # save last model if we didn't pick the best during training
    if not biqe_args.valid_every_epoch:
        LOGGER.info("Saving final model")
        save_model(biqe_args, model)

    return best_valid_score


def obtain_attention(biqe_args, data_handler, masker, tokenizer, model, device):
    model.eval()
    all_attns, all_input_ids = [],[]
    data_handler.read_examples(is_training=False)
    get_eval_dataloader(biqe_args, masker, data_handler, tokenizer)
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, position_ids, loss_mask, _, _, _, example_indices in\
                tqdm(data_handler.eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            position_ids = position_ids.to(device)
            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                                    position_ids=position_ids)
            attn = torch.mean(outputs[-1][-1], dim=1) # Take the last layer and average across all heads
            all_attns.extend(attn.tolist())
            all_input_ids.extend(input_ids.tolist())

    process_attns(all_attns, all_input_ids, tokenizer, biqe_args.output_dir)

def process_attns(all_attns, all_input_ids, tokenizer, output_dir):

    def extract_matrix(attn, input_ids):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        matrix, y_axis, x_axis = [],[],[]
        source_counter, rel_counter = 1, dict()
        mask_counter = 1
        query_count=3
        for count,token in enumerate(tokens):
            if token=='[PAD]':
                break
            elif token == '[MASK]':
                counter = mask_counter if mask_counter<=2 else query_count
                y_axis.append(f'[M]({counter})')
                x_axis.append(f'[M]({counter})')
                matrix.append(attn[count])
                if mask_counter>2:
                    query_count+=1
                mask_counter += 1
            elif '/m/' in token: # entity token
                x_axis.append(f"S{source_counter}")
                source_counter+=1
            elif token == '[SEP]' or token == '':
                x_axis.append('[S]')
                mask_counter = 1
            else:
                if token not in rel_counter:
                    rel_counter[token] = len(rel_counter)+1
                x_axis.append(f"R{rel_counter[token]}")
        for count,row in enumerate(matrix):
            row = np.asarray(row[:len(x_axis)])
            matrix[count] = row/np.sum(row)
        df = pd.DataFrame(data=np.asarray(matrix),index=y_axis, columns=x_axis)
        return df
    if not os.path.exists(os.path.join(output_dir,'attn_dfs')):
        os.mkdir(os.path.join(output_dir,'attn_dfs'))
    for count in range(len(all_input_ids)):
        data_frame = extract_matrix(all_attns[count], all_input_ids[count])
        data_frame.to_pickle(os.path.join(output_dir,'attn_dfs',f'dag_{count}'))





def validate(best_valid_score, biqe_args, data_handler_predict, masker, tokenizer, model, device,
             epoch):
    """
    After an epoch of training, validate on the validation set.

    :param best_valid_score: the currently best validation score
    :param biqe_args: an instance of :py:class:NspArguments
    :param data_handler_predict: instance or subclass instance of :py:class:Bitext,
     on which to run prediction
    :param masker: an instance of a subclass of :py:class:Masking
    :param tokenizer: the BERT tokenizer
    :param model: the BERT model
    :param device: where to run computations
    :param epoch: the current epoch
    :return: the new best validation score
    """
    model.eval()
    if best_valid_score == 0.0:  # then first epoch, save model
        save_model(biqe_args, model)
    deciding_score = predict(biqe_args, data_handler_predict, masker, tokenizer, model,
                             device, epoch)
    if best_valid_score < deciding_score:
        LOGGER.info("Epoch %s: Saving new best model: %s vs. previous %s",
                    epoch, deciding_score, best_valid_score)
        save_model(biqe_args, model)
        best_valid_score = deciding_score
    model.train()
    return best_valid_score
