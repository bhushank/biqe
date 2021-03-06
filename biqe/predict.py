# coding=utf-8
#   File:     predict.py
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
"""
Handles the prediction with biqe, for generation, token and sentence classification.
"""

import sys
import logging
import os
import math
import re

from tqdm import tqdm
import numpy as np
import scipy

import torch
from torch.utils.data import DataLoader, SequentialSampler

from .util import write_list_to_file, compute_softmax

LOGGER = logging.getLogger(__name__)


def get_predictor(biqe_args):
    """
    Factory for returning various predictors
    :param biqe_args: an instance of :py:class:NspArguments
    :return: an instance of :py:class:GreedyPredictor or a subclass
    """
    predictor = None
    if biqe_args.predict == 'one_step_greedy' or biqe_args.predict == 'greedy' \
            or biqe_args.predict == 'all_at_once':
        predictor = GreedyPredictor(biqe_args)
    elif biqe_args.predict == 'left2right' or biqe_args.predict == 'max_probability' \
            or biqe_args.predict == 'min_entropy' \
            or biqe_args.predict == 'right2left' or biqe_args.predict == 'no_look_ahead':
        predictor = IterativeGreedy(biqe_args)
    else:
        LOGGER.error("Argument for --predict invalid. Aborting.")
        sys.exit(1)
    return predictor


class GreedyPredictor():
    """
    Predicts the most likely token for each [MASK] until the first [SEP] is predicted in one step.
    Also support classification on the [CLS] token and token classification.
    """
    def __init__(self, biqe_args):
        """
        Initializes the Predictor using an instance of :py:class:NspArguments.

        If biqe_args.plus_classify_sequence > 0, prediction on the [CLS] token is run
        If biqe_args.plus_classify_tokens > 0, predicton on each token is run
        If biqe_args.plus_generation > 0, generate words from masks
        If biqe_args.return_attention, then the probability of the attention matrices are returned
        (not implemented yet)

        :param biqe_args: instance of :py:class:NspArguments
        """
        self.plus_classify_sequence = biqe_args.plus_classify_sequence
        self.plus_classify_tokens = biqe_args.plus_classify_tokens
        self.plus_generation = biqe_args.plus_generation
        self.return_attention = biqe_args.return_attention

    def predict_dataset(self, data_handler, tokenizer, model, device):
        """
        Given a data set (via data_handler), run predictions
        :param data_handler: an instance or a subclass instance of :py:class:BitextHandler
        :param tokenizer: an instance of :py:class:BertTokenizer
        :param model: the model that should predict
        :param device: the device to move the computation to
        :return: a tuple of:
                 a list of generated sequences
                 a list of classification outputs as predicted on [CLS]
                 a list of token classifications (both Part A & Part B,
                    what is actually needed, can be decided by the dataset handler in
                    arrange_token_classify_output()
                 attention probability matrices (not implemented)
                 the prediction order (not returned here, since all at once where produced)
        """
        all_results_gen = []
        all_results_classify = []
        all_results_classify_tokens = []
        # Iterate over prediction dataset
        for input_ids, input_mask, segment_ids, position_ids,loss_mask, _, _, _, example_indices in \
                tqdm(data_handler.eval_dataloader, desc="Evaluating"):
            batch_gen_label_ids, batch_classify_id_cls_logits, \
            batch_classify_tokens_logits, _ = \
                self.get_model_output(model, input_ids, segment_ids, input_mask,position_ids,loss_mask, device)
            # example_indices keeps track of which overall example we are operating on,
            # despite the minibatching
            for i, example_index in enumerate(example_indices):
                current_example = data_handler.examples[example_index]

                # get classification on [CLS]
                if batch_classify_id_cls_logits is not None:
                    classify_logits = batch_classify_id_cls_logits[i].detach().cpu().tolist()
                    max_classify_index = self.predict_classification(classify_logits)
                    # change output as specified for dataset,
                    # e.g. mapping a number to the name of the label
                    classified_example = data_handler.arrange_classify_output(current_example,
                                                                              max_classify_index)
                    all_results_classify.append(classified_example)

                # get token classification
                if batch_classify_tokens_logits is not None:
                    classify_tokens_logits = batch_classify_tokens_logits[i].detach().cpu().numpy()
                    current_input_ids = input_ids[i].detach().cpu().tolist()
                    classified_example = data_handler.token_classification_prediction(
                        data_handler.examples[example_index], classify_tokens_logits, current_input_ids, tokenizer)
                    ''''
                    max_classified_tokens_indices = \
                        self.predict_token_classification(classify_tokens_logits)
                    # change output as specified for dataset,
                    # e.g. mapping a number to the name of the label
                    classified_example = data_handler.arrange_token_classify_output(
                        data_handler.examples[example_index], max_classified_tokens_indices,
                        current_input_ids, tokenizer)
                    '''
                    all_results_classify_tokens.append(classified_example)

                # get generation
                if batch_gen_label_ids is not None:
                    gen_logits = batch_gen_label_ids[i].detach().cpu().numpy()
                    generated_text = self.predict_greedy_generation(input_ids[i],
                                                                    gen_logits,
                                                                    tokenizer)
                    generated_text = " ".join(generated_text)
                    generated_text = generated_text.replace(" ##", "")
                    #LOGGER.info("generated_text: %s" % generated_text)
                    # change output as specified for dataset, e.g. creating a json object for
                    # downstream evaluation
                    arranged_text = data_handler.arrange_generated_output(
                        data_handler.examples[example_index], generated_text.strip())
                    all_results_gen.append(arranged_text)
                else:
                    all_results_gen.append("")
        return all_results_gen, all_results_classify, all_results_classify_tokens, None, None

    def get_model_output2(self, model, input_ids, segment_ids, input_mask, device):
        """
        For a set of inputs, get the output the model produces.
        :param model: the model
        :param input_ids: minibatch of input ids (see corresponding subclass instance of
        :py:class:Masking)
        :param segment_ids: minibatch of segment ids (see corresponding subclass instance of
        :py:class:Masking)
        :param input_mask: minibatch of input masks (see corresponding subclass instance of
        :py:class:Masking)
        :param device: where to run the computation, e.g. gpu
        :return: a tuple, where an element of the tuple is none if the corresponding model does not
        predict this type
                1. the minibatch of ids for generation
                2. the minibatch of ids for classification on [CLS] token
                3. the minibatch of ids for classification on the tokens
                4. the attention probabilitiy matrices (not implemented yet)
        """
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        batch_gen_label_ids = None
        batch_classify_id_cls_logits = None
        batch_classify_tokens_logits = None
        if self.plus_generation > 0:
            batch_gen_label_ids = []
        if self.plus_classify_sequence > 0:
            batch_classify_id_cls_logits = []
        if self.plus_classify_tokens > 0:
            batch_classify_tokens_logits = []
        hidden_states = None
        attention_probs = None
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            counter = 0
            # the following if order needs to stay the same and needs to follow BertGST
            for i in range(self.plus_generation):
                batch_gen_label_ids.append(outputs[i])
            for i in range(self.plus_classify_sequence):
                batch_classify_id_cls_logits.append(outputs[i])
            for i in range(self.plus_classify_tokens):
                batch_classify_tokens_logits.append(outputs[i])
            #TODO: access biqe_args here to see if hidden states or attention are true
        return batch_gen_label_ids, batch_classify_id_cls_logits, batch_classify_tokens_logits, \
               attention_probs

    def get_model_output(self, model, input_ids, segment_ids, input_mask, position_ids,loss_mask, device):
        """
        For a set of inputs, get the output the model produces.
        :param model: the model
        :param input_ids: minibatch of input ids (see corresponding subclass instance of
        :py:class:Masking)
        :param segment_ids: minibatch of segment ids (see corresponding subclass instance of
        :py:class:Masking)
        :param input_mask: minibatch of input masks (see corresponding subclass instance of
        :py:class:Masking)
        :param device: where to run the computation, e.g. gpu
        :return: a tuple, where an element of the tuple is none if the corresponding model does not
        predict this type
                1. the minibatch of ids for generation
                2. the minibatch of ids for classification on [CLS] token
                3. the minibatch of ids for classification on the tokens
                4. the attention probabilitiy matrices (not implemented yet)
        """
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        position_ids = position_ids.to(device)
        loss_mask = loss_mask.to(device)
        batch_gen_label_ids = None
        batch_classify_id_cls_logits = None
        batch_classify_tokens_logits = None
        hidden_states = None
        attention_probs = None
        with torch.no_grad():
            #try:
                outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, position_ids=position_ids)
                counter = 0
                # the following if order needs to stay the same and needs to follow BertGST
                if self.plus_generation > 0:
                    batch_gen_label_ids = outputs[counter]
                    counter += 1
                if self.plus_classify_sequence > 0:
                    batch_classify_id_cls_logits = outputs[counter]
                    counter += 1
                if self.plus_classify_tokens > 0:
                    batch_classify_tokens_logits = outputs[counter]
                    #batch_classify_tokens_logits = batch_classify_tokens_logits*loss_mask.unsqueeze(2)
                    #batch_classify_tokens_logits = torch.mean(batch_classify_tokens_logits,1)
                    #assert len(batch_classify_tokens_logits.size())<=2
                    counter += 1
            #except Exception as e:
                #f = open('log.txt', 'w')
                #f.write(f"Error {e}")
                #f.close()
            #TODO: access biqe_args here to see if hidden states or attention are true
        return batch_gen_label_ids, batch_classify_id_cls_logits, batch_classify_tokens_logits, \
               attention_probs

    @staticmethod
    def predict_token_classification(logits):
        """
        Given a data set, index for current example and logits for classification on several tokens,
        output the most likely classification label
        :param data_handler: an instance of a subclass of :py:class:DatasetHandler
        :param logits: logits for each token to be classified, each with dimension
                data_handler.num_labels_tok
        :param example_index: index of the current example, example can be accessed via
                data_handler.examples[example_index]
        :return: a list of most likely class label
        """
        classified_tokens = []
        for i, _ in enumerate(logits):
            max_vocab_index = np.argmax(logits[i])
            classified_tokens.append(max_vocab_index)
        return classified_tokens

    @staticmethod
    def predict_classification(logits):
        """
        Given a data set, index for current example and logits for classification on [CLS],
        output the most likely classification label
        :param data_handler: an instance of a subclass of :py:class:DatasetHandler
        :param logits: logits for the [CLS] token with dimension data_handler.num_labels_cls
        :param example_index: index of the current example, example can be accessed via
                data_handler.examples[example_index]
        :return: the most likely class label
        """
        max_classify_index = np.argmax(logits)
        return max_classify_index

    @staticmethod
    def predict_greedy_generation(input_ids, logits, tokenizer):
        """
        Given a data set, index for current example and logits for generation,
        obtain the most likely generation sequence in a greedy manner
        :param input_ids: the input ids, will produce token if id equals [MASK]
        :param logits: logits for the sequence of dimenson [max_seq_length][vocabulary_size]
        :param example_index: index of the current example, example can be accessed via
                data_handler.examples[example_index]
        :return: a list of word pieces
        """
        generated_text = []
        for j, token_logits in enumerate(logits):
            # then we are done processing Part A + first [SEP]
            if input_ids[j] == tokenizer.vocab['[MASK]']:
                max_vocab_index = np.argmax(token_logits)
                predicted_token = tokenizer.ids_to_tokens[max_vocab_index]
                if predicted_token == "[SEP]":
                    # then we predicted end of sequence token and we should stop generating
                    break
                generated_text.append(predicted_token)
        return generated_text


#TODO can we speed this up by using computations from the previous model call?
class IterativeGreedy(GreedyPredictor):
    """
    Iteratively predicts a token, then recomputes with the new token in place.
    Prediction options are:
    1. left to right (left2right)
    2. highest probability (max_probability)
    3. lowest entropy (min_entropy)
    4. right to left (very slow) (right2left)
    5. left to right, but no attention to future tokens (no_look_ahead)

    Inherits from GreedyPredictor for the functions:
    get_model_output, predict_token_classification and predict_classification
    """
    def __init__(self, biqe_args):
        """
        Initializes the predictor.

        If biqe_args.plus_classify_sequence, prediction on the [CLS] token is run
        If biqe_args.plus_classify_tokens, predicton on each token is run
        If biqe_args.plus_generation, generate words from masks in Part B

        biqe_args.predict is the prediction strategy, see above for the options of this predictor.

        :param biqe_args: instance of :py:class:NspArguments
        """
        super().__init__(biqe_args)
        self.predict = biqe_args.predict

    @staticmethod
    def cut_future_connections(tokenizer, input_ids, segment_ids, input_mask):
        """
        Cuts the future connections for the prediction type 'no_look_ahead'.
        :param tokenizer: a BERT tokenizer
        :param input_ids: minibatch of input ids (see corresponding subclass instance of
                :py:class:Masking)
        :param segment_ids: minibatch of segment ids (see corresponding subclass instance of
                :py:class:Masking)
        :param input_mask: minibatch of input masks (see corresponding subclass instance of
                :py:class:Masking)
        :return: (input_ids, segment_ids, input_mask) where the future connections are cut.
        """
        for i, _ in enumerate(input_ids):  # was example_indices previously
            first_mask = True
            for j, _ in enumerate(input_ids[i]):
                if input_ids[i][j] == tokenizer.vocab["[MASK]"]:
                    if first_mask is True:
                        first_mask = False
                    else:
                        input_mask[i][j] = 0
                        segment_ids[i][j] = 0
        return input_ids, segment_ids, input_mask

    def iteratively_opterate_on_batch(self, example_indices, data_handler, tokenizer, model,
                                      input_ids, segment_ids, input_mask, device):
        """
        Given a batch, iteratively calls the model until all instances are done generating.

        :param example_indices: keeps track of which overall example we are operating on
        :param data_handler: an instance or a subclass instance of :py:class:BitextHandler
        :param tokenizer: an instance of :py:class:BertTokenizer
        :param model: the model that should predict
        :param input_ids: minibatch of input ids (see corresponding subclass instance of
                :py:class:Masking)
        :param segment_ids: minibatch of segment ids (see corresponding subclass instance of
                :py:class:Masking)
        :param input_mask: minibatch of input masks (see corresponding subclass instance of
                :py:class:Masking)
        :param device: the device to move the computation to
        :return: a tuple of
                1. a list of all generated texts for this batch
                2. the prediciton order of the invidual texts
                    (only applicable to 'max_probability' and 'min_entropy')
                3. batch_classify_id_cls_logits: the logits on the [CLS] token of the last iteration
                4. batch_classify_tokens_logits: the token classify logits of the last iteration
        """
        # TODO finished examples should be taken out of the calling list since it is now
        # unnecessary  -> causes overhead atm
        # Keeps track of which instances in the minibatch have found their [SEP] token
        finished_generating = [0] * len(example_indices)
        generated_texts = [0] * len(example_indices)
        next_to_uncover_index = [-1] * len(example_indices)
        # On the first call, it will record the first MASK position found and the prediction
        # order that is printed out will be relative to this first position
        if self.predict == 'max_probability' or self.predict == 'min_entropy':
            first_gen_position = [None] * len(example_indices)
            prediction_order = [None] * len(example_indices)
        else:
            first_gen_position = None
            prediction_order = None
        batch_classify_id_cls_logits = None
        batch_classify_tokens_logits = None
        while np.sum(finished_generating) != len(example_indices):
            # break once all instance in minibatch are done
            batch_gen_label_ids, batch_classify_id_cls_logits, batch_classify_tokens_logits, _ = \
                self.get_model_output(model, input_ids, segment_ids, input_mask, device)
            # example_indices keeps track of which overall example we are operating on,
            # despite the minibatching
            for i, example_index in enumerate(example_indices):
                if finished_generating[i] == 1:  # then this example is already done
                    continue
                # get generation part
                gen_logits = batch_gen_label_ids[i].detach().cpu().numpy()
                if self.predict == 'left2right' or self.predict == 'no_look_ahead':
                    input_ids[i], input_mask[i], segment_ids[i], next_to_uncover_index[i] = \
                        self.predict_left_to_right(data_handler, gen_logits, example_index,
                                                   tokenizer, input_ids[i], input_mask[i],
                                                   segment_ids[i], next_to_uncover_index[i])
                elif self.predict == 'right2left':
                    input_ids[i], input_mask[i], segment_ids[i] = \
                        self.predict_right_to_left(data_handler, gen_logits, example_index,
                                                   tokenizer, input_ids[i], input_mask[i],
                                                   segment_ids[i])
                elif self.predict == 'max_probability' or self.predict == 'min_entropy':
                    input_ids[i], input_mask[i], segment_ids[i], position_generated, \
                    first_gen_position[i] = \
                        self.predict_iterative_greedy(data_handler, gen_logits, example_index,
                                                      tokenizer, input_ids[i], input_mask[i],
                                                      segment_ids[i], first_gen_position[i])
                    if prediction_order[i] is None:
                        prediction_order[i] = [position_generated]
                    else:
                        prediction_order[i].append(position_generated)
                if tokenizer.vocab["[MASK]"] not in input_ids[i]:
                    # then all positions have been generated
                    finished_generating[i] = 1
                    generated_ids = []
                    collect_id = False
                    for current_id in input_ids[i]:
                        if current_id == tokenizer.vocab["[SEP]"]:
                            if collect_id is False:
                                # First [SEP], everything after is generated
                                collect_id = True
                                continue
                            else:  # Second [SEP], we are done
                                break
                        if collect_id is True:
                            generated_ids.append(current_id.item())
                    generated_text = tokenizer.convert_ids_to_tokens(generated_ids)
                    generated_text = " ".join(generated_text)
                    generated_text = generated_text.replace(" ##", "")
                    arranged_text = data_handler.arrange_generated_output(
                        data_handler.examples[example_index], generated_text.strip())
                    generated_texts[i] = arranged_text
                    #LOGGER.info("generated_text: %s" % generated_text)
        return generated_texts, prediction_order, \
               batch_classify_id_cls_logits, batch_classify_tokens_logits

    def predict_dataset(self, data_handler, tokenizer, model, device):
        """
        Given a data set (via data_handler), run predictions,
        for options see description of the class.
        :param data_handler: an instance or a subclass instance of :py:class:BitextHandler
        :param tokenizer: an instance of :py:class:BertTokenizer
        :param model: the model that should predict
        :param device: the device to move the computation to
        :return: a tuple of:
                 a list of generated sequences
                 a list of classification outputs as predicted on [CLS]
                 a list of token classifications (both Part A & Part B,
                    what is actually needed, can be decided by the dataset handler in
                    arrange_token_classify_output()
                 attention probability matrices (not implemented)
                 the prediction order (relevant for maximum probability or minimum entropy)
        """
        all_results_gen = []
        all_results_classify = []
        all_results_classify_tokens = []
        all_attention = None
        all_prediction_order = None

        for input_ids, input_mask, segment_ids, _, _, _, example_indices in \
                tqdm(data_handler.eval_dataloader, desc="Evaluating"):
            # for left2right, to keep track of which position should be unconvered next
            if self.predict == 'no_look_ahead':
                # then cut future connections
                input_ids, segment_ids, input_mask = self.cut_future_connections(tokenizer,
                                                                                 input_ids,
                                                                                 segment_ids,
                                                                                 input_mask)

            # iteratively get output until generation for everything is done
            generated_texts, prediction_order, \
            batch_classify_id_cls_logits, batch_classify_tokens_logits = \
                self.iteratively_opterate_on_batch(example_indices, data_handler, tokenizer, model,
                                                   input_ids, segment_ids, input_mask, device)

            # get classification from the last state,
            # the idea here is that now the most information is available
            cls_classified_example, token_classified_example = \
                self.get_classification_from_last_state(example_indices, data_handler, input_ids,
                                                        tokenizer, batch_classify_id_cls_logits,
                                                        batch_classify_tokens_logits)
            if cls_classified_example is not None:
                all_results_classify += cls_classified_example

            if token_classified_example is not None:
                all_results_classify_tokens += token_classified_example

            all_results_gen += generated_texts
            if prediction_order is not None:
                if all_prediction_order is None:
                    all_prediction_order = []
                all_prediction_order += prediction_order
        return all_results_gen, all_results_classify, all_results_classify_tokens, \
               all_attention, all_prediction_order

    def get_classification_from_last_state(self, example_indices, data_handler, input_ids,
                                           tokenizer, batch_classify_id_cls_logits,
                                           batch_classify_tokens_logits):
        """
        For sentence ([CLS]) and token classification,
        get the output from the last generating state.

        :param example_indices: keeps track of which overall example we are operating on
        :param data_handler: an instance or a subclass instance of :py:class:BitextHandler
        :param input_ids: minibatch of input ids (see corresponding subclass instance of
                :py:class:Masking)
        :param tokenizer: an instance of :py:class:BertTokenizer
        :return: a tuple of the [CLS] classification and the token classification
        """
        cls_classified_example = None
        token_classified_example = None
        if batch_classify_id_cls_logits is not None:
            cls_classified_example = [0] * len(example_indices)
        if batch_classify_tokens_logits is not None:
            token_classified_example = [0] * len(example_indices)
        for i, example_index in enumerate(example_indices):
            current_example = data_handler.examples[example_index]
            if batch_classify_id_cls_logits is not None:
                max_classify_index = self.predict_classification(
                    batch_classify_id_cls_logits[i].detach().cpu().tolist())
                cls_classified_example[i] = data_handler.arrange_classify_output(current_example,
                                                                              max_classify_index)

            if batch_classify_tokens_logits is not None:
                current_input_ids = input_ids[i].detach().cpu().tolist()
                max_classified_tokens_indices = \
                    self.predict_token_classification(
                        batch_classify_tokens_logits[i].detach().cpu().numpy())
                token_classified_example[i] = data_handler.arrange_token_classify_output(
                    data_handler.examples[example_index], max_classified_tokens_indices,
                    current_input_ids, tokenizer)

        return cls_classified_example, token_classified_example

    def predict_left_to_right(self, data_handler, logits, example_index, tokenizer,
                              input_ids, input_mask, segment_ids, next_to_uncover_index=-1):

        """
        Given a data set, index for current example and logits for generation,
        obtain the most likelist output for the left-most [MASK]

        :param data_handler: an instance of a subclass of :py:class:DatasetHandler
        :param logits: logits for the sequence of dimenson [max_seq_length][vocabulary_size]
        :param example_index: index of the current example, example can be accessed via
                data_handler.examples[example_index]
        :param tokenizer: instance of :py:class:BertTokenizer
        :param input_ids: minibatch of input ids (see corresponding subclass instance of
                :py:class:Masking)
        :param segment_ids: minibatch of segment ids (see corresponding subclass instance of
                :py:class:Masking)
        :param input_mask: minibatch of input masks (see corresponding subclass instance of
                :py:class:Masking)
        :param next_to_uncover_index: the position at which to uncover the next word,
                if not supplied, we iterate to find the first mask token (more time consuming)
        :return: a tuple of:
                1. input_ids: the newly generated word is written added so it can be called with
                   this information in the next turn
                2. input_mask: changes if
                   a) for no look ahead: switch on the attention to the next token only
                   b) [SEP] is found: no more generation necessary, everything after is set to 0
                3. segment_ids: same as input_mask
                4. the next position to uncover in the next turn
        """
        current_feature = data_handler.features[example_index]
        assert len(current_feature.segment_ids) == len(logits)
        if next_to_uncover_index == -1:
            # If this information is not available, we need to iterate to find the first mask token
            for j, _ in enumerate(logits):
                if input_ids[j] == tokenizer.vocab["[MASK]"]:
                    next_to_uncover_index = j
                    break
        if next_to_uncover_index < len(input_ids):
            token_probabilities = compute_softmax(logits[next_to_uncover_index], 1.0)
            # for current output position, find most likely token in vocab
            vocab_index = np.argmax(token_probabilities)
            input_ids[next_to_uncover_index] = int(vocab_index)
            if self.predict == 'no_look_ahead' and (next_to_uncover_index+1) < len(input_mask):
                # switch on for next word
                input_mask[next_to_uncover_index+1] = 1
                segment_ids[next_to_uncover_index+1] = 1
            if vocab_index == tokenizer.vocab["[SEP]"]:
                # then we modify the input, which will indicate to the calling function
                # that there are no [MASK] tokens left and generation can be stopped
                j = next_to_uncover_index+1
                while True:
                    if j == len(input_ids):  # entire sequence is used
                        break
                    if input_ids[j] == 0:
                        break
                    input_ids[j] = 0
                    input_mask[j] = 0
                    segment_ids[j] = 0
                    j += 1
        next_to_uncover_index += 1
        return input_ids, input_mask, segment_ids, next_to_uncover_index

    # The function would benefit from a next_to_uncover_index as predict_left_to_right,
    # but it is so slow, we probably won't use it again anyway.
    @staticmethod
    def predict_right_to_left(data_handler, logits, example_index, tokenizer,
                              input_ids, input_mask, segment_ids):
        """
        Given a data set, index for current example and logits for generation,
        obtain the most likelist output for the right-most [MASK]

        :param data_handler: an instance of a subclass of :py:class:DatasetHandler
        :param logits: logits for the sequence of dimenson [max_seq_length][vocabulary_size]
        :param example_index: index of the current example, example can be accessed via
                data_handler.examples[example_index]
        :param tokenizer: instance of :py:class:BertTokenizer
        :param input_ids: minibatch of input ids (see corresponding subclass instance of
                :py:class:Masking)
        :param segment_ids: minibatch of segment ids (see corresponding subclass instance of
                :py:class:Masking)
        :param input_mask: minibatch of input masks (see corresponding subclass instance of
                :py:class:Masking)
        :param next_to_uncover_index: the position at which to uncover the next word,
                if not supplied, we iterate to find the first mask token (more time consuming)
        :return: a tuple of:
                1. input_ids: the newly generated word is written added so it can be called with
                   this information in the next turn
                2. input_mask: changes if
                   a) for no look ahead: switch on the attention to the next token only
                   b) [SEP] is found: no more generation necessary, everything after is set to 0
                3. segment_ids: same as input_mask
                4. the next position to uncover in the next turn
        """
        current_feature = data_handler.features[example_index]
        assert len(current_feature.segment_ids) == len(logits)
        vocab_index = 0.0
        current_index = 0
        for j in reversed(range(len(logits))):
            token_logits = logits[j]
            if input_ids[j] == tokenizer.vocab["[MASK]"]:  # then we found next prediction location
                token_probabilities = compute_softmax(token_logits, 1.0)
                # for current output position, find most likely token in vocab
                vocab_index = np.argmax(token_probabilities)
                input_ids[j] = int(vocab_index)
                current_index = j
                break
        if vocab_index == tokenizer.vocab["[SEP]"]:
            # then we modify the input so that we do not attend to anything that happens after [SEP]
            j = current_index + 1
            while True:
                if input_ids[j] == 0:
                    break
                input_ids[j] = 0
                input_mask[j] = 0
                segment_ids[j] = 0
                j += 1
                if j == len(input_ids):  # entire sequence is used
                    break
        return input_ids, input_mask, segment_ids

    def predict_iterative_greedy(self, data_handler, logits, example_index, tokenizer,
                                 input_ids, input_mask, segment_ids, first_gen_position=None):
        """
        Given a data set, index for current example and logits for generation,
        obtain the most likelist output for either
        1. the token with highest probability or
        2. the [MASK] with lowest entropy over the output vocab (i.e. highest certainty)

        Note that the sequence length can change over iterations, e.g. we might first place a [SEP]
        on position 10, but later place another at position 8, then everything after position 8 is
        deleted.

        :param data_handler: an instance of a subclass of :py:class:DatasetHandler
        :param logits: logits for the sequence of dimenson [max_seq_length][vocabulary_size]
        :param example_index: index of the current example, example can be accessed via
                data_handler.examples[example_index]
        :param tokenizer: instance of :py:class:BertTokenizer
        :param input_ids: One instance of input ids
        :param input_mask: One instance of input masks
        :param segment_ids: One instance of segment ids
        :param first_gen_position: minibatch, where each entry indicates for this example
                in which position the first [MASK] token was found. It will be used to print out
                the prediction order relative to this position.
                (Intended to be used when a whole sequence is produced.)
        :return: a tuple of:
                1. input_ids: the newly generated word is written added so it can be called with
                   this information in the next turn
                2. input_mask: changes if [SEP] is found: no attending further than that [SEP]
                3. segment_ids: same as input_mask
                4. which position the just generated token is at
                  (Note that there will be a potential mismatch at the end because words could
                  be split into subwords and we record subword positions).
        """
        current_feature = data_handler.features[example_index]
        assert len(current_feature.segment_ids) == len(logits)
        max_token_prob = - math.inf  # over all outputs, find the most likely token
        max_token_prob_index = -1
        max_token_prob_vocab_index = -1
        for j, token_logits in enumerate(logits):
            if input_ids[j] == tokenizer.vocab["[MASK]"]:
                if first_gen_position is None:
                    first_gen_position = j
            if input_ids[j] == tokenizer.vocab["[MASK]"]:
                # then we want to compute the softmax over the output vocab.
                token_probabilities = compute_softmax(token_logits, 1.0)
                # for current output position, find most likely token in vocab
                # max_vocab_index is also relevant for min_entropy strategy,
                # if we found the min_entropy [MASK}, we still want the most likeliest token
                # for this position
                max_vocab_index = np.argmax(token_probabilities)
                if self.predict == 'min_entropy':
                    # we negate the entropy so we can still look for the max,
                    # in line with looking for the max probability
                    neg_entropy = -scipy.stats.entropy(token_probabilities)
                    max_vocab_prob = neg_entropy
                else:
                    max_vocab_prob = token_probabilities[max_vocab_index]
                if max_vocab_prob > max_token_prob:
                    max_token_prob = max_vocab_prob
                    max_token_prob_index = j
                    max_token_prob_vocab_index = max_vocab_index
        input_ids[max_token_prob_index] = int(max_token_prob_vocab_index)
        if max_token_prob_vocab_index == tokenizer.vocab["[SEP]"]:
            # then we modify the input so that we do not attend to anything that happens after [SEP]
            j = max_token_prob_index+1
            while True:
                if j == len(input_ids):  # entire sequence is used
                    break
                if input_ids[j] == 0:
                    break
                input_ids[j] = 0
                input_mask[j] = 0
                segment_ids[j] = 0
                j += 1
        position_generated = max_token_prob_index - first_gen_position
        return input_ids, input_mask, segment_ids, position_generated, first_gen_position


def get_eval_dataloader(biqe_args, masker, data_handler, tokenizer):
    """
    Creates a sequential DataLoader for the data to be predicted on.
    :param biqe_args: instance of :py:class:NspArguments
    :param masker:
    :param data_handler: an instance or a subclass instance of :py:class:BitextHandler
    :param tokenizer: instance of :py:class:BertTokenizer
    :return: a DataLoader
    """
    is_training = False
    masker.convert_examples_to_features(
        data_handler=data_handler,
        tokenizer=tokenizer,
        max_seq_length=biqe_args.max_seq_length,
        max_part_a=biqe_args.max_part_a,
        is_training=is_training,
        biqe_args=biqe_args)

    LOGGER.info("***** Running predictions *****")
    LOGGER.info("  Num orig examples = %d", len(data_handler.examples))
    LOGGER.info("  Num split examples = %d", len(data_handler.features))
    LOGGER.info("  Batch size = %d", biqe_args.predict_batch_size)

    eval_data = data_handler.create_tensor_dataset()
    eval_sampler = SequentialSampler(eval_data)
    data_handler.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=biqe_args.predict_batch_size)
    return 0


def predict(biqe_args, data_handler, masker, tokenizer, model, device, epoch_counter=None):
    """
    Predicts outputs for a dataset.

    :param biqe_args: instance of :py:class:NspArguments
    :param data_handler: an instance or a subclass instance of :py:class:BitextHandler
    :param tokenizer: instance of :py:class:BertTokenizer
    :param model: instance of a Bert model
    :param device: device used for computations
    :param epoch_counter: if we are predicting during training, this is the epoch we are in
    :return:
    """
    LOGGER.info("Start evaluating")
    if epoch_counter is not None:
        LOGGER.info("  Epoch = %d", epoch_counter)

    # Prepare data set
    data_handler.read_examples(is_training=False)
    get_eval_dataloader(biqe_args, masker, data_handler, tokenizer)

    # Run prediction for full data
    predictor = get_predictor(biqe_args)
    all_results_gen, all_results_classify, all_results_classify_tokens, all_attention, \
    all_prediction_order = predictor.predict_dataset(data_handler, tokenizer, model, device)

    output_prediction_file = os.path.join(biqe_args.output_dir, "predictions.")

    if epoch_counter is not None:
        output_prediction_file = os.path.join(output_prediction_file+str(epoch_counter)+".")

    results_collection = {}

    # Write prediction order to file if applicable
    # (highest probability and lowest entropy predict strategies)
    if all_prediction_order is not None:
        write_list_to_file(all_prediction_order, output_prediction_file+'gen.order')

    #TODO write attentions to file.

    # Token classification
    if biqe_args.plus_classify_tokens > 0:
        # writes out the raw output
        write_list_to_file(all_results_classify_tokens, output_prediction_file+'tok')
        # asks the data_handler to combine the classification appropriately
        # the two might be identical for some datasets
        combined_classification_tokens = \
            data_handler.combine_classification_tokens(all_results_classify_tokens)
        data_handler.write_tok_predictions(combined_classification_tokens,
                                           output_prediction_file+'combined_tok')
        valid_gold = biqe_args.valid_gold[0] if isinstance(biqe_args.valid_gold, list) else biqe_args.valid_gold
        results = data_handler.evaluate(output_prediction_file+'combined_tok', valid_gold,tokenizer)
        results_collection.update(results)
        data_handler.write_eval(results, output_prediction_file+'combined_tok.eval')
        if epoch_counter is not None:
            LOGGER.info("Epoch %s, Token Classification Validation results: %s",
                        epoch_counter, results)
        else:
            LOGGER.info("Token Classification Validation results: %s", results)

    # [CLS] classification
    if biqe_args.plus_classify_sequence > 0:
        data_handler.write_cls_predictions(all_results_classify, output_prediction_file+'cls')
        results = data_handler.evaluate(output_prediction_file+'cls', 'cls')
        results_collection.update(results)
        data_handler.write_eval(results, output_prediction_file+'cls.eval')
        if epoch_counter is not None:
            LOGGER.info("Epoch %s: CLS Classification Validation results: %s",
                        epoch_counter, results)
        else:
            LOGGER.info("CLS Classification Validation results: %s", results)

    # Generation
    if biqe_args.plus_generation > 0:
        data_handler.write_predictions(all_results_gen, output_prediction_file+'gen')
        results = data_handler.evaluate(output_prediction_file+'gen', 'generation')
        data_handler.write_eval(results, output_prediction_file+'gen.eval')
        results_collection.update(results)
        if epoch_counter is not None:
            LOGGER.info("Epoch %s, Generated Validation results: %s",
                        epoch_counter, results)
        else:
            LOGGER.info("Generated Validation results: %s", results)

    # For the data set, select the score that will decide which model to keep
    deciding_score = data_handler.select_deciding_score(results_collection)

    return deciding_score
