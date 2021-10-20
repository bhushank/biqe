# coding=utf-8

#   File:     masking.py
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

import sys
import random
import logging
import numpy as np
from tqdm import tqdm
LOGGER = logging.getLogger(__name__)


class GenInputFeatures:
    """Feature for one data point."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 position_ids,
                 loss_mask,
                 gen_label_ids,
                 classify_id_cls,
                 classify_id_tokens=None):
        """
        General possible structure of an input sentence:
        [CLS] Part A [SEP] Part B [SEP] <Padding until max_seq_length>
        :param input_ids: contains the vocabulary id for each unmasked token,
                            masked tokens receive the value of [MASK]
        :param input_mask: 1 prior to padding, 0 for padding
        :param segment_ids: 0 for Part A, 1 for Part B, 0 for padding.
        :param gen_label_ids: -1 for unmasked tokens, vocabulary id for masked tokens
        :param classify_id_cls: the gold value that we want to predict on the [CLS] token
        :param classify_id_tokens: the gold values for token classification
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.loss_mask = loss_mask
        self.gen_label_ids = gen_label_ids
        self.classify_id_cls = classify_id_cls
        self.classify_id_tokens = classify_id_tokens


def get_masker(nsp_args):
    """
    Factory for returning a masker.
    :param nsp_args: the command line arguments
    :return: an instance of :py:class:Masking
    """
    return Masking(nsp_args)


class Masking:
    """
    Superclass for maskers.
    They take a data_handler, subclass instance of :py:class:DatasetHandler,
    and convert the elements of data_handler.examples into a set of features,
    which are stored in data_handler.features.

    Same index indicates same example/feature.

    data_handler.examples is a list of subclass instances of :py:class:GenExample
    data_hanlder.examples ist a list of instances of :py:class:GenInputFeatures

    Subclasses should implement handle_masking, should call this class's init in its own.
    convert_examples_to_features should stay the same.
    """
    def __init__(self, nsp_args):
        """
        Keeps track of some statistics.

        violate_max_part_a_len: how often in data_handler, the maximum query
        length (Part A) was violated
        violate_max_gen_len: how often in data_handler, the maximum generation
        length (Part B) was violated
        trunc_part_b: how often Part B was truncated
        trunc_part_a: how often Part A was truncated
        plus_classify_tokens: needs to be set to True if we classify on each token

        """
        self.violate_max_part_a_len = 0
        self.violate_max_gen_len = 0
        self.trunc_part_b = 0
        self.trunc_part_a = 0
        self.max_gen_b_length = nsp_args.max_gen_b_length
        self.max_gen_a_length = nsp_args.max_gen_a_length
        self.plus_generation = nsp_args.plus_generation
        self.plus_classify_tokens = nsp_args.plus_classify_tokens
        self.max_part_a = nsp_args.max_part_a
        self.mask_in_a = nsp_args.mask_in_a
        self.mask_in_b = nsp_args.mask_in_b
        self.masking_strategy = nsp_args.masking_strategy
        # for masking_strategy = 'bernoulli' or 'gaussian'
        self.mean = nsp_args.distribution_mean
        # for masking_strategy = 'gaussian'
        self.stdev = nsp_args.distribution_stdev
        LOGGER.info("Mean: %s, Variance: %s", self.mean, self.stdev)

    def _binomial(self, mask_list):
        for index, _ in enumerate(mask_list):
            if mask_list[index] == 1.0:
                binomial = np.random.binomial(1, self.mean)
                if binomial == 0:
                    mask_list[index] = 0.0
        return mask_list

    def set_up_mask(self, data_handler, part_a, part_b):
        if self.masking_strategy == 'dataset_dependent':
            # Create masks (does not cover [CLS] or [SEP]
            mask_list_a, mask_list_b = \
                data_handler.possible_mask_locations(part_a, part_b, is_training=True)
            # at the moment, mask_list would mask all possible instances, now apply binomial
            mask_list_a = self._binomial(mask_list_a)
            mask_list_b = self._binomial(mask_list_b)
        else:
            mask_list_a = self._create_stochastic_mask(len(part_a))
            mask_list_b = self._create_stochastic_mask(len(part_b))
        return mask_list_a, mask_list_b

    def _create_stochastic_mask(self, len_mask_list):
        """
        Given a length, it uses the specified masking strategy to create a corresponding
        masking list.

        :param len_mask_list: the length the masking list will have to be.
        :return: the masking list, where it is 1.0 if a mask should be placed in that position
        """
        mask_list = [0.0] * len_mask_list
        if self.masking_strategy == 'bernoulli':
            #1.0 means mask
            for i, _ in enumerate(mask_list):
                sample = random.random()
                if sample < self.mean:
                    mask_list[i] = 1.0
        elif self.masking_strategy == 'gaussian':
            current_threshold = np.random.normal(self.mean, self.stdev)
            if current_threshold > 1.0:
                current_threshold = 1.0
            if current_threshold < 0.0:
                current_threshold = 0.0
            nr_masks = int(round(current_threshold * len_mask_list))
            mask_list = [1.0] * nr_masks + [0.0] * (len_mask_list - nr_masks)
            random.shuffle(mask_list)
        return mask_list

    def handle_masking(self, part_a, part_b, is_training, max_seq_length, tokenizer,
                       example_index=-1, data_handler=None):
        """
        Convert a part_a and a part_b into 4 lists needed to instantiate :py:class:GenInputFeatures
        :param part_a: a string of text of Part A, i.e. part_a of a subclass instance of
        :py:class:GenExample
        :param part_b: a string of text of Part B, i.e. part_b of a subclass instance of
        :py:class:GenExample
        :param is_training: true if training, part_b is only considered for training
        :param max_seq_length: the maximum sequence length (Part A + Part B)
        :param tokenizer: an instance of :py:class: BertTokenizer
        :param example_index: the index of the current sample, e.g. i when iterating over
        data_handler.examples[i]
        :param data_handler: an instance or subclass instance of :py:class:BitextHandler
        :return: a 4-tuple of lists, each with length max_seq_length
            input_ids: ids of "[cls] part a [sep] part b [sep]" or a masking thereof
            input_mask: 1 for all spots that should be attended to
            segment_ids: 0 up to and including the first [sep], 1 until second [sep] or for
            remainder of sequence
            gen_label_ids: -1 for positions in input_ids that should not be predicted,
                           the id of the to-be-predicted token,
                           should be always -1 at test time
        """
        mask_list_a = None
        mask_list_b = None
        # +1 because the [SEP] token needs to be learnt too in the case of masking
        part_a_with_sep = part_a + ['[SEP]']
        part_b_with_sep = part_b #+ ['[SEP]']
        # note: even if only one is true, we do set up both but just never use the other mask list
        if (self.mask_in_a is True or self.mask_in_b is True) and is_training is True:
            #TODO: what if we have a LM task and still want to see how good we do when we predict on
            #MASK tokens, then we need to place MASK tokens at prediction time
            #easier option might be to just sort this out in the test set by writing MASK in the
            #input file.
            #if self.max_gen_b_length == 0 and self.max_gen_a_length == 0:
            mask_list_a, mask_list_b = self.set_up_mask(data_handler,
                                                        part_a_with_sep, part_b_with_sep)
            assert len(mask_list_a) == len(part_a_with_sep)
            assert len(mask_list_b) == len(part_b_with_sep)

        tokens = []
        segment_ids = []
        gen_label_ids = []
        position_ids, loss_mask = [], []
        # no need of CLS
        #tokens.append("[CLS]")
        #segment_ids.append(0)
        #gen_label_ids.append(-1)

        # Part A
        # if: then add only padding
        if is_training is False and self.max_gen_a_length > 0:
            for _ in range(self.max_gen_a_length):
                tokens.append('[MASK]')
                segment_ids.append(0)
                gen_label_ids.append(-1)
        # else: then if is_training apply masks, if not apply no masks, used for sequence and token
        # classification, or if plus_generation, then assumed that the input has [MASK] in the
        # correct positions
        else:
            # self.max_part_a - 1: save space for [sep]
            if self.mask_in_a is True and is_training is True:
                part_a_tokens, part_a_segment_ids, part_a_gen_label_ids = \
                    self.apply_masking(part_a_with_sep, example_index, tokenizer, mask_list_a, 0,
                                       (self.max_part_a - 1))
            else:
                part_a_tokens, part_a_segment_ids, part_a_gen_label_ids = \
                    self.apply_no_masking(part_a_with_sep, example_index, 0, (self.max_part_a - 1))
            tokens += part_a_tokens
            segment_ids += part_a_segment_ids
            gen_label_ids += part_a_gen_label_ids

            # If generation is set up, pad with mask tokens up to max_gen_a_length
            if len(part_a_with_sep) < self.max_gen_a_length and self.plus_generation > 0:
                for _ in range(self.max_gen_a_length - len(part_a_with_sep)):
                    tokens.append('[MASK]')
                    segment_ids.append(0)
                    gen_label_ids.append(-1)

        # Part B
        # if: then add only padding
        if is_training is False and self.max_gen_b_length > 0:
            for _ in range(self.max_gen_b_length):
                tokens.append('[MASK]')
                segment_ids.append(1)
                gen_label_ids.append(-1)
        # else: then if is_training apply masks, if not apply no masks, used for sequence and token
        # classification, or if plus_generation, then assumed that the input has [MASK] in the
        # correct positions
        else:
            if self.mask_in_b is True and is_training is True:
                part_b_tokens, part_b_segment_ids, part_b_gen_label_ids = \
                    self.apply_masking(part_b_with_sep, example_index, tokenizer, mask_list_b, 1,
                                       max_seq_length)
            else:
                part_b_tokens, part_b_segment_ids, part_b_gen_label_ids = \
                    self.apply_no_masking(part_b_with_sep, example_index, 1, max_seq_length)
            tokens += part_b_tokens
            segment_ids += part_b_segment_ids
            gen_label_ids += part_b_gen_label_ids

            # If generation is set up, pad with mask tokens up to max_gen_b_length
            if len(part_b_with_sep) < self.max_gen_b_length and self.plus_generation > 0:
                for _ in range(self.max_gen_b_length - len(part_b_with_sep)):
                    tokens.append('[MASK]')
                    segment_ids.append(1)
                    gen_label_ids.append(-1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        #Bhushan
        #position_ids = list(range(len(tokens)))
        #loss_mask = [0]*len(tokens)

        counter = 0
        for t in tokens:
            position_ids.append(counter)
            counter+=1
            if t=='[SEP]':
                counter=0
            if t=='[MASK]':
                loss_mask.append(1)
            else:
                loss_mask.append(0)
        # Pad to maximum sequence length

        last_pos = max_seq_length-1
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            gen_label_ids.append(-1)
            position_ids.append(last_pos)
            loss_mask.append(0)

        return input_ids, input_mask, segment_ids, [gen_label_ids], position_ids, loss_mask

    def convert_examples_to_features(self, data_handler, tokenizer, max_seq_length, max_part_a,
                                     is_training, nsp_args):
        """
        From a list of examples (subclass instances of :py:class:GenExample),
        creates a list of instances of :py:class:GenInputFeatures
        :param data_handler: a subclass instance of :py:class:DatasetHandler;
                will access data_handler.examples (list of subclass instances of
                :py:class:GenExample)
                and will set data_handler.features (list of instances of :py:class:GenInputFeatures)
        :param tokenizer: an instance of :py:class: BertTokenizer
        :param max_seq_length: the maximum sequence length ([CLS] + Part A + [SEP] + Part B + [SEP])
        :param max_part_a: the maximum length of Part A
        :param is_training: true if training, handles gold label construction
        :param nsp_args: the command line arguments
        :return:0 on success
        """
        data_handler.features = []
        max_a = 0
        max_b = 0
        plus_generation_warning_given = False
        plus_classify_sequence_warning_given = False
        plus_classify_tokens_warning_given = False

        # iterate over subclass instances of :py:class:GenExample
        for i, example in enumerate(tqdm(data_handler.examples)):
            data_handler.update_info(i)
            # Part A
            part_a = tokenizer.tokenize(example.part_a)
            max_a = max(max_a, len(part_a))
            if len(part_a) > max_part_a:
                if data_handler.truncate_end:
                    part_a = part_a[0:max_part_a]
                    self.trunc_part_a += 1
                else:  # truncate beginning
                    # +2 because we save space for [CLS] and [SEP]
                    first_trunc_index = len(part_a) - max_part_a + 2
                    part_a = part_a[first_trunc_index:]
                    self.trunc_part_a += 1

            # Part B
            part_b = tokenizer.tokenize(example.part_b)
            max_b = max(max_b, len(part_b))
            max_part_b = max_seq_length + max_part_a
            if len(part_b) > max_part_b:
                if data_handler.truncate_end:
                    part_b = part_b[0:max_part_b]
                    self.trunc_part_b += 1
                else:  # truncate beginning
                    # +1 because we save space for [SEP]
                    first_trunc_index = len(part_b) - max_part_b + 1
                    part_b = part_b[first_trunc_index:]
                    self.trunc_part_b += 1

            classify_id_cls = []
            if is_training:
                classify_id_cls = example.classify_id_cls

            # Masking for one example, handled by subclass of :py:class:Masker
            input_ids, input_mask, segment_ids, gen_label_ids, position_ids, loss_mask = \
                self.handle_masking(part_a, part_b, is_training, max_seq_length, tokenizer, i,
                                    data_handler)

            # How tokens should be classified is the job of the dataset specific class
            classify_id_tokens = []
            if self.plus_classify_tokens > 0:
                classify_id_tokens = data_handler.get_token_classification_ids(example,
                                                                               input_ids,
                                                                               tokenizer)
            # sanity checks
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(position_ids) == max_seq_length
            assert len(loss_mask) == max_seq_length

            for head in gen_label_ids:
                assert len(head) == max_seq_length
            if len(gen_label_ids) != nsp_args.plus_generation and \
                    plus_generation_warning_given is not True:
                LOGGER.warning("The number of generation heads assumed by the dataset handler does "
                               "not match the number of generation heads specified by the command "
                               "line argument plus_generation")
                plus_generation_warning_given = True
            if len(classify_id_cls) != nsp_args.plus_classify_sequence and \
                    plus_classify_sequence_warning_given is not True:
                LOGGER.warning("The number of sequence classification heads assumed by the dataset "
                               "handler does not match the number of generation heads specified "
                               "by the command line argument plus_classify_sequence")
                plus_classify_sequence_warning_given = True
            if len(classify_id_tokens) != nsp_args.plus_classify_sequence and \
                    plus_classify_tokens_warning_given is not True:
                LOGGER.warning("The number of token classification heads assumed by the dataset "
                               "handler does not match the number of generation heads specified "
                               "by the command line argument plus_classify_tokens")
                plus_classify_tokens_warning_given = True

            if example.example_index < 1:
                LOGGER.info("*** Example ***")
                LOGGER.info("Feature for example: %s", i)
                LOGGER.info("part_a: %s", part_a)
                LOGGER.info("part_b: %s", part_b)
                LOGGER.info("input_ids: %s", input_ids)
                LOGGER.info("input_mask: %s", input_mask)
                LOGGER.info("segment_ids: %s", segment_ids)
                LOGGER.info("gen_label_ids: %s", gen_label_ids)
                LOGGER.info("classify_id_cls: %s", classify_id_cls)
                LOGGER.info("classify_id_tokens: %s", classify_id_tokens)

            feature = GenInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                position_ids = position_ids,
                loss_mask = loss_mask,
                gen_label_ids=gen_label_ids,  #transformer head expects a list, mock it here (it is not further implemented because it has not been needed yet)
                classify_id_cls=classify_id_cls,
                classify_id_tokens=classify_id_tokens)
            data_handler.features.append(feature)

        # Every example has exactly one corresponding features at the same index
        assert len(data_handler.examples) == len(data_handler.features)
        LOGGER.info("Maximum Part A is: %s", max_a)
        LOGGER.info("Maximum Part B is: %s", max_b)
        LOGGER.warning("Couldn't encode query length %s times.", self.violate_max_part_a_len)
        LOGGER.warning("Couldn't encode generation length %s times.", self.violate_max_gen_len)
        LOGGER.warning("Truncated part b %s times.", self.trunc_part_b)
        LOGGER.warning("Truncated part a %s times.", self.trunc_part_a)
        return 0

    def apply_no_masking(self, part, example_index, seg_id, max_len):
        tokens = []
        segment_ids = []
        gen_label_ids = []
        for token in part:
            tokens.append(token)
            segment_ids.append(seg_id)
            gen_label_ids.append(-1)
            if len(tokens) == max_len:  # save space for [SEP]
                LOGGER.debug("Can't encode the maximum Part A length of example number %s",
                             example_index)
                self.violate_max_part_a_len += 1
                break
        return tokens, segment_ids, gen_label_ids

    def apply_masking(self, part, example_index, tokenizer, mask_list, seg_id, max_len):
        tokens = []
        segment_ids = []
        gen_label_ids = []
        for idx, token in enumerate(part):
            if mask_list[idx] == 1.0:
                tokens.append('[MASK]')
                gen_label_ids.append(tokenizer.vocab[token])
            else:
                tokens.append(token)
                gen_label_ids.append(-1)
            # always supply gen label even if not masked, previously produced worse results
            #gen_label_ids.append(tokenizer.vocab[token])
            segment_ids.append(seg_id)
            if len(tokens) == max_len:  # save space for [SEP]
                LOGGER.debug("Can't encode the maximum Part A length of example number %s",
                             example_index)
                self.violate_max_part_a_len += 1
                break
        return tokens, segment_ids, gen_label_ids



