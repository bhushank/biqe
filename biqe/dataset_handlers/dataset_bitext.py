# coding=utf-8


#   File:     dataset_bitext.py
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
Handles any bitext, where input (Part A) and output (Part B) are separated by a tab.

Possible heads:
plus_generation=1
"""
import logging
import re
import subprocess
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

from nsp.util import write_list_to_file, write_json_to_file, read_lines_in_list
from nsp.evals.f1_word_match import evaluate as evaluate_f1_word_match

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

LOGGER = logging.getLogger(__name__)


class GenExample():
    """
    This class encodes the bare minimum an instance needs to specify for a BERT model to run on it.
    """
    def __init__(self):
        """
        A single set of data. Subclasses should overwrite as appropriate.
        :param classify_id_cls: the id index that is the gold target for the prediction on [CLS]
        (pass None if not applicable, if applicable, needs to be in a list, in case there are
        more than one CLS heads)
        :param part_a: text string for Part A
        :param part_b: text string for Part B
        """
        self.classify_id_cls = [-1]
        self.part_a = None
        self.part_b = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        collect_string = ""
        collect_string += "x: %s" % self.part_a
        collect_string += ", y: %s" % self.part_b
        return collect_string


class BitextHandler(object):
    """
    Base class for bitext data sets.
    Other classes should inherit from src.is.

    A subclass should have the following variables
    examples: a list of examples, example should be dataset specific, for a very generic version see
                :py:class:GenExample
    features: a list of features, where a feature at index i maps to the example at index i
                in examples list
                See :py:class:GenInputFeature for an example.
    write_predictions: How to write predictions, either write_list_to_file or write_json_to_file
    write_eval: How to write evaluations, either write_list_to_file or write_json_to_file

    For sequence classification:
    num_labels_cls: Number of labels for the [CLS] token. Should be given as a list, where each
                    element corresponds to the number of classification classes for one [CLS]
                    classification head. (In most cases there will only be None then set
                    variable to None, or there will be one in which case the list only has one
                    entry.
    num_labels_tok: Number of labels for token classification. Should be given as a list.
                    Concept is the same as for num_labels_cls above.
    text2id: list of dictionaries that maps text to ids for CLS prediction
    id2text: list of dictionaries that maps ids to text for CLS prediction

    For token classification:
    num_labels_tok: Number of labels for token classification
    text2id_tok: dictionary that maps text to ids for token prediction
    id2text_tok: dictionary that maps ids to text for token prediction

    For a class implementing all aspects (generation, classify tokens and classify on [CLS],
    see dataset_sharc.py)
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, nsp_args):
        """

        examples: a list of examples of type :py:class:BitextExample
        features: a list of features of type :py:class:GenInputFeature
        """
        self.examples = []
        self.features = []
        self.write_predictions = write_list_to_file
        self.write_tok_predictions = self.write_predictions
        self.write_cls_predictions = self.write_predictions
        self.write_eval = write_json_to_file
        self.num_labels_cls = []
        self.num_labels_tok = []
        self.text2id = None
        self.id2text = None
        self.text2id_tok = None
        self.id2text_tok = None
        # if True, convert_examples_to_features in masking.py will truncate the end
        # if it exceeds max_part_a
        self.truncate_end = True
        self.train_file = nsp_args.train_file
        self.predict_file = nsp_args.predict_file
        self.valid_gold = nsp_args.valid_gold
        self.train_dataloader = None
        self.eval_dataloader = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__

    def read_examples(self, is_training=False):
        """
        Reads a bitext that is separated by a tab, e.g. word1 word2 \t word3 word4
        Everything before the tab will become Part A, rest Part B
        :param input_file: the file containing the tab-separated data for training,
        and just Part A for predict
        :param is_training: True for training, then we expect \t, else we do not.
        :return: 0 on success
        """
        if is_training is True:
            input_file = self.train_file
        else:
            input_file = self.predict_file
        self.examples = []  # reset previous lot
        LOGGER.info("Part a: prior to tab")
        LOGGER.info("Part b: post tab")
        input_data = read_lines_in_list(input_file)

        example_counter = 0
        for entry in input_data:
            part_a = entry
            part_b = ""
            if is_training is True:
                split_line = entry.split("\t")
                assert len(split_line) == 2
                part_a = split_line[0]
                part_b = split_line[1]

            example = BitextHandler.BitextExample(
                example_index=example_counter,
                part_a=part_a,
                part_b=part_b)
            self.examples.append(example)
            example_counter += 1
        return 0

    # pylint: disable=no-self-use
    def get_token_classification_ids(self, current_example, input_ids, tokenizer):
        """
        For BERT training, this function sets the gold IDs sequence for token classification.
        We cannot set this prior to calling masking, because the dataset handlers do not know about
        tokenization. So we call this function within masker witht he input_ids of the entire
        sequence and the tokenizer. Then we let each dataset handler decide on which tokens
        which label should be classified.

        On the Sharc dataset this equals to setting the classification label
        (Yes, No, More, Irrelevant),
        for every token in the rule text

        :param current_example: an instance of py:class:BitextHandler.BitextExample
        :param input_ids: the masker prepared input ids for this example
        :param tokenizer: the tokenizer to use
        :return: a list of lists, where each inner list is of length max_seq_length with
                the classification label set to -1.
                The outer list respects the fact that there could be several heads.
        """
        del current_example
        del tokenizer
        #the outer list respects that there could be more than 1 token classification head.
        classify_id_tokens = [[-1] * len(input_ids)]
        return classify_id_tokens

    # pylint: disable=no-self-use
    def arrange_generated_output(self, current_example, generated_text):
        """
        Simply returns generated_text, other data sets can arrange the output
        ready for evaluation here.
        :param current_example: The current example
        :param generated_text: The text generated by the model
        :return: generated_text
        """
        del current_example
        return generated_text

    # pylint: disable=no-self-use
    def arrange_classify_output(self, current_example, max_classify_index):
        """
        Simply returns the classification index, other data sets can arrange
        the output ready for evaluation here.
        :param current_example: The current example
        :param max_classify_index: the index of the most likely class,
            can be converted back into a string via id2text[max_classify_index]
        :return: the most likely class
        """
        del current_example
        return max_classify_index

    # pylint: disable=no-self-use
    def arrange_token_classify_output(self, current_example, classification_tokens, input_ids,
                                      tokenizer):
        """
        Simply returns all classification elements, other data sets can arrange the output
        as needed here.
        :param current_example: The current example
        :param classification_tokens: the classification labels for all tokens
        :param input_ids: the input_ids as they where given to the BERT model
        :param tokenizer: the BERT tokenizer used
        :return: classification_tokens
        """
        del current_example
        del classification_tokens
        del input_ids
        del tokenizer
        return classification_tokens

    # pylint: disable=no-self-use
    def combine_classification_tokens(self, classification_tokens):
        """
        Given a list of classified tokens, we can combine them as required by a specific data set.
        Here we just leave the list untouched.
        :param classification_tokens: a list of classified tokens
        :return: the same list
        """
        return classification_tokens

    def evaluate(self, output_prediction_file, mode, tokenizer=None):
        """
        Given the location of the prediction and gold output file,
        calls a dataset specific evaluation script.
        Here it calls case-sensitive BLEU script and F1 word match.
        :param output_prediction_file: the file location of the predictions
        :param mode: possible values: generation, cls, token as called from src.edict.py;
        bitext handles all the same
        :return: a dictionary with various statistics
        """
        def _convert_to_float(convert):
            try:
                convert = float(convert)
            except OverflowError:
                convert = 0.0
            return convert

        # BLEU
        with open(output_prediction_file, "r") as file:
            eval_process = \
                subprocess.Popen([DIR_PATH+"/../evals/multi-bleu.perl", "-lc", self.valid_gold],
                                 stdin=file, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = eval_process.communicate()
        #format example:
        # BLEU = 26.27, 59.8/38.8/32.5/27.1 (BP=0.695, ratio=0.733, hyp_len=4933, ref_len=6729)
        bleu_all = stdout.decode("utf-8")
        if bleu_all.startswith("Illegal division"):
            results = {mode+"_"+'moses_bleu': 0.0, mode+"_"+'moses_bleu_1': 0.0,
                       mode+"_"+'moses_bleu_2': 0.0, mode+"_"+'moses_bleu_3': 0.0,
                       mode+"_"+'moses_bleu_4': 0.0}
        else:
            bleu = 0.0
            try:
                bleu = float(re.compile('BLEU = (.*?),').findall(bleu_all)[0])
            except (OverflowError, IndexError):  # if all translations are the empty string
                pass

            bleu_all = re.sub(r".*?, ", '', bleu_all, 1)
            bleu_all = re.sub(r" .BP.*\n", '', bleu_all)
            #format now: 159.8/38.8/32.5/27.1
            bleu_all = bleu_all.split("/")
            try:
                results = {mode+"_"+'moses_bleu': bleu,
                           mode+"_"+'moses_bleu_1': _convert_to_float(bleu_all[0]),
                           mode+"_"+'moses_bleu_2': _convert_to_float(bleu_all[1]),
                           mode+"_"+'moses_bleu_3': _convert_to_float(bleu_all[2]),
                           mode+"_"+'moses_bleu_4': _convert_to_float(bleu_all[3])}
            except OverflowError:
                results = {mode+"_"+'moses_bleu': 0.0, mode+"_"+'moses_bleu_1': 0.0,
                           mode+"_"+'moses_bleu_2': 0.0, mode+"_"+'moses_bleu_3': 0.0,
                           mode+"_"+'moses_bleu_4': 0.0}

        # F1 word match
        f1_results = self.get_f1(output_prediction_file)
        for key in f1_results:
            results[mode+"_"+key] = f1_results[key]
        return results

    def get_f1(self, output_prediction_file):
        """
        Given an output prediction file and a true gold file,
        calculate f1 word overlap scores.
        :param output_prediction_file: the output prediction file
        :return: the F1 scores (dictionary with several different values)
        """
        hyps = read_lines_in_list(output_prediction_file)
        golds = read_lines_in_list(self.valid_gold)
        f1_results = evaluate_f1_word_match(golds, hyps)
        return f1_results


    def select_deciding_score(self, results):
        """
        Returns the score that should be used to decide whether or not
        a model is best compared to a previous score.
        Here we return BLEU-4
        :param results: what is returned by the method evaluate,
        a dictionary that should contain 'bleu_4'
        :return: BLEU-4 value
        """
        return results['generation_moses_bleu_4']

    def possible_mask_locations(self, part_a, part_b, is_training):
        """
        Given an example's part_a and part_b, decide which positions may be masked.
        By default, this class allows any position to be masked.
        Subclasses may overwrite this functionality.
        :param part_a: taken from part_a from a subclass instance of :py:class:GenExample,
                       but it is potentially tokenised
        :param part_b: taken from part_b from a subclass instance of :py:class:GenExample,
                       but it is potentially tokenised
        :param is_training: True if training, else False
        :return: a tuple of two lists, one for part_a and one for part_b. Each list is
                of length part_a/part_b, where value of 1.0 indicates that this
                position may be masked and 0.0 indicates that it should not be masked.
        """
        possible_mask_locations_a = [1.0] * len(part_a)
        possible_mask_locations_b = [1.0] * len(part_b)
        return possible_mask_locations_a, possible_mask_locations_b

    def update_info(self, i):
        """
        This function is used by MetaHandler in datasets_meta.py.
        For all other datasets we just do nothing here.
        :param i: the ith example being processed of self.examples
        """
        pass

    def create_tensor_dataset(self):
        """
        Using a data_handler, whose features have been filled via the function
        convert_examples_to_features from a subclass instance of :py:class:Masking,
        convert the features into a TensorDataset
        :param data_handler: instance or subclass instance of :py:class:Bitext
        :return: the features represented as a TensorDataset
        """
        # pylint: disable=not-callable
        # pylint: disable=no-member
        all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.features],
                                      dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.features],
                                       dtype=torch.long)
        all_gen_label_ids = torch.tensor([f.gen_label_ids for f in self.features],
                                         dtype=torch.long)
        all_classify_ids_cls = torch.tensor([f.classify_id_cls for f in self.features])
        all_loss_masks = torch.tensor(
            [f.loss_mask for f in self.features])
        all_position_ids = torch.tensor(
            [f.position_ids for f in self.features])
        all_classify_ids_tokens = torch.tensor(
            [f.classify_id_tokens for f in self.features])
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        data_set = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_position_ids, all_loss_masks, all_gen_label_ids,
                                 all_classify_ids_cls, all_classify_ids_tokens, all_example_index)
        return data_set


    def token_classification_prediction(self,example, logits, current_input_ids, tokenizer):
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

        classified_tokens_indices = predict_token_classification(logits)
        return self.arrange_token_classify_output(example, classified_tokens_indices,
                        current_input_ids, tokenizer)

    class BitextExample(GenExample):
        """A single training/test example from src.Bitext.
        """
        # pylint: disable=too-few-public-methods
        def __init__(self,
                     example_index,
                     part_a,
                     part_b,
                     cls_gold=None):
            super().__init__()
            self.example_index = example_index
            self.part_a = part_a
            self.part_b = part_b
            if cls_gold is not None:
                self.cls_gold = cls_gold
            else:
                self.cls_gold = [-1]
