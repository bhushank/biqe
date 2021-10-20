# coding=utf-8

#   File:     dataset_biqe.py
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

import logging
from biqe.util import write_json_to_file
from biqe.dataset_handlers.dataset_bitext import BitextHandler, GenExample
from biqe.evals.evaluator_biqe import eval_bio
import numpy as np
from tqdm import tqdm
import json, pickle
from collections import defaultdict
import torch
from torch.utils.data import TensorDataset
LOGGER = logging.getLogger(__name__)


class Biqe(BitextHandler):
    def __init__(self, biqe_args=None):
        super().__init__(biqe_args)

        self.examples = []
        self.features = []
        index_file = biqe_args.token_index_path
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        self.num_ents = len(index)
        self.text2id_tok = [index]
        self.id2text_tok = [{v: k for k, v in index.items()}]
        self.num_labels_tok = [len(index)]
        self.write_predictions = write_json_to_file
        self.write_eval = write_json_to_file
        self.write_tok_predictions = write_json_to_file

    def read_examples(self, is_training=False):
        """
                Reads the MMKG FB15k dataset, each entry in self.examples holds a
                :py:class:MMKG_FB15kExample object
                :param input_file: the file containing  MMKG FB15k dataset (-#- has been changed to space)
                :param is_training: True for training, then we read in gold labels, else we do not.
                :return: 0 on success
                """
        if is_training is True:
            input_file = self.train_file
        else:
            input_file = self.predict_file
        self.examples = []  # reset previous lot
        example_counter = 0
        with open(input_file, "r", encoding='utf-8') as reader:
            data = json.load(reader)
        num_gold = []
        for ex in tqdm(data, desc='Reading File'):
            '''
            np.random.shuffle(ex)
            parts = ex[0].split('\t')
            path, gold_targets = parts[0], [self.text2id_tok[0][x] for x in parts[1:]]
            path_str = [path]
            elems = set(path.split('-#-'))
            if len(ex)>1:
                for count in range(1,len(ex)):
                    parts = ex[count].split('\t')[0]
                    parts = [e for e in parts.split('-#-') if e not in elems]
                    elems.update(set(parts))
                    path_str.append('-#-'.join(parts))
            path = '-#-'.join(path_str)
            '''
            path_str, gold_targets = [], []
            q_type = 'path' if len(ex) <= 1 else 'DAG'
            num_masks = 0
            for p in ex:
                parts = p.split('\t')
                path_str.append(parts[0])
                gold_targets.extend([self.text2id_tok[0][x] for x in parts[1:]])
                path = parts[0].split('-#-')
                num_masks += len([1 for x in path if x == '[MASK]'])
            # assert num_masks==len(gold_targets)
            path = '-#-[SEP]-#-'.join(path_str)
            example = Biqe.Biqe_Example(
                example_index=example_counter,
                path=path,
                gold_targets=gold_targets,
                q_type=q_type)
            self.examples.append(example)
            num_gold.append(len(gold_targets))
            example_counter += 1
            # if example_counter>=10:# and is_training:
            # break
        # print(f"Mean num of Ans. {np.mean(num_gold)}, Max. Num Ans {np.max(num_gold)}")
        # exit(0)
        return 0


    def get_token_classification_ids(self, example, input_ids, tokenizer):
        '''
        This method maps tokens from vocabulory to the tokens in the output softmax.
        all relations are mapped to relation token R
        :param example:
        :param input_ids:
        :param tokenizer:
        :return:
        '''
        num_gold = 2
        #np.random.shuffle(example.negs)
        classify_id_tokens = []
        for i in range(num_gold):
            tokens = [-1 for x in range(len(input_ids))]
            classify_id_tokens.append(tokens)
        inp_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        for count,token in enumerate(inp_tokens):
            if token=='[MASK]':
                #for ind,t in enumerate(gold_targets):
                classify_id_tokens[0][count] = example.gold_targets[0]
                classify_id_tokens[1][count] = example.negatives[0]

        return classify_id_tokens
    #for CQ Datasets
    def token_classification_prediction(self, current_example, logits, current_input_ids, tokenizer):
        all_scores, all_ranks = defaultdict(list), defaultdict(list)
        correct_ids = current_example.gold_targets[:]
        # correct_ents = [self.id2text_tok[0][x] for x in correct_ids]
        assert logits.shape[0] == len(current_input_ids)
        two_counter, mask_counter = 0, 2
        label_acc = defaultdict(list)
        for i, _ in enumerate(logits):
            if current_input_ids[i] == 1:
                scores = logits[i] * -1
                if two_counter < 2:
                    all_scores[two_counter].append(scores)
                    label_acc[two_counter].append(correct_ids.pop(0))
                    two_counter += 1
                else:
                    all_scores[mask_counter].append(scores)
                    label_acc[mask_counter].append(correct_ids.pop(0))
                    mask_counter += 1
            elif current_input_ids[i] == 3:  # [SEP] token
                two_counter = 0
        labels = []
        for count, mask_count in enumerate(all_scores):
            scores = np.mean(all_scores[mask_count], axis=0)
            # scores = all_scores[mask_count][0]
            sorted_args = np.argsort(scores)
            filt_ids = []
            label = label_acc[mask_count][0]
            labels.append(label)
            for id in sorted_args:
                if id == label:
                    gold_score = scores[id]
                    equiv_ids = self.find_equivalent(id, gold_score, scores)
                    fil_id_set = set(filt_ids)
                    for i in equiv_ids:
                        ent = self.id2text_tok[0][i]
                        if ent not in fil_id_set:
                            filt_ids.append(self.id2text_tok[0][i])
                    break
                filt_ids.append(self.id2text_tok[0][id])
            all_ranks[mask_count] = filt_ids
        assert len(all_ranks) == len(labels)
        correct_ents = [self.id2text_tok[0][x] for x in labels]
        return {"id": current_example.example_index, "ranks": all_ranks, "gold": correct_ents}


    def find_equivalent(self,gold_id, gold_score, scores):
        ids = set()
        for i in range(scores.shape[0]):
            if gold_score==scores[i]:
                ids.add(i)
        ids.remove(gold_id)
        return ids
    # for CQ datasets
    def token_classification_prediction_cq(self, current_example, logits, current_input_ids, tokenizer):
        all_scores, all_ranks = defaultdict(list), defaultdict(list)
        correct_ids = current_example.gold_targets[:]
        #correct_ents = [self.id2text_tok[0][x] for x in correct_ids]
        assert logits.shape[0]==len(current_input_ids)
        two_counter, mask_counter = 0,2
        label_acc = defaultdict(list)
        for i,_ in enumerate(logits):
            if current_input_ids[i]==1:
                scores = logits[i]*-1
                if two_counter<2:
                    all_scores[two_counter].append(scores)
                    label_acc[two_counter].append(correct_ids.pop(0))
                    two_counter+=1
                else:
                    all_scores[mask_counter].append(scores)
                    label_acc[mask_counter].append(correct_ids.pop(0))
                    mask_counter+=1
            elif current_input_ids[i]==3: # [SEP] token
                two_counter=0
        labels = []
        for count,mask_count in enumerate(all_scores):
            scores = np.mean(all_scores[mask_count], axis=0)
            #scores = all_scores[mask_count][0]
            sorted_args = np.argsort(scores)
            filt_ids = []
            label = label_acc[mask_count][0]
            labels.append(label)
            for id in sorted_args:
                if id == label:
                    gold_score = scores[id]
                    equiv_ids = self.find_equivalent(id,gold_score, scores)
                    fil_id_set = set(filt_ids)
                    for i in equiv_ids:
                        ent = self.id2text_tok[0][i]
                        if ent not in fil_id_set:
                            filt_ids.append(self.id2text_tok[0][i])
                    break
                filt_ids.append(self.id2text_tok[0][id])
            all_ranks[mask_count] = filt_ids
        assert len(all_ranks)==len(labels)
        correct_ents = [self.id2text_tok[0][x] for x in labels]
        return {"id":current_example.example_index,"ranks":all_ranks, "gold": correct_ents}

    def evaluate(self, output_prediction_file, valid_gold, tokenizer=None):
        res = kg_evaluate(output_prediction_file, valid_gold)
        return res

    def select_deciding_score(self, results):
        """
        Returns the score that should be used to decide whether or not
        a model is best compared to a previous score.

        :param results: what is returned by the method evaluate,
        a dictionary that should contain 'bleu_4'
        :return: mrr value
        """
        return float(results['auc'])


    def q2b_eval(self, score, count):
        nentity = len(self.text2id_tok[0])
        query = self.queries[count][0]
        score -= (np.min(score) - 1)
        ans = self.test_ans[query]
        hard_ans = self.test_ans_hard[query]
        all_idx = set(range(nentity))
        false_ans = all_idx - ans
        hard_ans_list = list(hard_ans)
        false_ans_list = list(false_ans)
        ans_idxs = np.array(hard_ans_list)
        vals = np.zeros((len(ans_idxs), nentity))
        vals[np.arange(len(ans_idxs)), ans_idxs] = 1
        axis2 = np.tile(false_ans_list, len(ans_idxs))
        axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))
        vals[axis1, axis2] = 1
        b = torch.from_numpy(vals).to(self.device)
        score = torch.from_numpy(score).to(self.device)
        filter_score = b * score
        argsort = torch.argsort(filter_score, dim=1, descending=True)
        ans_tensor = torch.LongTensor(hard_ans_list).to(self.device)
        argsort = torch.transpose(torch.transpose(argsort, 0, 1) - ans_tensor, 0, 1)
        ranking = (argsort == 0).nonzero()
        ranking = ranking[:, 1]
        ranking = ranking + 1
        hits3 = torch.mean((ranking <= 3).to(torch.float)).item()
        return hits3

    # for Q2B
    def get_token_classification_ids_q2b(self, example, input_ids, tokenizer):
        '''
        This method maps tokens from vocabulory to the tokens in the output softmax.
        all relations are mapped to relation token R
        :param example:
        :param input_ids:
        :param tokenizer:
        :return:
        '''
        num_gold = 5
        np.random.shuffle(example.gold_targets)
        gold_targets = example.gold_targets[:num_gold]
        classify_id_tokens = []
        for i in range(num_gold):
            tokens = [-1 for x in range(len(input_ids))]
            classify_id_tokens.append(tokens)
        inp_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        for count,token in enumerate(inp_tokens):
            if token=='[MASK]':
                for ind,t in enumerate(gold_targets):
                    classify_id_tokens[ind][count] = t
        return classify_id_tokens

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

    class Biqe_Example(GenExample):
        """A single training/test example for the MMKG FB15k corpus.
        """
        # pylint: disable=too-few-public-methods
        def __init__(self,
                     example_index,
                     path,
                     gold_targets,
                     q_type):
            super().__init__()
            self.example_index = example_index
            assert isinstance(path,str)
            assert isinstance(gold_targets, list)
            self.path = path
            self.gold_targets = gold_targets  # never use as part_a or part_b, only for evaluation!
            #self.negatives = negatives
            self.part_a = self.path
            self.part_b = ""
            self.q_type = q_type



