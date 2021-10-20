#   File:     datasets_meta.py
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
Handles the user of multiple datasets.

Possible heads:
Depends on the datasets registered in the handler
"""
import logging

import numpy as np

import torch
from torch.utils.data import TensorDataset

from .datasets_factory import DATASET_CLASS

LOGGER = logging.getLogger(__name__)


class MetaHandler():
    """
    For the moment, we assume only one shared LM head across any datasets.
    All token and sequence classification heads are separate.
    The correct numbers need to be set in the external script
    (could later modify so that the correct number of heads are inferred)

    """
    def __init__(self, biqe_args):
        self.features = []
        self.examples = []
        self.num_labels_cls = []
        self.num_labels_tok = []
        self.dataset_handlers = []
        import copy
        self.biqe_args = copy.deepcopy(biqe_args)
        self.train_files = self.biqe_args.train_file
        self.predict_files = self.biqe_args.predict_file
        self.valid_golds = self.biqe_args.valid_gold
        for i, dataset in enumerate(biqe_args.data_set):
            self.biqe_args.train_file = self.train_files[i]
            self.biqe_args.predict_file = self.predict_files[i]
            self.biqe_args.valid_gold = self.valid_golds[i]
            new_dataset = DATASET_CLASS[dataset](self.biqe_args)
            self.dataset_handlers.append(new_dataset)
            self.num_labels_cls += new_dataset.num_labels_cls
            self.num_labels_tok += new_dataset.num_labels_tok
        self.map_to_datasets = []
        self.index_to_map = 0
        self._set_up_functions(0)  # set up all relevant function to the first datahandler

    def read_examples(self, is_training=False):
        for i, _ in enumerate(self.dataset_handlers):
            self.dataset_handlers[i].read_examples(is_training)
            self.map_to_datasets.append(len(self.examples))
            self.examples += self.dataset_handlers[i].examples

    def _set_up_functions(self, index):
        self.truncate_end = self.dataset_handlers[index].truncate_end
        self.get_token_classification_ids = \
            self.dataset_handlers[index].get_token_classification_ids
        self.possible_mask_locations = \
            self.dataset_handlers[index].possible_mask_locations

    def update_info(self, index):
        if self.index_to_map < len(self.map_to_datasets):  # else we are on the last dataset handler
            if index == self.map_to_datasets[self.index_to_map]:
                self._set_up_functions(self.index_to_map)
                self.index_to_map += 1

    def get_dataset_handler(self, index):
        """
        Given an index, figure out to which dataset handler the example belongs to.

        :param index: an index pointing to an element in either self.examples or self.features
        :return: correct element of self.dataset_handlers so that the dataset hanlder that belongs
                to the example is returned.
        """
        # use self.map_to_datasets[]
        raise NotImplementedError

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

        # Assemble sequence classification ids
        # each example and its feature has to have the full range of heads but if a head is not
        # applicable for a dataset, then it write -1 in its place
        arranged_classify_id_cls = [[] for _ in range(len(self.num_labels_cls))]
        local_index_to_map = 0
        for index, f in enumerate(self.features):
            if local_index_to_map < len(self.map_to_datasets):  # else we are on the last dataset handler
                if index == self.map_to_datasets[local_index_to_map]:
                    local_index_to_map += 1
            for inner_index in range(len(self.num_labels_cls)):
                if inner_index == (local_index_to_map - 1):
                    arranged_classify_id_cls[inner_index] += f.classify_id_cls
                else:
                    arranged_classify_id_cls[inner_index] += [-1]
        # need to transform into example centric view
        arranged_classify_id_cls = np.array(arranged_classify_id_cls)
        arranged_classify_id_cls = np.swapaxes(arranged_classify_id_cls, 0, 1)
        all_classify_ids_cls = torch.tensor(arranged_classify_id_cls)

        # Assemble token classification ids
        # each example and its feature has to have the full range of heads but if a head is not
        # applicable for a dataset, then it write -1 in its place
        arranged_classify_ids_tokens = [[] for _ in range(len(self.num_labels_tok))]
        local_index_to_map = 0
        for index, f in enumerate(self.features):
            if local_index_to_map < len(self.map_to_datasets):  # else we are on the last dataset handler
                if index == self.map_to_datasets[local_index_to_map]:
                    local_index_to_map += 1
            for inner_index in range(len(self.num_labels_tok)):
                if inner_index == (local_index_to_map - 1):
                    arranged_classify_ids_tokens[inner_index] += f.classify_id_tokens
                else:
                    arranged_classify_ids_tokens[inner_index] += [[-1] * len(f.input_ids)]
        # need to transform into example centric view
        arranged_classify_ids_tokens = np.array(arranged_classify_ids_tokens)
        arranged_classify_ids_tokens = np.swapaxes(arranged_classify_ids_tokens, 0, 1)
        all_classify_ids_tokens = torch.tensor(arranged_classify_ids_tokens)

        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        #TODO find out how we can control the order of TensorDataset
        #is this possible? or do we need to implement a custom class? If so, this class should
        #inherit from TensorDataset.
        data_set = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_gen_label_ids,
                                 all_classify_ids_cls, all_classify_ids_tokens, all_example_index)
        return data_set
