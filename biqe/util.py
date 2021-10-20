# encoding=utf-8

#   File:     util.py
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
Provides some utility functions.
"""

import codecs
import json
import numpy as np
import pickle, torch
from collections import defaultdict
from biqe.bio_graph import Query, Graph

def write_list_to_file(list_to_write, file_to_write):
    """
    Write a list to a file.
    :param list_to_write: the list to be written to a file
    :param file_to_write: the file to write to
    :return: 0 on success
    """
    with codecs.open(file_to_write, 'w', encoding='utf8') as file:
        for line in list_to_write:
            print(line, file=file)
    return 0


def write_json_to_file(json_object, file_to_write):
    """
    Write a json object to a file.
    :param json: the json object to write
    :param file_to_write: the location to write to
    :return: 0 on success
    """
    with open(file_to_write, "w") as writer:
        writer.write(json.dumps(json_object, indent=4) + "\n")
    return 0


def write_list_to_jsonl_file(list_to_write, file_to_write):
    """
    Write a list to a file.
    :param list_to_write: the list to be written to a file
    :param file_to_write: the file to write to
    :return: 0 on success
    """
    with codecs.open(file_to_write, 'w', encoding='utf8') as file:
        for line in list_to_write:
            file.write(json.dumps(line) + "\n")
    return 0


def read_lines_in_list(file_to_read):
    """
    Reads a file into a list.
    :param file_to_read: the location of the file to be read
    :return: a list where each entry corresponds to a line in the file
    """
    read_list = []
    with codecs.open(file_to_read, 'r', encoding='utf8') as file:
        for line in file:
            read_list.append(line.rstrip('\n'))
    return read_list


def read_json(json_to_read):
    """
    Read a json file
    :param json_to_read: the json object to read
    :return: the json object
    """
    with open(json_to_read, "r") as reader:
        json_object = json.loads(reader)
    return json_object


def sublist_start_index(search_for_this, find_here):
    """
    Given a list a, checks if the entire list is contain in sequential order in b.
    If so, return the first index in b where a starts.
    :param search_for_this: the list to be found in find_here
    :param find_here: the list searched for search_for_this
    :return: If a in b, then start index in b, else None
    """

    if len(search_for_this)<=0 or (len(search_for_this) > len(find_here)) or (search_for_this[0] not in find_here):
        return None
    for i in range(find_here.index(search_for_this[0]), len(find_here) - len(search_for_this) + 1):
        if find_here[i:i + len(search_for_this)] == search_for_this:
            return i
    return None


def compute_softmax(scores, alpha=1.0):
    """
    Computes softmax probaility over raw logits
    :param scores: a numpy array with logits
    :param alpha: temperature parameter, values >1.0 approach argmax, values <1.0 approach uniform
    :return: a numpy array with probability scores
    """
    scores = scores * float(alpha)
    scores = scores - np.max(scores)
    scores = np.exp(scores)
    probs = scores / np.sum(scores)

    return probs


def process_verb(sent, verb):
    """
    Processes verbs.

    :param sent: a sentence
    :param verb: a verb
    :return: the verb
    """
    verb_map = {'be': [","], 'has': ["'s", "'"], 'is': [",", "'", "is"], 'is in': ["is in", ","]}
    if verb in verb_map:
        for instance in verb_map[verb]:
            if instance in sent:
                verb = instance
                break
    else:
        toks = verb.split(' ')
        toks = [t for t in toks if sent.find(t) >= 0]
        if len(toks) > 0:
            verb = " ".join(toks)
    return verb


def bert_sw(bert_tokens):
    """
    bert_sw

    :param bert_tokens:
    :return: tokens
    """
    stack = []
    tokens = []
    for token in bert_tokens:
        if token.startswith('#'):
            stack.append(token.replace('#', ''))
        else:
            if len(stack) > 0:
                prev_word = [x for x in stack]
                tokens.append(prev_word)
                stack.clear()
                stack.append(token)
            else:
                stack.append(token)
    if len(stack) > 0:
        prev_word = [x for x in stack]
        tokens.append(prev_word)
    tokens = ["".join(t) for t in tokens]
    return tokens



def load_queries(data_file, keep_graph=False):
    raw_info = pickle.load(open(data_file, "rb"))
    return [Query.deserialize(info, keep_graph=keep_graph) for info in raw_info]

def load_queries_by_formula(data_file):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = defaultdict(lambda : defaultdict(list))
    for raw_query in raw_info:
        query = Query.deserialize(raw_query)
        queries[query.formula.query_type][query.formula].append(query)
    return queries

def load_test_queries_by_formula(data_file):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = {"full_neg" : defaultdict(lambda : defaultdict(list)),
            "one_neg" : defaultdict(lambda : defaultdict(list))}
    for raw_query in raw_info:
        neg_type = "full_neg" if len(raw_query[1]) > 1 else "one_neg"
        query = Query.deserialize(raw_query)
        queries[neg_type][query.formula.query_type][query.formula].append(query)
    return queries

def load_graph(data_dir, embed_dim):
    rels, adj_lists, node_maps = pickle.load(open(data_dir+"/graph_data.pkl", "rb"))
    node_maps = {m : {n : i for i, n in enumerate(id_list)} for m, id_list in node_maps.items()}
    for m in node_maps:
        node_maps[m][-1] = -1
    feature_dims = {m : embed_dim for m in rels}
    feature_modules = {m : torch.nn.Embedding(len(node_maps[m])+1, embed_dim) for m in rels}
    for mode in rels:
        feature_modules[mode].weight.data.normal_(0, 1./embed_dim)
    features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor([node_maps[mode][n] for n in nodes])+1))
    graph = Graph(features, feature_dims, rels, adj_lists)
    return graph, feature_modules, node_maps