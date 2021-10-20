#   File:     evaluator_biqe.py
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


import json, pickle
import numpy as np
from collections import defaultdict, OrderedDict
import os
import sklearn.metrics
import torch

def eval_q2b(prediction_file):
    with open(prediction_file, encoding='utf8') as f:
        preds = json.load(f)
    query_metrics = defaultdict(list)
    for ex in preds:
        query_metrics[ex['q_type']].append(ex['hits3'])
    metrics = []
    for query in query_metrics:
        if 'union' not in query:
            metrics.append(np.mean(query_metrics[query]))
        print(f"{query}: {np.mean(query_metrics[query])}")
    return {'hits3':np.mean(metrics)}


def eval_bio(prediction_file):
    with open(prediction_file, encoding='utf8') as f:
        preds = json.load(f)
    y_pos, y_neg =[],[]
    results_dict = defaultdict(list)
    for ex in preds:
        logits = ex['logits']
        results_dict[ex['q_type']].append((float(logits[0]),float(logits[1])))
    metrics = []
    for q_type in results_dict:
        pos,neg = zip(*results_dict[q_type])
        pos = list(pos)
        neg = list(neg)
        y_scores = pos+neg
        y_true = [1]*len(pos) + [0]*len(neg)
        auc = sklearn.metrics.roc_auc_score(y_true, y_scores)
        metrics.append(auc)
    return {'auc':np.mean(metrics)}

def sort_filters_ent(gold_dict):
    tuples = []
    for key in gold_dict:
        if key=='tail':
            new_key = 0
        elif key=='inter':
            new_key = 1
        else:
            br_num,pos = key.split('-')
            new_key = (int(br_num)+1)*10 + int(pos)
        tuples.append((new_key,gold_dict[key]))
    sorted_tuples = sorted(tuples)
    _, vals = zip(*sorted_tuples)
    return list(vals)

def eval_cq(prediction_file, gold_file):
    '''

    :param prediction_file:
    :param gold_file:
    :param mode:
    :return:
    '''

    with open(prediction_file, encoding='utf8') as f:
        preds = json.load(f)
    with open(gold_file, encoding='utf8') as f:
        filters = json.load(f)
    ranks, results = [], dict()
    errors = 0
    grouped_ranks = defaultdict(list)
    for count,ex in enumerate(preds):
        filter_ents = sort_filters_ent(filters[count])
        if len(ex['gold']) != len(ex['ranks']):
            print(f"Warning, length mismatch {count}")
            errors += 1
            continue
        for mask_count in ex['ranks']:
            ents = ex['ranks'][mask_count]
            filter_set = filter_ents[int(mask_count)]
            if ex['gold'][int(mask_count)] not in filter_set:
                print(f"Warning, incorrect filter, Number {count}")
                errors+=1
            #assert arr_labels[int(mask_count)] in filter_set
            diff = set(ents).difference(filter_set)
            rank = len(diff)+1
            ranks.append(rank)
            if int(mask_count)==0:
                new_key = 'tail'
            elif int(mask_count)==1:
                new_key = 'inter'
            else:
                new_key = 'others'
            grouped_ranks[new_key].append(rank)
    for key in grouped_ranks:
        mrr = compute_mrr(grouped_ranks[key])
        h10 = compute_h10(grouped_ranks[key])
        grouped_ranks[key] = [mrr,h10]
    with open(os.path.join('/home/kotnis/data/fb15k-237/DAGs/grouped_results.json'),'w',encoding='utf8') as f:
        json.dump(grouped_ranks,f, indent=None, separators=(',\n',': '))
    mrr = compute_mrr(ranks)
    h10 = compute_h10(ranks)
    results['mrr'] = mrr
    results['h10'] = h10
    print(f"Errors {errors}")
    return results

def evaluate_paths(prediction_file, gold_file):
    '''

    :param prediction_file:
    :param gold_file:
    :param mode:
    :return:
    '''

    with open(prediction_file, encoding='utf8') as f:
        preds = json.load(f)
    with open(gold_file, encoding='utf8') as f:
        gold = json.load(f)
    ranks, results = [], dict()
    for count,ex in enumerate(preds):
        gold_ents = gold[count]
        correct = ex['gold']
        for ind,e in enumerate(ex['ranks']):
            ents = set(e)
            gold_set = gold_ents[ind]
            assert correct[ind] in gold_set
            diff = ents.difference(gold_set)
            rank = len(diff)+1
            ranks.append(rank)
    mrr = compute_mrr(ranks)
    h10 = compute_h10(ranks)
    results['mrr'] = mrr
    results['h10'] = h10
    return results



def compute_mrr(ranks):
    return np.mean([1/r for r in ranks])

def compute_h10(ranks):
    'ranks start from 1 therefore <='
    def hit10(x):
        return x<=10
    return np.mean([hit10(r) for r in ranks])
