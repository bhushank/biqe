#   File:     arguments.py
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

import argparse
import logging

from .transformer_heads import MODEL_CLASSES

LOGGER = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser = __add_arguments(parser)
    args = parser.parse_args()
    return args




def __add_required_arguments(parser):
    return parser


def __add_arguments(parser):
    #required arguments
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                             help="The output directory where all relevant files will be "
                                  "written to.")
    parser.add_argument("--data_set", type=str, required=True, nargs='+',
                        help="Which dataset(s) to expect.\n Options are: %s" %
                             ", ".join(['biqe']))
    parser.add_argument("--model_type", default=None, type=str, required=True,
                            help="Which model type to usee: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: "
                             #+ ", ".join(ALL_MODELS)
                             + "\n Leave empty to initialize a new model")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, nargs='+',
                             help="Input file for training")
    parser.add_argument("--predict_file", default=None, type=str, nargs='+',
                             help="Input file for prediction.")
    parser.add_argument("--valid_gold", default=None, type=str, nargs='+',
                             help="Location of gold file for evaluating predictions.")
    parser.add_argument('--valid_every_epoch',
                             action='store_true',
                             help="Whether to validate on the validation set after every "
                                  "epoch, save best model according to the evaluation metric "
                                  "indicated by each specific dataset class.")
    parser.add_argument("--load_prev_model", default=None, type=str,
                             help="Provide a file location if a previous model should be "
                                  "loaded. (Note that Adam Optimizer paramters are lost.")
    parser.add_argument("--adam_schedule", default="warmup_linear", const="warmup_linear",
                             nargs='?', type=str,
                             choices=['warmup_linear', 'warmup_constant', 'warmup_cosine'],
                             help="Warm up schedule to use for Adam.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                             help="The maximum total sequence length (Part A + B) after "
                                  "tokenization.")
    parser.add_argument("--max_part_a", default=64, type=int,
                             help="The maximum number of tokens for Part A. Sequences longer "
                                  "than this will be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Should be true to run "
                                                                     "taining.")
    parser.add_argument("--do_predict", action='store_true',
                             help="Should be true to run predictition.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                             help="Total batch size for training. "
                                  "Actual batch size will be divided by "
                                  "gradient_accumulation_steps and clipped to closest int.")
    parser.add_argument("--predict_batch_size", default=8, type=int,
                             help="Batch size to use for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                             help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                             help="How many training epochs to run.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                             help="Proportion of training to perform linear learning rate "
                                  "warmup e.g., 0.1 = 10 percent of training.")
    parser.add_argument("--verbose_logging", action='store_true',
                             help="Log more information.")
    parser.add_argument("--no_cuda",
                             action='store_true',
                             help="Use CPU even if GPU is available.")
    parser.add_argument('--seed',
                             type=int,
                             default=42,
                             help="Random seed for initialization, "
                                  "set to -1 to draw a random number.")
    parser.add_argument('--gradient_accumulation_steps',
                             type=int,
                             default=1,
                             help="Number of updates steps to accumulate before performing "
                                  "a backward/update pass.")
    parser.add_argument("--do_lower_case",
                             action='store_true',
                             help="Whether to lower case the input text. "
                                  "Should be True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                             type=int,
                             default=-1,
                             help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                             action='store_true',
                             help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                             type=float, default=0,
                             help="Loss scaling to improve fp16 numeric stability. "
                                  "Only used when fp16 set to True.\n"
                                  "0 (default value): dynamic loss scaling.\n"
                                  "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                             help="Set the value above which gradients will be clipped.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    # pertraining BERT
    parser.add_argument("--vocab_size", type=int, default=30522,
                             help="The desired vocabulary size, only use in conjunction with "
                                  "bert-vanilla.")
    parser.add_argument("--no_basic_tok", type=bool,
                             help="If supplied, basic tokenization will not be performed "
                                  "(e.g. punctuation).")
    parser.add_argument("--output_hidden_states", type=bool,
                             help="If supplied, hidden states will be returned.")
    parser.add_argument("--output_attentions", type=bool,
                             help="If supplied, attention will be returned.")

    #pertaining masking
    parser.add_argument("--mask_in_a", type=bool,
                             help="If supplied, masking (as set by masking_strategy, "
                                  "distribution_mean and distribution_stdev) is applied to part a.")
    parser.add_argument("--mask_in_b", type=bool,
                             help="If supplied, masking (as set by masking_strategy, "
                                  "distribution_mean and distribution_stdev) is applied to part b.")
    parser.add_argument("--max_gen_b_length", default=0, type=int,
                             help="Maximum length for output generation sequence (Part A).")
    parser.add_argument("--max_gen_a_length", default=0, type=int,
                             help="Maximum length for output generation sequence (Part B).")
    parser.add_argument("--masking_strategy", default=None, type=str, const=None,
                             nargs='?',
                             choices=['bernoulli', 'gaussian', 'dataset_dependent'],
                             help="Which masking strategy to us, options are: "
                                  "bernoulli, gaussian, dataset_dependent")
    parser.add_argument("--distribution_mean", default=1.0, type=float,
                             help="The mean (for Bernoulli and Gaussian sampling).")
    parser.add_argument("--distribution_stdev", default=0.0, type=float,
                             help="The standard deviation (for Gaussian sampling).")

    #pertaining prediction
    parser.add_argument("--predict", type=str, default='one_step_greedy',
                             const='one_step_greedy',
                             nargs='?',
                             choices=['one_step_greedy', 'left2right', 'max_probability',
                                      'min_entropy', 'right2left', 'no_look_ahead'],
                             help="How perdiction should be run.")
    parser.add_argument("--top_k", type=int, default=-1,
                             help="Normalize over the top_k logits, If -1: normalize over all.")
    parser.add_argument("--return_attention",
                             action='store_true',
                             help="Get the attention probabilities when calling the model.")

    # Choose combination of BERT heads
    parser.add_argument("--plus_classify_sequence",
                        type=int, default=0,
                        help="Number of [CLS] (i.e. sequence) classificaiton heads to use, "
                             "number of labels should be set as a list in "
                             "data_handler.num_labels_cls and the mapping in"
                             "data_handler.text2id and data_handler.id2text.")
    parser.add_argument("--plus_classify_tokens",
                        type=int, default=0,
                        help="How many token classification heads should be used, "
                             "number of labels should be set as a list in"
                             "data_handler.num_labels_tok and the mapping in"
                             "data_handler.text2id_tok and data_handler.id2text_tok.")
    parser.add_argument("--plus_generation",
                        type=int, default=0,
                        help="Define how many generation heads there should be."
                             "(Generation head = use same vocabulary as input.")
    parser.add_argument("--token_index_path",
                        help="Dataset Biqe: Provide token index {name:id} for token classification")

    return parser
