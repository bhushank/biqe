# coding=utf-8

#   File:     model_helper.py
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
Implements methods that help with handling the transformer models.

"""

import logging
import os
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.optimization import AdamW,get_linear_schedule_with_warmup, \
        get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup#, WarmupLinearSchedule, WarmupConstantSchedule, \
    #WarmupCosineSchedule

from .transformer_heads import MODEL_CLASSES

LOGGER = logging.getLogger(__name__)


def save_model(nsp_args, model, prefix=None):
    """
    Saves a model.

    :param nsp_args: instance of :py:class:NspArguments
    :param model: the model to save
    :param prefix: the prefix to attach to the file name
    :return: the location of the output file
    """
    # Only save the model it-self
    if prefix:
        output_model_file = os.path.join(nsp_args.output_dir, prefix)
    else:
        output_model_file = nsp_args.output_dir
    if torch.cuda.device_count() > 1:
        model = model.module
    model.save_pretrained(output_model_file)
    return output_model_file


def set_seed(seed, n_gpu=1):
    """
    Sets the seed.

    :param seed: seed to set, set -1 to draw a random number
    :param n_gpu:
    :return: 0 on success
    """
    if seed == -1:
        seed = random.randrange(2**32 - 1)
    LOGGER.info("Seed: %s", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    return 0


def get_train_dataloader(nsp_args, masker, data_handler, tokenizer):
    """
    Prepares the TensorDataset for training.

    :param nsp_args: instance of :py:class:NspArguments
    :param masker: the masker which will mask the data as appropriate, an instance of a subclass of
    :py:class:Masking
    :param data_handler: the dataset handler, an instance of :py:class:BitextHandler or a subclass
    :param tokenizer: a tokenizer
    :return: train_dataloader, an instance of :py:class:TensorDataset
    """
    #if not os.path.isfile(os.path.join(nsp_args.output_dir,'train.pt')):
    masker.convert_examples_to_features(
            data_handler=data_handler,
            tokenizer=tokenizer,
            max_seq_length=nsp_args.max_seq_length,
            max_part_a=nsp_args.max_part_a,
            is_training=True,
            nsp_args=nsp_args)
    train_data = data_handler.create_tensor_dataset()
        #torch.save(train_data,os.path.join(nsp_args.output_dir,'train.pt'))
    #else:
        #train_data = torch.load(os.path.join(nsp_args.output_dir,'train.pt'))

    if nsp_args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    data_handler.train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                               batch_size=nsp_args.train_batch_size)
    return 0


def get_model_elements(nsp_args, data_handler):
    """
    Loads model, tokenizer and config according to provided arguments.

    :param nsp_args: instance of :py:class:NspArguments
    :param data_handler: the dataset handler, an instance of :py:class:BitextHandler or a subclass
    :return:
    """
    config_class, model_class, tokenizer_class = MODEL_CLASSES[nsp_args.model_type]
    additional_arguments = get_additional_arguments(nsp_args, data_handler)
    nsp_args.output_attentions = True
    config = config_class.from_pretrained(
        nsp_args.config_name if nsp_args.config_name else nsp_args.model_name_or_path,
        output_hidden_states=nsp_args.output_hidden_states,
        output_attentions=nsp_args.output_attentions)
    tokenizer = tokenizer_class.from_pretrained(
        nsp_args.tokenizer_name if nsp_args.tokenizer_name else nsp_args.model_name_or_path,
        do_lower_case=nsp_args.do_lower_case)

    LOGGER.info('Using the following configuration: %s' % config_class)
    LOGGER.info('Using the following tokenizer: %s' % tokenizer_class)
    LOGGER.info('Using the following model: %s' % model_class)
    LOGGER.info('Using the following additional arguments: %s' % additional_arguments)
    if nsp_args.model_name_or_path == "vanilla":
        if nsp_args.do_train:
            LOGGER.info("Creating a new model with initial weights")
            model = model_class(config=config, **additional_arguments)
        else:
            model = model_class(config=config, **additional_arguments)
            path = os.path.join(nsp_args.output_dir, 'pytorch_model.bin')
            model.load_state_dict(torch.load(path))
    else:
        if nsp_args.do_train:
            LOGGER.info("Loading a previous model: %s" % nsp_args.model_name_or_path)
            model = model_class.from_pretrained(nsp_args.model_name_or_path,
                                            from_tf=bool('.ckpt' in nsp_args.model_name_or_path),
                                            config=config, **additional_arguments)
        else:
            model = model_class(config=config, **additional_arguments)
            path = os.path.join(nsp_args.output_dir, 'pytorch_model.bin')
            model.load_state_dict(torch.load(path))
    return config, tokenizer, model


def get_additional_arguments(nsp_args, data_handler):
    """
    Given the arguments, potentially add further arguments to a dictionary for model instantiation.
    Due to the transformers library set up, num_labels needs to be passed to the config,
    whereas all other additional values (e.g. here num_labels_tok) need to be passed directly
    to the model.
    :param nsp_args:
    :return:
    """
    additional_arguments = {}
    if nsp_args.plus_generation > 0:
        additional_arguments["generate"] = nsp_args.plus_generation
    if nsp_args.plus_classify_sequence > 0:
        additional_arguments["classify_sequence"] = nsp_args.plus_classify_sequence
        additional_arguments["num_labels_cls"] = data_handler.num_labels_cls
    if nsp_args.plus_classify_tokens > 0:
        additional_arguments["classify_tokens"] = nsp_args.plus_classify_tokens
        additional_arguments["num_labels_tok"] = data_handler.num_labels_tok
    return additional_arguments


def set_up_device(nsp_args):
    """
    Sets ups the device, gpu vs cpu and number of gpus if applicable.

    :param nsp_args: an instance of :py:class:NspArguments
    :return: tuple of:
            1. device: the device on which computations will be run
            2. n_gpu: the number of gpu's
    """
    if nsp_args.local_rank == -1 or nsp_args.no_cuda:
        # pylint: disable=not-callable
        # pylint: disable=no-member
        device = torch.device("cuda" if torch.cuda.is_available() and not nsp_args.no_cuda
                              else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(0)
        # pylint: disable=not-callable
        # pylint: disable=no-member
        device = torch.device("cuda", 0)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    return device, n_gpu


def move_model(nsp_args, model, device, n_gpu):
    """
    Move model to correct device.

    :param nsp_args: instance of :py:class:NspArguments
    :param model: a BERT model
    :param device: the device to move to
    :param n_gpu: the number of GPUs to use
    :return: the moved model
    """
    if nsp_args.fp16:
        model.half()
    model.to(device)
    if nsp_args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            LOGGER.warning("Please install apex from https://www.github.com/nvidia/apex "
                           "to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        # TODO if true, then error: when trying to save model:
        # 'DataParallel' object has no attribute 'save_pretrained'
        model = torch.nn.DataParallel(model)
    return model


def argument_sanity_check(nsp_args):
    """
    Performs a couple of additional sanity check on the provided arguments.

    :param nsp_args: instance of :py:class:NspArguments
    :return: 0 on success (else an error is raise)
    """
    if nsp_args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(nsp_args.gradient_accumulation_steps))

    if not nsp_args.do_train and not nsp_args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if nsp_args.do_train:
        if not nsp_args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if nsp_args.do_predict:
        if not nsp_args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(os.path.join(nsp_args.output_dir, "pytorch_model.bin")) \
            and nsp_args.do_train:
        if not nsp_args.load_prev_model:
            raise ValueError("Output directory already contains a saved model (pytorch_model.bin).")
    os.makedirs(nsp_args.output_dir, exist_ok=True)
    return 0


def prepare_optimizer(nsp_args, model, t_total):
    """
    Prepares the optimizer for training.
    :param nsp_args: instance of :py:class:NspArguments
    :param model: the model for which the optimizer will be created
    :param t_total: the total number of training steps that will be performed
            (need for learning rate schedules that depend on this)
    :return: the optimizer, the learning rate scheduler and the number of total steps
    """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    if nsp_args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = AdamW(optimizer_grouped_parameters, lr=nsp_args.learning_rate,
                      weight_decay=0.01, correct_bias=False, eps=1e-6)
    scheduler = None
    warmup_steps = nsp_args.warmup_proportion * t_total/2
    if nsp_args.adam_schedule == 'warmup_linear':
        LOGGER.info("Using linear warmup")
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps,t_total)
    elif nsp_args.adam_schedule == 'warmup_constant':
        LOGGER.info("Using constant warmup")
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    elif nsp_args.adam_schedule == 'warmup_cosine':
        LOGGER.info("Using cosine warmup")
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, t_total)

    return optimizer, scheduler

