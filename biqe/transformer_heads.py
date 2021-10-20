# coding=utf-8
#   File:     transformer_heads.py
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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead, BertLMPredictionHead
from transformers import (BertConfig, BertTokenizer, XLMConfig, XLMTokenizer, XLMWithLMHeadModel,
                          XLNetConfig, XLNetTokenizer, XLNetLMHeadModel)
from nsp.tokenizers.kg_tokenizers import FB15KTokenizer
from transformers.configuration_utils import PretrainedConfig

LOGGER = logging.getLogger(__name__)

#TODO: add the class where only one head but expects lists


# Generation plus Sequence and Token Classification
class BertGST(BertPreTrainedModel):
    """BERT model for token-level and sentence-level classification as well as generation.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer and a parallel masked language modeling head.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `generate`: True if the generation head should be active
        `classify_sequence`: True if a sequence should be classified
                                (set num_labels_cls to 1 for regression)
        `classify_tokens`: True if tokens should be classified
        `num_labels_tok`: the number of classes for the classifier on the tokens.
        `num_labels_cls`: the number of classes for the classifier on the [CLS] token
            (for sequence classification).

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in
            the scripts `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with
            the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and
            type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with
            indices selected in [0, 1]. It's a mask to be used if the input sequence length is
            smaller than the max input sequence length in the current batch. It's the mask that we
            typically use for attention when a batch has varying length sentences.
        `labels_tok`: labels for the classification output: torch.LongTensor of shape
            [batch_size, sequence_length] with indices selected in [0, ..., num_labels_tok].
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape
            [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size].
            All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `labels_cls`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels_cls].

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
            **loss**: (`optional`, returned when ``masked_lm_labels``, ``labels_tok``
                      or ``labels_cls`` are provided)
                ``torch.FloatTensor`` of shape ``(1,)``:
                Combined loss.
            **prediction_scores**: ``torch.FloatTensor`` of shape
                ``(batch_size, sequence_length, config.vocab_size)``
                Prediction scores of the language modeling head
                (scores for each vocabulary token before SoftMax).
            **tok_logits**: **scores**: ``torch.FloatTensor`` of shape
                ``(batch_size, sequence_length, num_labels_tok)``
                Classification scores (before SoftMax).
            **cls_logits**: ``torch.FloatTensor`` of shape ``(batch_size, num_labels_cls)``
                Classification (or regression if num_labels_cls==1) scores (before SoftMax).
            **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average
            in the self-attention heads.


    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]
    """

    def __init__(self, config, generate=False, classify_sequence=False, classify_tokens=False,
                 num_labels_tok=0, num_labels_cls=0):
        super(BertGST, self).__init__(config)

        # shared
        self.bert = BertModel(config)
        self.generate = generate
        self.classify_sequence = classify_sequence
        self.classify_tokens = classify_tokens

        # sequence generation
        if self.generate is True:
            self.cls = BertOnlyMLMHead(config)
            self.tie_weights()

        # CLS classification
        if self.classify_sequence is True:
            self.num_labels_cls = num_labels_cls
            self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.cls_classifier = nn.Linear(config.hidden_size, num_labels_cls)

        # token classification
        if self.classify_tokens is True:
            self.num_labels_tok = num_labels_tok
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, num_labels_tok)

        self.init_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        # tie_weights is called in model_utils.py if it exists, but we only actually want
        # to tie them when we have generation on
        if self.generate is True:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels_tok=None, masked_lm_labels=None, labels_cls=None):
        # outputs: sequence_output, pooled_output, (hidden_states), (attentions)
        model_outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = model_outputs[0]
        pooled_output = model_outputs[1]

        outputs = ()

        # Sequence Generation
        if self.generate is True:
            prediction_scores = self.cls(sequence_output)
            outputs = outputs + (prediction_scores,)

        # CLS classification
        # BertForSequenceClassification uses dropout here but BertForNextSentencePrediction doesn't
        if self.classify_sequence is True:
            pooled_output = self.cls_dropout(pooled_output)
            cls_logits = self.cls_classifier(pooled_output)
            outputs = outputs + (cls_logits,)

        # Token Classification
        if self.classify_tokens is True:
            sequence_output_dropout = self.dropout(sequence_output)
            tok_logits = self.classifier(sequence_output_dropout)
            outputs = outputs + (tok_logits,)

        outputs = outputs + model_outputs[2:]  # add hidden states and attention if they are here

        total_loss = 0.0

        # Sequence Generation
        if self.generate is True and masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            generation_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                      masked_lm_labels.view(-1))
            total_loss += generation_loss

        # CLS classification
        if self.classify_sequence is True and labels_cls is not None:
            if self.num_labels_cls == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                cls_loss = loss_fct(cls_logits.view(-1), labels_cls.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                cls_loss = loss_fct(cls_logits.view(-1, self.num_labels_cls), labels_cls.view(-1))
            total_loss += cls_loss

        # Token Classification
        if self.classify_tokens is True and labels_tok is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = tok_logits.view(-1, self.num_labels_tok)[active_loss]
                active_labels = labels_tok.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(tok_logits.view(-1, self.num_labels_tok), labels_tok.view(-1))
            token_loss = loss
            total_loss += token_loss

        if masked_lm_labels is not None or labels_cls is not None or labels_tok is not None:
            outputs = (total_loss,) + outputs

        # total_loss, prediction_scores, tok_logits, cls_logits, (hidden_states), (attentions)
        return outputs


class BertGSTList(BertPreTrainedModel):
    """BERT model for token-level and sentence-level classification as well as generation.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer and a parallel masked language modeling head.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `generate`: True if the generation head should be active
        `classify_sequence`: True if a sequence should be classified
                                (set num_labels_cls to 1 for regression)
        `classify_tokens`: True if tokens should be classified
        `num_labels_tok`: the number of classes for the classifier on the tokens.
        `num_labels_cls`: the number of classes for the classifier on the [CLS] token
            (for sequence classification).

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in
            the scripts `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with
            the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and
            type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with
            indices selected in [0, 1]. It's a mask to be used if the input sequence length is
            smaller than the max input sequence length in the current batch. It's the mask that we
            typically use for attention when a batch has varying length sentences.
        `labels_tok`: labels for the classification output: torch.LongTensor of shape
            [batch_size, sequence_length] with indices selected in [0, ..., num_labels_tok].
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape
            [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size].
            All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `labels_cls`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels_cls].

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
            **loss**: (`optional`, returned when ``masked_lm_labels``, ``labels_tok``
                      or ``labels_cls`` are provided)
                ``torch.FloatTensor`` of shape ``(1,)``:
                Combined loss.
            **prediction_scores**: ``torch.FloatTensor`` of shape
                ``(batch_size, sequence_length, config.vocab_size)``
                Prediction scores of the language modeling head
                (scores for each vocabulary token before SoftMax).
            **tok_logits**: **scores**: ``torch.FloatTensor`` of shape
                ``(batch_size, sequence_length, num_labels_tok)``
                Classification scores (before SoftMax).
            **cls_logits**: ``torch.FloatTensor`` of shape ``(batch_size, num_labels_cls)``
                Classification (or regression if num_labels_cls==1) scores (before SoftMax).
            **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average
            in the self-attention heads.


    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]
    """

    def __init__(self, config: PretrainedConfig,
                 generate: int = 0,
                 classify_sequence: int = 0,
                 classify_tokens: int = 0,
                 num_labels_tok: list = None,
                 num_labels_cls: list = None):
        super(BertGSTList, self).__init__(config)

        # shared
        self.bert = BertModel(config)
        self.generate = generate
        self.classify_sequence = classify_sequence
        self.classify_tokens = classify_tokens
        self.num_labels_cls = num_labels_cls
        self.num_labels_tok = num_labels_tok

        # sequence generation
        if self.generate > 0:
            self.cls = BertOnlyMLMHead(config)
            self.tie_weights()

        # CLS classification
        if self.classify_sequence > 0:
            self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.cls_classifier = nn.Linear(config.hidden_size, num_labels_cls[0])

        # token classification
        if self.classify_tokens > 0:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, num_labels_tok[0])

        self.init_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        # tie_weights is called in model_utils.py if it exists, but we only actually want
        # to tie them when we have generation on
        if self.generate is True:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels_tok=None, masked_lm_labels=None, labels_cls=None, loss_mask=None):
        # outputs: sequence_output, pooled_output, (hidden_states), (attentions)
        model_outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output = model_outputs[0]
        pooled_output = model_outputs[1]

        outputs = ()

        # Sequence Generation
        if self.generate > 0:
            prediction_scores = self.cls(sequence_output)
            outputs = outputs + (prediction_scores,)

        # CLS classification
        # BertForSequenceClassification uses dropout here but BertForNextSentencePrediction doesn't
        if self.classify_sequence > 0:
            pooled_output = self.cls_dropout(pooled_output)
            cls_logits = self.cls_classifier(pooled_output)
            outputs = outputs + (cls_logits,)

        # Token Classification
        if self.classify_tokens > 0:
            sequence_output_dropout = self.dropout(sequence_output)
            tok_logits = self.classifier(sequence_output_dropout)
            outputs = outputs + (tok_logits,)

        outputs = outputs + model_outputs[2:]  # add hidden states and attention if they are here

        total_loss = 0.0

        # Sequence Generation
        if self.generate > 0 and masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            generation_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                      masked_lm_labels.view(-1))
            total_loss += generation_loss

        # CLS classification
        if self.classify_sequence > 0 and labels_cls is not None:
            if self.num_labels_cls[0] == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                cls_loss = loss_fct(cls_logits.view(-1), labels_cls.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                cls_loss = loss_fct(cls_logits.view(-1, self.num_labels_cls[0]), labels_cls.view(-1))
            total_loss += cls_loss

        # Token Classification
        if self.classify_tokens > 0 and labels_tok is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = tok_logits.view(-1, self.num_labels_tok[0])[active_loss]
                active_labels = labels_tok.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(tok_logits.view(-1, self.num_labels_tok[0]), labels_tok.view(-1))
            token_loss = loss
            total_loss += token_loss

        if masked_lm_labels is not None or labels_cls is not None or labels_tok is not None:
            outputs = (total_loss,) + outputs

        # total_loss, prediction_scores, tok_logits, cls_logits, (hidden_states), (attentions)
        return outputs


class BertGSTListHeads(BertPreTrainedModel):
    """BERT model for token-level and sentence-level classification as well as generation.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer and a parallel masked language modeling head.

    Each layer head type (token, cls/sequence and generation) is modelled as a list (nn.Module)
    and there can thus be arbitrarily many heads.
    Currently, the CLS head is not loaded from the pre-trained weights.
    For each generation head, we copy the weights from the pre-training.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `generate`: True if the generation head should be active
        `classify_sequence`: True if a sequence should be classified
                                (set num_labels_cls to 1 for regression)
        `classify_tokens`: True if tokens should be classified
        `num_labels_tok`: the number of classes for the classifier on the tokens.
        `num_labels_cls`: the number of classes for the classifier on the [CLS] token
            (for sequence classification).

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in
            the scripts `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with
            the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and
            type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with
            indices selected in [0, 1]. It's a mask to be used if the input sequence length is
            smaller than the max input sequence length in the current batch. It's the mask that we
            typically use for attention when a batch has varying length sentences.
        `labels_tok`: labels for the classification output: torch.LongTensor of shape
            [batch_size, sequence_length] with indices selected in [0, ..., num_labels_tok].
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape
            [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size].
            All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `labels_cls`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels_cls].

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
            **loss**: (`optional`, returned when ``masked_lm_labels``, ``labels_tok``
                      or ``labels_cls`` are provided)
                ``torch.FloatTensor`` of shape ``(1,)``:
                Combined loss.
            **prediction_scores**: ``torch.FloatTensor`` of shape
                ``(batch_size, sequence_length, config.vocab_size)``
                Prediction scores of the language modeling head
                (scores for each vocabulary token before SoftMax).
            **tok_logits**: **scores**: ``torch.FloatTensor`` of shape
                ``(batch_size, sequence_length, num_labels_tok)``
                Classification scores (before SoftMax).
            **cls_logits**: ``torch.FloatTensor`` of shape ``(batch_size, num_labels_cls)``
                Classification (or regression if num_labels_cls==1) scores (before SoftMax).
            **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average
            in the self-attention heads.


    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]
    """

    def __init__(self, config: PretrainedConfig,
                 generate: int = 0,
                 classify_sequence: int = 0,
                 classify_tokens: int = 0,
                 num_labels_tok: list = None,
                 num_labels_cls: list = None):
        """
        Loads a transformer encoder with various heads.
        :param config: The configuration for the transformer encoder
        :param generate: How many LM generation heads to use (same output vocabulary as input)
        :param classify_sequence: How many sequence classification heads to use.
        :param classify_tokens: How many token classification heads to use.
        :param num_labels_tok: The number of classes for the token classification heads.
                                Given as a list where the list length should be equal to the number
                                of token classification heads.
        :param num_labels_cls: The number of classes for the sequence classification heads.
                                Given as a list where the list length should be equal to the number
                                of sequence classification heads.
        """
        super(BertGSTListHeads, self).__init__(config)

        # shared
        self.bert = BertModel(config)
        self.generate = generate
        self.classify_sequence = classify_sequence
        self.classify_tokens = classify_tokens
        self.num_labels_cls = num_labels_cls
        self.num_labels_tok = num_labels_tok

        # sequence generation
        if self.generate > 0:
            # the first head should be called cls so we can load the pre-trained weights
            # further heads are in self.lm_heads
            self.cls = BertOnlyMLMHead(config)
            self.lm_heads = nn.ModuleList()
            for _ in range(1, self.generate):
                # TODO support different config for different heads.
                self.lm_heads.append(BertOnlyMLMHead(config))
            self.tie_weights()

        # CLS classification
        if self.classify_sequence > 0:
            self.cls_dropout = nn.ModuleList()
            self.cls_classifier = nn.ModuleList()
            for i in range(self.classify_sequence):
                self.cls_dropout.append(nn.Dropout(config.hidden_dropout_prob))
                self.cls_classifier.append(nn.Linear(config.hidden_size, num_labels_cls[i]))

        # token classification
        if self.classify_tokens > 0:
            self.dropout = nn.ModuleList()
            self.classifier = nn.ModuleList()
            for i in range(self.classify_tokens):
                self.dropout.append(nn.Dropout(config.hidden_dropout_prob))
                self.classifier.append(nn.Linear(config.hidden_size, num_labels_tok[i]))

        self.init_weights()
        LOGGER.info("Using %d generation heads, %d token classification heads "
                    "and %d sequence/sequence pair classification/regression heads.",
                    self.generate, self.classify_sequence, self.classify_tokens)

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        # tie_weights is called in model_utils.py if it exists, but we only actually want
        # to tie them when we have generation on
        if self.generate > 0:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
            if self.generate > 1:
                for i, _ in enumerate(self.lm_heads):
                    self._tie_or_clone_weights(self.lm_heads[i].predictions.decoder,
                                            self.bert.embeddings.word_embeddings)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None, loss_mask: torch.FloatTensor=None,
                token_type_ids: torch.LongTensor = None, position_ids: torch.LongTensor = None,
                head_mask: torch.FloatTensor = None, labels_tok: torch.tensor = None,
                masked_lm_labels: torch.tensor = None, labels_cls: torch.tensor = None):
        """
        labels_tok, masked_lm_labels, labels_cls: Have to be a list of lists. Each entry in the list
        should map to the corresponding head. Pass None in the list if a training example should
        not effect a certain head. Pass overall None if no such head is used at all.

        :param input_ids: torch tensor of size batch x max_length, contains tokenized input as IDs
        :param attention_mask: torch tensor of size batch x max_length, contains 1 if position
                                should be attended to, else 0
        :param token_type_ids: torch tensor of size batch x max_length, traditionally contains 0 for
                                Part A and 1 for Part B, could be used to model other connections
        :param position_ids: from the transformer repo documentation: "Indices of positions of
                                each input sequence tokens in the position embeddings."
        :param head_mask: used to mask heads by the underlying BERT models,
                            we don't intend to use it here
        :param labels_tok: torch tensor of size self.generate x batch x max_length, -1 for tokens that we do not
                            want to predict on, else the classification that should be assigned to
                            that token
        :param masked_lm_labels: torch tensor of size self.generate x batch x max_length,
                                -1 for tokens that we do not want to predict on,
                                else the vocabulary ID this token should become
        :param labels_cls: torch tensor of size self.generate x batch x 1, contains the
                            classifcation that should be assigned to the entire input
                            (i.e. sequence or sequence pair classification)
        :return: A tuple which contains elements depending on the configuration and model call:
        (total_loss), (prediction_scores * self.generate), (tok_logits * self.classify_sequence),
        (cls_logits * self.classify_tokens), (hidden_states), (attentions)

        total_loss (torch.FloatTensor) is contained if any of the three where set when the
        function was called: labels_tok, masked_lm_labels, labels_cls

        For generation, prediction_scores (torch.LongTensor, batch x max_length):
            prediction logits for each generation head
        For token classification, tok_logits:  (torch.LongTensor, batch x max_length):
            token logits for each tokenisation head
        For sequence classification, classify_sequence:  (torch.LongTensor, batch x max_length):
            logits for each sequence/sequence pair head
        """
        # outputs: sequence_output, pooled_output, (hidden_states), (attentions)
        model_outputs = self.bert(input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=head_mask)

        sequence_output = model_outputs[0]
        pooled_output = model_outputs[1]

        outputs = ()

        # Sequence Generation
        prediction_scores = []
        if self.generate > 0:
            # Iterate over all heads
            head_prediction_scores = self.cls(sequence_output)
            prediction_scores.append(head_prediction_scores)
            outputs = outputs + (head_prediction_scores,)
            if self.generate > 1:
                for i, _ in enumerate(self.lm_heads):
                    head_prediction_scores = self.lm_heads[i](sequence_output)
                    prediction_scores.append(head_prediction_scores)
                    outputs = outputs + (head_prediction_scores,)

        # CLS classification
        # BertForSequenceClassification uses dropout here but BertForNextSentencePrediction doesn't
        cls_logits = []
        if self.classify_sequence > 0:
            # Iterate over all heads
            for i in range(self.classify_sequence):
                pooled_output = self.cls_dropout[i](pooled_output)
                cls_logits_head = self.cls_classifier[i](pooled_output)
                cls_logits.append(cls_logits_head)
                outputs = outputs + (cls_logits_head,)

        # Token Classification
        tok_logits = []
        if self.classify_tokens > 0:
            # Iterate over all heads
            for i in range(self.classify_tokens):
                sequence_output_dropout = self.dropout[i](sequence_output)
                tok_logits_head = self.classifier[i](sequence_output_dropout)
                tok_logits.append(tok_logits_head)
                outputs = outputs + (tok_logits_head,)

        outputs = outputs + model_outputs[2:]  # add hidden states and attention if they are here

        total_loss = 0.0

        # Sequence Generation
        if self.generate > 0 and masked_lm_labels is not None:
            # Iterate over sequence generation heads
            masked_lm_labels_reshaped = masked_lm_labels.permute(1, 0, 2)
            for i in range(self.generate):
                # Maybe this head does not have any labels, then skip
                if masked_lm_labels_reshaped[i] is not [-1]:
                    loss_fct = CrossEntropyLoss(ignore_index=-1)
                    generation_loss = loss_fct(prediction_scores[i].view(-1, self.config.vocab_size),
                                               masked_lm_labels_reshaped[i].view(-1))
                    total_loss += generation_loss

        # CLS classification
        if self.classify_sequence > 0 and labels_cls is not None:
            # Iterate over CLS classification heads
            labels_cls_reshaped = labels_cls.permute(1, 0)
            for i in range(self.classify_sequence):
                # Maybe this head does not have any labels, then skip
                if labels_cls_reshaped[i] is not [-1]:  # Untested (cant have none in tensor)
                    if self.num_labels_cls[i] == 1:
                        #  We are doing regression
                        loss_fct = MSELoss()
                        cls_loss = loss_fct(cls_logits[i].view(-1), labels_cls_reshaped[i].view(-1))
                    else:
                        loss_fct = CrossEntropyLoss(ignore_index=-1)
                        cls_loss = loss_fct(cls_logits[i].view(-1, self.num_labels_cls[i]),
                                            labels_cls_reshaped[i].view(-1))
                    total_loss += cls_loss

        # Token Classification
        if self.classify_tokens > 0 and labels_tok is not None:
            # Iterate over CLS classification heads
            # This operation potentially causes non-contiguous tensors, so we call contiguous below
            # this will copy data though, is there a way to change it?
            #labels_tok_reshaped = labels_tok.permute(1, 0, 2)
            #labels_tok = labels_tok.squeeze(1)
            # cq best results
            labels_tok_reshaped = labels_tok.permute(1, 0, 2)
            for i in range(self.classify_tokens):
                # Maybe this head does not have any labels, then skip
                if labels_tok_reshaped[i] is not [-1]:
                    loss_fct = CrossEntropyLoss(ignore_index=-1)
                    # compute averaged loss
                    loss = 0
                    if attention_mask is not None:
                        active_loss = attention_mask.view(-1) == 1
                        active_logits = tok_logits[i].view(-1, self.num_labels_tok[i])[active_loss]
                        active_labels = labels_tok_reshaped[i].contiguous().view(-1)[active_loss]
                        loss += loss_fct(active_logits, active_labels)
                    else:
                        loss = loss_fct(tok_logits[i].view(-1, self.num_labels_tok[i]),
                                        labels_tok_reshaped[i].contiguous().view(-1))
                    token_loss = loss
                    total_loss += token_loss

        '''
        For Q2B use this code
        #q2b best results
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = 0
            active_loss = attention_mask.view(-1) == 1
            active_logits = tok_logits[0].view(-1, self.num_labels_tok[0])[active_loss]
            num_active, num_ones = 1, -1.0*labels_tok[:,0,:].size()[0] * labels_tok[:,0,:].size()[1]
            for i in range(labels_tok.size()[1]):
                if torch.sum(labels_tok[:,i,:]) <= num_ones:
                    continue
                active_labels = labels_tok[:,i,:].contiguous().view(-1)[active_loss]
                loss += loss_fct(active_logits, active_labels)
                num_active+=1
            token_loss = loss/num_active
            #token_loss = loss / 50
            total_loss += token_loss
        '''
        
        if masked_lm_labels is not None or labels_cls is not None or labels_tok is not None:
            outputs = (total_loss,) + outputs
        # (total_loss), (prediction_scores * self.generate), (tok_logits * self.classify_sequence),
        # (cls_logits * self.classify_tokens), (hidden_states), (attentions)
        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super(BertGSTListHeads, cls).from_pretrained(pretrained_model_name_or_path,
                                                             *model_args, **kwargs)
        # if there is more than one generation head, copy the weights to there
        if model.generate > 1:
            for i, _ in enumerate(model.lm_heads):
                model.lm_heads[i].load_state_dict(model.cls.state_dict())
        return model

#TODO: add support for other models (atm the only implement seq class and lm head but not tok class)

#ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                 # for conf in (BertConfig, XLNetConfig, XLMConfig)), ())


MODEL_CLASSES = {
    'bert': (BertConfig, BertGSTListHeads, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
    'xlm': (XLMConfig, XLMWithLMHeadModel, XLMTokenizer),
    'kg': (BertConfig, BertGSTListHeads, FB15KTokenizer)
    #'albert': (AlbertConfig, AlbertForMaskedLM,AlbertTokenizer)
}
