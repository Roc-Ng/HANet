import torch
import torch.nn as nn
import numpy as np

import t2vretrieval.encoders.graph
import t2vretrieval.encoders.sentence


class RoleGraphEncoderConfig(t2vretrieval.encoders.sentence.SentEncoderConfig):
  def __init__(self):
    super().__init__()
    self.num_roles = 16

    self.gcn_num_layers = 1
    self.gcn_attention = False
    self.gcn_dropout = 0.5


class RoleGraphEncoder(t2vretrieval.encoders.sentence.SentAttnEncoder):
  def __init__(self, config):
    super().__init__(config)
    if self.config.num_roles > 0:
      self.role_embedding = nn.Embedding(self.config.num_roles, self.config.rnn_hidden_size)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # GCN parameters
    print('GCN in Text:{}'.format(self.config.gcn_attention))
    self.gcn = t2vretrieval.encoders.graph.GCNEncoder(self.config.rnn_hidden_size, 
      self.config.rnn_hidden_size, self.config.gcn_num_layers, 
      attention=self.config.gcn_attention,
      embed_first=False, dropout=self.config.gcn_dropout)
    
    #########################################3
    verb_concept = 512
    noun_concept = 1024
    self.bn_verb = nn.BatchNorm1d(verb_concept)
    self.bn_noun = nn.BatchNorm1d(noun_concept)
    self.classifier_verb = nn.Sequential(nn.Conv1d(self.config.rnn_hidden_size, verb_concept, 1), self.bn_verb,
                                         nn.Sigmoid())
    self.classifier_noun = nn.Sequential(nn.Conv1d(self.config.rnn_hidden_size, noun_concept, 1), self.bn_noun,
                                         nn.Sigmoid())
    self.fc = nn.Linear(self.config.rnn_hidden_size, self.config.rnn_hidden_size)
    #########################################

  def pool_phrases(self, word_embeds, phrase_masks, pool_type='avg'):
    '''
    Args:
      word_embeds: (batch, max_sent_len, embed_size)
      phrase_masks: (batch, num_phrases, max_sent_len)
    Returns:
      phrase_embeds: (batch, num_phrases, embed_size)
    '''
    if pool_type == 'avg':
      # (batch, num_phrases, max_sent_len, embed_size)
      phrase_masks = phrase_masks.float()
      phrase_embeds = torch.bmm(phrase_masks, word_embeds) / torch.sum(phrase_masks, 2, keepdim=2).clamp(min=1)
    elif pool_type == 'max':
      embeds = word_embeds.unsqueeze(1).masked_fill(phrase_masks.unsqueeze(3)==0, -1e10)
      phrase_embeds = torch.max(embeds, 2)[0] 
    else:
      raise NotImplementedError
    return phrase_embeds

  def forward(self, sent_ids, sent_lens, verb_masks, noun_masks, node_roles, rel_edges):
    '''
    Args:
      sent_ids: (batch, max_sent_len)
      sent_lens: (batch, )
      verb_masks: (batch, num_verbs, max_sent_len)
      noun_masks: (batch, num_nouns, max_sent_len)
      node_roles: (batch, num_verbs + num_nouns)
    '''
    # (batch, max_sent_len, embed_size)
    word_embeds, word_attn_scores = super().forward(sent_ids, sent_lens, return_dense=True)
    sent_embeds = torch.sum(word_embeds * word_attn_scores.unsqueeze(2), 1) ## event embeds
    
    #####################################################################################
    word_embeds_re = word_embeds.permute(0, 2, 1)
    logits_verb = self.classifier_verb(word_embeds_re)  # batch*seq_len*concept
    logits_noun = self.classifier_noun(word_embeds_re)  # batch*seq_len*concept
    logits_verb = logits_verb.permute(0, 2, 1)
    logits_noun = logits_noun.permute(0, 2, 1)
    seq_len = sent_lens.cpu().numpy()
    k = np.ceil(seq_len / 8).astype('int32')
    instance_logits_verb = torch.zeros(0).to(self.device)  # batch* concept
    instance_logits_noun = torch.zeros(0).to(self.device)
    for i in range(logits_verb.shape[0]):
      tmp, _ = torch.topk(logits_verb[i][:seq_len[i]], k=int(k[i]), dim=0)
      instance_logits_verb = torch.cat([instance_logits_verb, torch.mean(tmp, 0, keepdim=True)], dim=0)
      tmp, _ = torch.topk(logits_noun[i][:seq_len[i]], k=int(k[i]), dim=0)
      instance_logits_noun = torch.cat([instance_logits_noun, torch.mean(tmp, 0, keepdim=True)], dim=0)
    ######################################################################################
    # (batch, num_phrases, embed_size)
    num_verbs = verb_masks.size(1)
    verb_embeds = self.pool_phrases(word_embeds, verb_masks, pool_type='max')
    embeds_1 = verb_embeds

    if self.config.num_roles > 0:
      verb_embeds = verb_embeds * self.role_embedding(node_roles[:, :num_verbs])

    noun_embeds = self.pool_phrases(word_embeds, noun_masks, pool_type='max')
    embeds_2 = noun_embeds
    if self.config.num_roles > 0:
      noun_embeds = noun_embeds * self.role_embedding(node_roles[:, num_verbs:])  # batch*n_n*dim

    node_embeds = torch.cat([sent_embeds.unsqueeze(1), verb_embeds, noun_embeds], 1)  # batch*(1+n_v+n_n)*dim
    node_ctx_embeds = self.gcn(node_embeds, rel_edges)  # batch*(1+n_v+n_n)*dim2
    sent_ctx_embeds = node_ctx_embeds[:, 0]
    verb_ctx_embeds = node_ctx_embeds[:, 1: 1 + num_verbs].contiguous()
    noun_ctx_embeds = node_ctx_embeds[:, 1 + num_verbs:].contiguous()
    return sent_ctx_embeds, verb_ctx_embeds, noun_ctx_embeds, [embeds_1, embeds_2], [instance_logits_verb, instance_logits_noun]

