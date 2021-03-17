import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import framework.configbase
import t2vretrieval.encoders.graph
import json
import time


class MultilevelEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.dim_fts = [2048]
    self.dim_embed = 1024
    self.dropout = 0

    self.num_levels = 3
    self.share_enc = False

    self.gcn_num_layers = 1
    self.gcn_attention = True
    self.gcn_dropout = 0.5

    # self.verb_label_file = ''
    # self.noun_label_file = ''

class MultilevelEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = sum(self.config.dim_fts)
    self.dropout = nn.Dropout(self.config.dropout)
    self.num_verbs = 10
    self.num_nouns = 15
    num_levels = 1 if self.config.share_enc else self.config.num_levels
    self.level_embeds = nn.ModuleList([
      nn.Linear(input_size, self.config.dim_embed, bias=True) for k in range(num_levels)])
    # self.embedLayer = nn.Sequential(nn.Linear(input_size, self.config.dim_embed, bias=True), self.dropout,
                                    # nn.ReLU())
    self.ft_attn = nn.Linear(self.config.dim_embed, 1, bias=True)
    # add classifier

    verb_concept = 512
    noun_concept = 1024
    self.bn_verb = nn.BatchNorm1d(verb_concept)
    self.bn_noun = nn.BatchNorm1d(noun_concept)
    self.classifier_verb = nn.Sequential(nn.Conv1d(self.config.dim_embed, verb_concept, 3, stride=1, padding=1),
                                         self.bn_verb, nn.Sigmoid())
    self.classifier_noun = nn.Sequential(nn.Conv1d(self.config.dim_embed, noun_concept, 1), self.bn_noun,
                                         nn.Sigmoid())
    #  GCN
    # GCN parameters
    self.gcn = t2vretrieval.encoders.graph.GCNEncoder(self.config.dim_embed,
                                                      self.config.dim_embed, self.config.gcn_num_layers,
                                                      attention=self.config.gcn_attention, # True
                                                      embed_first=False, dropout=self.config.gcn_dropout)

    self.num_nodes = 1 + self.num_verbs+self.num_nouns
    self.rel_matrix = torch.zeros(300, self.num_nodes, self.num_nodes, dtype=torch.float32).to(self.device)
    self.rel_matrix[:, 0, 1:self.num_verbs+1] = 1
    self.rel_matrix[:, 1:self.num_verbs + 1, 0] = 1
    self.rel_matrix[:, 1:self.num_verbs + 1, 1+self.num_verbs:] = 1
    self.rel_matrix[:, 1 + self.num_verbs:, 1:self.num_verbs + 1, ] = 1

  def softmax2(self, x):
    return (10*(torch.exp(x)-1))/(torch.sum((10*(torch.exp(x)-1)), dim=0)+1e-10)

  def forward(self, inputs, input_lens):
    '''
    Args:
      inputs: (batch, max_seq_len, dim_fts)
    Return:
      sent_embeds: (batch, dim_embed)
      verb_embeds: (batch, max_seq_len, dim_embed)
      noun_embeds: (batch, max_seq_len, dim_embed)
    '''
    embeds = []
    for k in range(self.config.num_levels):
      if self.config.share_enc:
        k = 0
      embeds.append(self.dropout(self.level_embeds[k](inputs)))
    # embeds = self.embedLayer(inputs)
    attn_scores = self.ft_attn(embeds[0]).squeeze(2)
    input_pad_masks = framework.ops.sequence_mask(input_lens, 
      max_len=attn_scores.size(1), inverse=True)
    attn_scores = attn_scores.masked_fill(input_pad_masks, -1e18)
    attn_scores = torch.softmax(attn_scores, dim=1)
    sent_embeds = torch.sum(embeds[0] * attn_scores.unsqueeze(2), 1)
    max_len = inputs.shape[1]
    batch_size = inputs.shape[0]
    ############################################################
    # print(inputs.shape)
    embeds_re = embeds[1].permute(0, 2, 1)
    logits_verb = self.classifier_verb(embeds_re)  # batch*seq_len*concept
    logits_verb = logits_verb.permute(0, 2, 1)
    embeds_re2 = embeds[2].permute(0, 2, 1)
    logits_noun = self.classifier_noun(embeds_re2)  # batch*seq_len*concept
    logits_noun = logits_noun.permute(0, 2, 1)
    seq_len = input_lens.cpu().numpy()
    k = np.ceil(seq_len / 8).astype('int32')
    instance_logits_verb = torch.zeros(0).to(self.device)  # batch* concept
    instance_logits_noun = torch.zeros(0).to(self.device)
    for i in range(batch_size):
      tmp, _ = torch.topk(logits_verb[i][:seq_len[i]], k=int(k[i]), dim=0)
      instance_logits_verb = torch.cat([instance_logits_verb, torch.mean(tmp, 0, keepdim=True)], dim=0)
      tmp, _ = torch.topk(logits_noun[i][:seq_len[i]], k=int(k[i]), dim=0)
      instance_logits_noun = torch.cat([instance_logits_noun, torch.mean(tmp, 0, keepdim=True)], dim=0)

    return sent_embeds, embeds[1], embeds[2], [instance_logits_verb, instance_logits_noun], max_len
    # logits 这里需要弱监督学习
    ## 训练的时候用真实的concept 测试时候用预测最高的结果
    batch = inputs.shape[0]
    rel_matrix = self.rel_matrix[:batch, :, :]
    embeds_verb = torch.zeros(0).to(self.device)
    embeds_noun = torch.zeros(0).to(self.device)

    # emded_verb = torch.zeros(batch, self.num_verbs, dim_emded, dtype=torch.float32).to(self.device)
    # emded_noun = torch.zeros(batch, self.num_nouns, dim_emded,  dtype=torch.float32).to(self.device)
    _, top_idx_verb = torch.topk(instance_logits_verb, k=self.num_verbs, dim=1) # batch*4
    _, top_idx_noun = torch.topk(instance_logits_noun, k=self.num_nouns, dim=1) # batch*6
    # print(top_idx_verb[0:10])
    # print(top_idx_noun[0:10])
    for i in range(batch):
      # st = time.time()
      logits_verb_tmp = logits_verb[i, :input_lens[i], top_idx_verb[i]]  # len*4
      logits_verb_tmp_norm = self.softmax2(logits_verb_tmp)
      # logits_verb_tmp_norm = logits_verb_tmp / (torch.sum(logits_verb_tmp, dim=0)+1e-10)  # len*num_verb
      embeds_verb_tmp = logits_verb_tmp_norm.permute(1, 0).mm(embeds[1][i, :input_lens[i], :])
      embeds_verb = torch.cat([embeds_verb, embeds_verb_tmp.unsqueeze(0)], dim=0)

      logits_noun_tmp = logits_noun[i, :input_lens[i], top_idx_noun[i]]  # len*6
      logits_noun_tmp_norm = self.softmax2(logits_noun_tmp)
      # logits_noun_tmp_norm = logits_noun_tmp / torch.sum(logits_noun_tmp, dim=0)  # len*num_noun
      embeds_noun_tmp = logits_noun_tmp_norm.permute(1, 0).mm(embeds[2][i, :input_lens[i], :])
      embeds_noun = torch.cat([embeds_noun, embeds_noun_tmp.unsqueeze(0)], dim=0)
    return sent_embeds, embeds_verb, embeds_noun, [instance_logits_verb, instance_logits_noun], max_len
    ## GCN

    node_embeds = torch.cat([sent_embeds.unsqueeze(1), embeds_verb, embeds_noun], 1)  # batch*(1+n_v+n_n)*dim
    node_ctx_embeds = self.gcn(node_embeds, rel_matrix)  # batch*(1+n_v+n_n)*dim2
    sent_ctx_embeds = node_ctx_embeds[:, 0]
    verb_ctx_embeds = node_ctx_embeds[:, 1: 1 + self.num_verbs].contiguous()
    noun_ctx_embeds = node_ctx_embeds[:, 1+self.num_verbs:].contiguous()
    # st3 = time.time()
    # print('gcn time:{:.3f}s'.format(st3-st2))

    return sent_ctx_embeds, verb_ctx_embeds, noun_ctx_embeds, [instance_logits_verb, instance_logits_noun], max_len