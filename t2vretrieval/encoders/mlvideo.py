import torch
import torch.nn as nn
import numpy as np
import framework.configbase


class MultilevelEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.dim_fts = [2048]
    self.dim_embed = 1024
    self.dropout = 0
    self.num_levels = 3
    self.share_enc = False


class SEBlock(nn.Module):
  def __init__(self, channel, r=16):
    super(SEBlock, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // r, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(channel // r, channel, bias=False),
      nn.Sigmoid(),
    )

  def forward(self, x):
    b, c, t = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1)
    y = torch.mul(x, y)
    return y


class MultilevelEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = sum(self.config.dim_fts)
    self.dropout = nn.Dropout(self.config.dropout)
    self.num_verbs = 10
    self.num_nouns = 20
    num_levels = 1 if self.config.share_enc else self.config.num_levels
    self.level_embeds = nn.ModuleList([
      nn.Linear(input_size, self.config.dim_embed, bias=True) for k in range(num_levels)])
    self.ft_attn = nn.Linear(self.config.dim_embed, 1, bias=True)

    verb_concept = 512
    noun_concept = 1024
    self.bn_verb = nn.BatchNorm1d(verb_concept)
    self.bn_noun = nn.BatchNorm1d(noun_concept)
    self.classifier_verb = nn.Sequential(nn.Conv1d(self.config.dim_embed, verb_concept, 5, stride=1, padding=2),
                                         self.bn_verb, nn.Sigmoid())
    self.classifier_noun = nn.Sequential(nn.Conv1d(self.config.dim_embed, noun_concept, 1), self.bn_noun,
                                         nn.Sigmoid())
    self.seblock1 = SEBlock(self.config.dim_embed)
    self.seblock2 = SEBlock(self.config.dim_embed)
    self.conv1 = nn.Sequential(nn.Conv1d(self.config.dim_embed, self.config.dim_embed, 1))
    self.conv2 = nn.Sequential(nn.Conv1d(self.config.dim_embed, self.config.dim_embed, 1))

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

    attn_scores = self.ft_attn(embeds[0]).squeeze(2)
    input_pad_masks = framework.ops.sequence_mask(input_lens, 
      max_len=attn_scores.size(1), inverse=True)
    attn_scores = attn_scores.masked_fill(input_pad_masks, -1e18)
    attn_scores = torch.softmax(attn_scores, dim=1)
    sent_embeds = torch.sum(embeds[0] * attn_scores.unsqueeze(2), 1)
    max_len = inputs.shape[1]
    batch_size = inputs.shape[0]
    ############################################################
    embeds_re1 = embeds[1].permute(0, 2, 1)
    embeds_re11 = self.seblock1(embeds_re1)
    embeds_re11 = embeds_re11.permute(0, 2, 1).contiguous()
    logits_verb = self.classifier_verb(embeds_re1)
    logits_verb = logits_verb.permute(0, 2, 1)  # batch*seq_len*concept

    embeds_re2 = embeds[2].permute(0, 2, 1)
    embeds_re22 = self.seblock2(embeds_re2)
    embeds_re22 = embeds_re22.permute(0, 2, 1).contiguous()
    logits_noun = self.classifier_noun(embeds_re2)
    logits_noun = logits_noun.permute(0, 2, 1)  # batch*seq_len*concept

    seq_len = input_lens.cpu().numpy()
    k = np.ceil(seq_len / 8).astype('int32')
    instance_logits_verb = torch.zeros(0).to(self.device)  # batch* concept
    instance_logits_noun = torch.zeros(0).to(self.device)
    for i in range(batch_size):
      tmp, _ = torch.topk(logits_verb[i][:seq_len[i]], k=int(k[i]), dim=0)
      instance_logits_verb = torch.cat([instance_logits_verb, torch.mean(tmp, 0, keepdim=True)], dim=0)
      tmp, _ = torch.topk(logits_noun[i][:seq_len[i]], k=int(k[i]), dim=0)
      instance_logits_noun = torch.cat([instance_logits_noun, torch.mean(tmp, 0, keepdim=True)], dim=0)

    embeds_verb = torch.zeros(0).to(self.device)
    embeds_noun = torch.zeros(0).to(self.device)
    _, top_idx_verb = torch.topk(instance_logits_verb, k=self.num_verbs, dim=1)  # batch*nv
    _, top_idx_noun = torch.topk(instance_logits_noun, k=self.num_nouns, dim=1)  # batch*nn

    for i in range(batch_size):
      logits_verb_tmp = logits_verb[i, :input_lens[i], top_idx_verb[i]]  # len*num_verb  logits_verb
      ind = torch.argmax(logits_verb_tmp, dim=0)  # num_verb
      if torch.max(ind)+2 < input_lens[i] and torch.min(ind) > 1:
        emb_tmp = (embeds_re11[i:i+1, ind-2, :]+embeds_re11[i:i+1, ind-1, :] + embeds_re11[i:i+1, ind, :]
                   + embeds_re11[i:i+1, ind+1, :]+embeds_re11[i:i+1, ind+2, :])/5
        embeds_verb = torch.cat([embeds_verb, emb_tmp], dim=0)
      else:
        embeds_verb = torch.cat([embeds_verb, embeds_re11[i:i + 1, ind, :]], dim=0)

      logits_noun_tmp = logits_noun[i, :input_lens[i], top_idx_noun[i]]  # len*num_noun

      if logits_noun_tmp.shape[0] > 2:
        _, ind = torch.topk(logits_noun_tmp, k=3, dim=0)  # 3*num_noun
        emb_tmp = (embeds_re22[i:i + 1, ind[0], :] + embeds_re22[i:i + 1, ind[1], :] + embeds_re22[i:i + 1, ind[2], :]) / 3
      else:
        ind = torch.argmax(logits_noun_tmp, dim=0)  # num_verb
        emb_tmp = embeds_re22[i:i + 1, ind, :]
      embeds_noun = torch.cat([embeds_noun, emb_tmp], dim=0)

    return sent_embeds, embeds_verb, embeds_noun, [embeds[1], embeds[2]], [instance_logits_verb, instance_logits_noun, top_idx_verb, top_idx_noun], max_len
