import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

NUM_TASK = 3
EARTHRADIUS = 6371.0


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, device, outdim):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True).to(device) for _ in range(nheads)]

        self.out = nn.Linear(nhid*nheads, outdim)

    def forward(self, x, adj):

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out(x)


        return x

class Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.__dict__.update(params.__dict__)
        self.th_size = 12  # +1 because of the padding value 0,
        self.tw_size = 7
        # self.epsilon = torch.FloatTensor([1e-12]).to(torch.device(f"cuda:{self.gpu}"))

        # https://blog.csdn.net/qq_39540454/article/details/115215056
        self.feature_u = nn.Embedding(self.uid_size + 1, self.u_dim, padding_idx=0)
        self.feature_th = nn.Embedding(self.th_size + 1, self.th_dim, padding_idx=0)
        self.feature_tw = nn.Embedding(self.tw_size + 1, self.tw_dim, padding_idx=0)
        self.feature_c = nn.Embedding(self.cid_size + 1, self.c_dim, padding_idx=0)
        self.feature_l = nn.Embedding(self.pid_size + 1, self.l_dim, padding_idx=0)

        self.l_gat = GAT(nfeat=self.l_dim,
                         nhid=self.nhid,
                         dropout=self.dropout,
                         nheads=self.num_heads,
                         alpha=self.alpha,
                         outdim=self.gat_outdim,
                         device=torch.device(f"cuda:{self.gpu}"))

        self.c_hidden_gat = GAT(nfeat=self.c_dim,
                                nhid=self.nhid,
                                dropout=self.dropout,
                                nheads=self.num_heads,
                                alpha=self.alpha,
                                outdim=self.c_dim,
                                device=torch.device(f"cuda:{self.gpu}"))

        self.c_gat = GAT(nfeat=self.c_dim,
                         nhid=self.nhid,
                         dropout=self.dropout,
                         nheads=self.num_heads,
                         alpha=self.alpha,
                         outdim=self.gat_outdim,
                         device=torch.device(f"cuda:{self.gpu}"))


        self.l_dropout = nn.Dropout(p=self.dropout)
        self.c_dropout = nn.Dropout(p=self.dropout)
        self.l_h_dropout = nn.Dropout(p=self.dropout)
        self.c_h_dropout = nn.Dropout(p=self.dropout)

        self.l_rnn = nn.LSTM(self.u_dim + self.gat_outdim * 2 + self.l_dim + self.th_dim + self.tw_dim, self.rnn_dim, self.rnn_layer,
                             batch_first=True)
        self.c_rnn = nn.LSTM(self.u_dim + self.c_dim + self.gat_outdim + self.th_dim + self.tw_dim, self.rnn_dim, self.rnn_layer, batch_first=True)


        if self.res_factor > 1e-3:
            self.l_residual = nn.Linear(self.rnn_dim, self.pid_size)

        self.l_h_predictor = nn.Linear(self.rnn_dim + self.rnn_dim, self.rnn_dim)
        self.l_predictor = nn.Linear(self.rnn_dim + self.cid_size, self.pid_size)
        self.c_predictor = nn.Linear(self.rnn_dim, self.cid_size)

        self.CrossEntropy = nn.CrossEntropyLoss()

        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, data, mask_batch, l_adj, c_h_adj, cc_adj):

        u, l, tw, th, c = data

        f_u = self.feature_u(u)
        f_l = self.feature_l(l)
        f_c = self.feature_c(c)
        f_th = self.feature_th(th)
        f_tw = self.feature_tw(tw)

        f_all_l = self.feature_l.weight[1:]
        f_all_c = self.feature_c.weight[1:]

        l_f = torch.cat((f_all_l, f_all_c), dim=0)

        f_l_gat = self.l_gat(f_all_l, l_adj)
        f_c_h_gat = self.c_hidden_gat(l_f, c_h_adj)

        f_c_h_gat = f_c_h_gat[self.pid_size:]

        f_c_gat_in = torch.cat((f_all_c, f_c_h_gat), dim=0)
        f_c_gat = self.c_gat(f_c_gat_in, cc_adj)
        f_c_gat = f_c_gat[:self.cid_size]

        f_l_gat = F.pad(f_l_gat, (0, 0, 1, 0))
        f_c_gat = F.pad(f_c_gat, (0, 0, 1, 0))
        f_l_gat = F.embedding(l, f_l_gat)
        f_c_gat = F.embedding(c, f_c_gat)

        f_ucl = torch.cat((f_u.expand(-1, l.shape[1], -1), f_th, f_tw, f_c_gat, f_l, f_l_gat), dim=-1)
        f_tc = torch.cat((f_u.expand(-1, l.shape[1], -1), f_th, f_tw, f_c, f_c_gat), dim=-1)

        f_ucl = self.l_dropout(f_ucl)
        f_tc = self.c_dropout(f_tc)

        l_pack = pack_padded_sequence(f_ucl, mask_batch[1], batch_first=True, enforce_sorted=False)
        c_pack = pack_padded_sequence(f_tc, mask_batch[1], batch_first=True, enforce_sorted=False)

        _, (l_fh, _) = self.l_rnn(l_pack)
        _, (c_fh, _) = self.c_rnn(c_pack)

        l_fh = torch.squeeze(l_fh)
        c_fh = torch.squeeze(c_fh)

        l_fh = self.l_h_dropout(l_fh)
        c_fh = self.c_h_dropout(c_fh)

        if self.res_factor > 1e-3:
            l_pred_residual = self.l_residual(l_fh)

        l_fh = self.l_h_predictor(torch.cat((l_fh, c_fh), dim=1))
        c_pred = self.c_predictor(c_fh)
        l_pred = self.l_predictor(torch.cat((l_fh, c_pred), dim=-1))

        if self.res_factor > 1e-3:
            l_pred = (1 - self.res_factor) * l_pred + self.res_factor * l_pred_residual

        return l_pred, c_pred

    def calculate_loss(self, l_pred, target_batch, unique_batch, num_unique_batch, count_unique_batch, gt_weight):

        loss = 0
        logits = torch.log(torch.softmax(l_pred, dim=1))
        for i in range(target_batch.shape[0]):
            if num_unique_batch[i] == 0:
                continue

            loss += (1-gt_weight) * torch.sum(count_unique_batch[i] * logits[i][unique_batch[i]-1])
            loss += gt_weight * logits[i][target_batch[i]-1]

        if self.sum_loss:
            return -loss
        else:
            return -loss / target_batch.shape[0]

    def calculate_recall(self, l_pred, target_batch):
        if self.setting == 2:
            # topk = [5, 10, 20, 50]
            topk = [1, 5, 10, 20]
        else:
            topk = [1, 5, 10, 20]
        # target_batch -= 1
        recall = np.zeros(4)
        _, pred_idx = l_pred.topk(20)

        for idx, pred in enumerate(pred_idx):
            target = target_batch[idx]
            if target == 0:  # pad
                continue
            if target in pred[:topk[0]]:
                recall += 1
            elif target in pred[:topk[1]]:
                recall[1:] += 1
            elif target in pred[:topk[2]]:
                recall[2:] += 1
            elif target in pred[:topk[3]]:
                recall[3:] += 1

        return recall / l_pred.shape[0]

    def calculate_ndcg(self, l_pred, target_batch):
        if self.setting == 2:
            # topk = [5, 10, 20, 50]
            topk = [1, 5, 10, 20]
        else:
            topk = [1, 5, 10, 20]

        # target_batch -= 1
        ndcg = np.zeros(4)
        _, pred_idx = l_pred.topk(20)

        for idx, pred in enumerate(pred_idx):

            target = target_batch[idx]
            if target == 0:  # pad
                continue

            if target in pred[:20]:
                i = torch.where(target == pred)[0].item()
            else:
                continue

            if target in pred[:topk[0]]:
                ndcg[0] += 1.0 / np.log2(i + 2)
            if target in pred[:topk[1]]:
                ndcg[1] += 1.0 / np.log2(i + 2)
            if target in pred[:topk[2]]:
                ndcg[2] += 1.0 / np.log2(i + 2)
            if target in pred[:topk[3]]:
                ndcg[3] += 1.0 / np.log2(i + 2)

        return ndcg / l_pred.shape[0]

def generate_mask(data_len):
    '''Generate mask
    Args:
        data_len : one dimension list, reflect sequence length
    '''

    mask = []
    for i_len in data_len:
        mask.append(torch.ones(i_len).bool())

    return ~pad_sequence(mask, batch_first=True)