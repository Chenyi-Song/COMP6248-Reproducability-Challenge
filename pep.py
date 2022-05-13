import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class PEPEmbedding(nn.Module):
    def __init__(self, threshold_type, latent_dim, field_dim, g_type, gk, threshold_init,
                 retrain=False, emb_save_path=None, retrain_emb_param=0, re_init=False):
        super(PEPEmbedding, self).__init__()
        self.threshold_type = threshold_type
        self.latent_dim = latent_dim
        self.field_dims = field_dim
        self.feature_num = sum(field_dim)
        self.field_num = len(field_dim)
        self.g_type = g_type
        self.gk = gk
        init = threshold_init
        self.retrain = retrain
        self.mask = None

        self.g = torch.sigmoid
        self.s = self.init_threshold(init)
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.long)

        self.v = torch.nn.Parameter(torch.rand(self.feature_num, self.latent_dim))
        torch.nn.init.xavier_uniform_(self.v)

        if self.retrain:
            self.init_retrain(emb_save_path, retrain_emb_param, re_init)
            print("Retrain:", emb_save_path)

        self.sparse_v = self.v.data

    def init_retrain(self, emb_save_path, retrain_emb_param, re_init=False):
        sparse_emb = np.load(emb_save_path.format(num=retrain_emb_param))
        sparse_emb = torch.from_numpy(sparse_emb)
        mask = torch.abs(torch.sign(sparse_emb))
        if re_init:
            init_emb = torch.nn.Parameter(torch.rand(self.feature_num, self.latent_dim))
            torch.nn.init.xavier_uniform_(init_emb)
        else:
            init_emb = np.load(emb_save_path.format(num='initial'))
            init_emb = torch.from_numpy(init_emb)

        init_emb = init_emb * mask
        self.v = torch.nn.Parameter(init_emb)
        self.mask = mask.to(device)
        self.gk = 0

    def init_threshold(self, init):
        if self.threshold_type == 'global':
            s = nn.Parameter(init * torch.ones(1))
        elif self.threshold_type == 'dimension':
            s = nn.Parameter(init * torch.ones([self.latent_dim]))
        elif self.threshold_type == 'feature':
            s = nn.Parameter(init * torch.ones([self.feature_num, 1]))
        elif self.threshold_type == 'field':
            s = nn.Parameter(init * torch.ones([self.field_num, 1]))
        elif self.threshold_type == 'feature_dim':
            s = nn.Parameter(init * torch.ones([self.feature_num, self.latent_dim]))
        elif self.threshold_type == 'field_dim':
            s = nn.Parameter(init * torch.ones([self.field_num, self.latent_dim]))
        else:
            raise ValueError('Invalid threshold_type: {}'.format(self.threshold_type))
        return s

    def soft_threshold(self, v, s):
        if s.size(0) == self.field_num:  # field-wise lambda
            field_v = torch.split(v, tuple(self.field_dims))
            concat_v = []
            for i, v in enumerate(field_v):
                v = torch.sign(v) * torch.relu(torch.abs(v) - (self.g(s[i]) * self.gk))
                concat_v.append(v)

            concat_v = torch.cat(concat_v, dim=0)
            return concat_v
        else:
            return torch.sign(v) * torch.relu(torch.abs(v) - (self.g(s) * self.gk))

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        self.sparse_v = self.soft_threshold(self.v, self.s)
        if self.retrain:
            self.sparse_v = self.sparse_v * self.mask
        xv = F.embedding(x, self.sparse_v)

        return xv
