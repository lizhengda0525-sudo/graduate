'''
  cMLP_dynamic.py
  生成动态的因果关系
'''

from models.model_helper import activation_helper
import torch.nn as nn
import torch

class MLPDynamic(nn.Module):
    def __init__(self, num_inputs, lag, hidden, activation='relu', time_emb_dim=16):
        super().__init__()
        self.activation = activation_helper(activation)
        self.lag = lag
        self.num_inputs = num_inputs
        self.hidden = hidden
        self.time_emb_dim = time_emb_dim

        # Hypernetwork生成第一层权重
        self.hyper_W = nn.Sequential(
            nn.Linear(time_emb_dim, hidden[0] * num_inputs * lag),
            nn.ReLU()
        )

        # 后续固定MLP层
        layers = []
        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            layers.append(nn.Conv1d(d_in, d_out, 1))
        self.layers = nn.ModuleList(layers)
    def forward(self, X, t_emb):
        """
        X: (batch, T, num_inputs)
        t_emb: (T, time_emb_dim)
        """
        X = X.transpose(2, 1)  # (batch, num_inputs, T)
        T = X.shape[2]

        # 生成动态第一层权重
        W_dyn = self.hyper_W(t_emb)  # (T, hidden * num_inputs * lag)
        W_dyn = W_dyn.view(T, self.hidden[0], self.num_inputs, self.lag)

        outputs = []
        for t in range(T - self.lag + 1):
            x_window = X[:, :, t:t+self.lag]  # (batch, num_inputs, lag)
            W_t = W_dyn[t]  # (hidden, num_inputs, lag)
            out = torch.einsum('hpl,bpl->bh', W_t, x_window)  # (batch, hidden)
            out = self.activation(out)
            for layer in self.layers[1:]:
                out = layer(out.unsqueeze(-1)).squeeze(-1)
            outputs.append(out.unsqueeze(2))
        return torch.cat(outputs, dim=2)  # (batch, 1, T-lag+1)


class cMLPDynamicHyper(nn.Module):
    def __init__(self, num_series, lag, hidden, activation='relu', time_emb_dim=16):
        super().__init__()
        self.p = num_series
        self.lag = lag
        self.time_emb_dim = time_emb_dim
        self.networks = nn.ModuleList([
            MLPDynamic(num_series, lag, hidden, activation, time_emb_dim)
            for _ in range(num_series)
        ])

    def forward(self, X, t_emb):
        return torch.cat([net(X, t_emb) for net in self.networks], dim=2)

    def dynamic_GC(self, t_emb, threshold=True, ignore_lag=True):
        """
        提取动态GC矩阵 (T, p, p)
        """
        T = t_emb.shape[0]
        GC_dyn = torch.zeros(T, self.p, self.p)
        for i, net in enumerate(self.networks):
            # 使用 hyper_W 输出的权重
            W_dyn = net.hyper_W(t_emb).view(T, net.hidden[0], net.num_inputs, net.lag)
            if ignore_lag:
                norm = torch.norm(W_dyn, dim=(1,3))  # (T, num_inputs)
            else:
                norm = torch.norm(W_dyn, dim=1)      # (T, num_inputs, lag)
            if threshold:
                norm = (norm > 0).int()
            GC_dyn[:, i, :] = norm
        return GC_dyn









