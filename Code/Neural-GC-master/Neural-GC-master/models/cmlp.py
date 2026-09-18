import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from models.model_helper import activation_helper
import datetime
import os


class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)

        # Set up network.
        layer = nn.Conv1d(num_series, hidden[0], lag)
        modules = [layer]

        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            layer = nn.Conv1d(d_in, d_out, 1)
            modules.append(layer)

        # Register parameters.
        self.layers = nn.ModuleList(modules)

    def forward(self, X):
        X = X.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0:
                X = self.activation(X)
            X = fc(X)

        return X.transpose(2, 1)


class cMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation='relu'):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation)
            for _ in range(num_series)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if ignore_lag:
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


class cMLPSparse(nn.Module):
    def __init__(self, num_series, sparsity, lag, hidden, activation='relu'):
        '''
        cMLP model that only uses specified interactions.

        Args:
          num_series: dimensionality of multivariate time series.
          sparsity: torch byte tensor indicating Granger causality, with size
            (num_series, num_series).
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLPSparse, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)
        self.sparsity = sparsity

        # Set up networks.
        self.networks = []
        for i in range(num_series):
            num_inputs = int(torch.sum(sparsity[i].int()))
            self.networks.append(MLP(num_inputs, lag, hidden, activation))

        # Register parameters.
        param_list = []
        for i in range(num_series):
            param_list += list(self.networks[i].parameters())
        self.param_list = nn.ParameterList(param_list)

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([self.networks[i](X[:, :, self.sparsity[i]])
                          for i in range(self.p)], dim=2)


def prox_update(network, lam, lr, penalty):
    '''
    Perform in place proximal update on first layer weight matrix.

    Args:
      network: MLP network.
      lam: regularization parameter.
      lr: learning rate.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    # 获取第一层卷积核权重，形状为 (hidden_units, 输入变量数 p, 滞后步数 lag)
    W = network.layers[0].weight
    hidden, p, lag = W.shape

    if penalty == 'GL':
        # ---- Group Lasso ----
        # 将每个输入变量 (p) 的所有 hidden × lag 权重作为一组
        # 计算每组的 L2 范数 (1, p, 1)
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        # 对每组做 soft-thresholding（软阈值收缩）
        # 如果某组范数小于 λ*lr，则整组收缩为 0
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        
    elif penalty == 'GSGL':
        # ---- Group Sparse Group Lasso ----
        # 第一步：对每个 (p, lag) 位置单独分组，先做收缩
        norm = torch.norm(W, dim=0, keepdim=True)   # (1, p, lag)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        
        # 第二步：再对每个输入变量 (p)，跨所有 lag 作为一组，再分组做一次收缩 group-threshold
        norm = torch.norm(W, dim=(0, 2), keepdim=True)   # (1, p, 1)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        
    elif penalty == 'H':
        # ---- Hierarchical 层次稀疏 ----
        # 从短滞后到长滞后依次做 group-thresholding
        # 规则：如果远期滞后非零，则近端滞后也必须非零
        for i in range(lag):
            # 计算前 (i+1) 个滞后对应权重的范数
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            # 对前 (i+1) 个滞后权重做收缩更新
            W.data[:, :, :(i+1)] = (
                (W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def regularize(network, lam, penalty):
    '''
    Calculate regularization term for first layer weight matrix.

    Args:
      network: MLP network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def train_model_gista(cmlp, X, lam, lam_ridge, lr, penalty, max_iter,
                      check_every=100, r=0.8, lr_min=1e-8, sigma=0.5,
                      monotone=False, m=10, lr_decay=0.5,
                      begin_line_search=True, switch_tol=1e-3, verbose=1):
    '''
    Train cMLP model with GISTA.

    Args:
      clstm: clstm model.
      X: tensor of data, shape (batch, T, p).
      lam: parameter for nonsmooth regularization.
      lam_ridge: parameter for ridge regularization on output layer.
      lr: learning rate.
      penalty: type of nonsmooth regularization.
      max_iter: max number of GISTA iterations.
      check_every: how frequently to record loss.
      r: for line search.
      lr_min: for line search.
      sigma: for line search.
      monotone: for line search.
      m: for line search.
      lr_decay: for adjusting initial learning rate of line search.
      begin_line_search: whether to begin with line search.
      switch_tol: tolerance for switching to line search.
      verbose: level of verbosity (0, 1, 2).
    '''
    p = cmlp.p
    lag = cmlp.lag
    cmlp_copy = deepcopy(cmlp)
    loss_fn = nn.MSELoss(reduction='mean')
    lr_list = [lr for _ in range(p)]

    # Calculate full loss.
    mse_list = []
    smooth_list = []
    loss_list = []
    for i in range(p):
        net = cmlp.networks[i]
        mse = loss_fn(net(X[:, :-1]), X[:, lag:, i:i+1])
        ridge = ridge_regularize(net, lam_ridge)
        smooth = mse + ridge
        mse_list.append(mse)
        smooth_list.append(smooth)
        with torch.no_grad():
            nonsmooth = regularize(net, lam, penalty)
            loss = smooth + nonsmooth
            loss_list.append(loss)

    # Set up lists for loss and mse.
    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p
    train_loss_list = [loss_mean]
    train_mse_list = [mse_mean]

    # For switching to line search.
    line_search = begin_line_search

    # For line search criterion.
    done = [False for _ in range(p)]
    assert 0 < sigma <= 1
    assert m > 0
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    for it in range(max_iter):
        # Backpropagate errors.
        sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

        # For next iteration.
        new_mse_list = []
        new_smooth_list = []
        new_loss_list = []

        # Perform GISTA step for each network.
        for i in range(p):
            # Skip if network converged.
            if done[i]:
                new_mse_list.append(mse_list[i])
                new_smooth_list.append(smooth_list[i])
                new_loss_list.append(loss_list[i])
                continue

            # Prepare for line search.
            step = False
            lr_it = lr_list[i]
            net = cmlp.networks[i]
            net_copy = cmlp_copy.networks[i]

            while not step:
                # Perform tentative ISTA step.
                for param, temp_param in zip(net.parameters(),
                                             net_copy.parameters()):
                    temp_param.data = param - lr_it * param.grad

                # Proximal update.
                prox_update(net_copy, lam, lr_it, penalty)

                # Check line search criterion.
                mse = loss_fn(net_copy(X[:, :-1]), X[:, lag:, i:i+1])
                ridge = ridge_regularize(net_copy, lam_ridge)
                smooth = mse + ridge
                with torch.no_grad():
                    nonsmooth = regularize(net_copy, lam, penalty)
                    loss = smooth + nonsmooth
                    tol = (0.5 * sigma / lr_it) * sum(
                        [torch.sum((param - temp_param) ** 2)
                         for param, temp_param in
                         zip(net.parameters(), net_copy.parameters())])

                comp = loss_list[i] if monotone else max(last_losses[i])
                if not line_search or (comp - loss) > tol:
                    step = True
                    if verbose > 1:
                        print('Taking step, network i = %d, lr = %f'
                              % (i, lr_it))
                        print('Gap = %f, tol = %f' % (comp - loss, tol))

                    # For next iteration.
                    new_mse_list.append(mse)
                    new_smooth_list.append(smooth)
                    new_loss_list.append(loss)

                    # Adjust initial learning rate.
                    lr_list[i] = (
                        (lr_list[i] ** (1 - lr_decay)) * (lr_it ** lr_decay))

                    if not monotone:
                        if len(last_losses[i]) == m:
                            last_losses[i].pop(0)
                        last_losses[i].append(loss)
                else:
                    # Reduce learning rate.
                    lr_it *= r
                    if lr_it < lr_min:
                        done[i] = True
                        new_mse_list.append(mse_list[i])
                        new_smooth_list.append(smooth_list[i])
                        new_loss_list.append(loss_list[i])
                        if verbose > 0:
                            print('Network %d converged' % (i + 1))
                        break

            # Clean up.
            net.zero_grad()

            if step:
                # Swap network parameters.
                cmlp.networks[i], cmlp_copy.networks[i] = net_copy, net

        # For next iteration.
        mse_list = new_mse_list
        smooth_list = new_smooth_list
        loss_list = new_loss_list

        # Check if all networks have converged.
        if sum(done) == p:
            if verbose > 0:
                print('Done at iteration = %d' % (it + 1))
            break

        # Check progress.
        if (it + 1) % check_every == 0:
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
                ridge_mean = (sum(smooth_list) - sum(mse_list)) / p
                nonsmooth_mean = (sum(loss_list) - sum(smooth_list)) / p

            train_loss_list.append(loss_mean)
            train_mse_list.append(mse_mean)

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Total loss = %f' % loss_mean)
                print('MSE = %f, Ridge = %f, Nonsmooth = %f'
                      % (mse_mean, ridge_mean, nonsmooth_mean))
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cmlp.GC().float())))

            # Check whether loss has increased.
            if not line_search:
                if train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                    line_search = True
                    if verbose > 0:
                        print('Switching to line search')

    return train_loss_list, train_mse_list


def train_model_adam(cmlp, X, lr, max_iter, lam=0, lam_ridge=0, penalty='H',
                     lookback=5, check_every=100, verbose=1):
    '''Train model with Adam.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                    for i in range(p)])

        # Add penalty terms.
        if lam > 0:
            loss = loss + sum([regularize(net, lam, penalty)
                               for net in cmlp.networks])
        if lam_ridge > 0:
            loss = loss + sum([ridge_regularize(net, lam_ridge)
                               for net in cmlp.networks])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        cmlp.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    return train_loss_list


# def train_model_ista(cmlp, X, lr, max_iter, lam=0, lam_ridge=0, penalty='H',
#                      lookback=5, check_every=100, verbose=1):
#     '''Train model with Adam.'''
#     lag = cmlp.lag      # 滞后长度
#     p = X.shape[-1]     # 时间序列维度 (变量个数)
#     loss_fn = nn.MSELoss(reduction='mean')   # 均方误差损失
#     train_loss_list = []

#     # ========== 初始化早停参数 ==========
#     best_it = 0
#     best_loss = np.inf  # 历史最佳损失 (初始化为正无穷)
#     best_model = None   # 保存最佳模型 (deepcopy)

#     # ========== 初始损失计算 ==========
#     # loss: 所有子网络的 MSE 之和
#     loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
#                 for i in range(p)])
#     # ridge: 所有子网络的 L2 正则和
#     ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
#     # smooth: 平滑部分 (MSE + ridge)
#     smooth = loss + ridge

#     # ========== 迭代训练 ==========
#     for it in range(max_iter):
#         # ---- 1. 计算梯度并做梯度步 ----
#         smooth.backward()
#         for param in cmlp.parameters():
#             param.data = param - lr * param.grad

#         # ---- 2. 做 proximal 步 (处理非平滑正则) ----
#         if lam > 0:
#             for net in cmlp.networks:
#                 prox_update(net, lam, lr, penalty)

#         # ---- 3. 清零梯度 (避免累积) ----
#         cmlp.zero_grad()

#         # ---- 4. 为下次迭代重新计算 smooth loss ----
#         loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
#                     for i in range(p)])
#         ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
#         smooth = loss + ridge

#          # ---- 5. 每 check_every 次检查一次进展 ----
#         if (it + 1) % check_every == 0:
#             # 计算非平滑正则
#             nonsmooth = sum([regularize(net, lam, penalty)
#                              for net in cmlp.networks])
#             # 平均损失 (包含平滑+非平滑, 除以变量数 p)
#             mean_loss = (smooth + nonsmooth) / p
#             # 记录训练损失 (注意 detach 避免梯度跟踪)
#             train_loss_list.append(mean_loss.detach())


#             # ---- 打印日志 ----
#             if verbose > 0:
#                 print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
#                 print('Loss = %f' % mean_loss)
#                 # 打印变量使用率 (GC 矩阵中非零元素的比例)
#                 print('Variable usage = %.2f%%'
#                       % (100 * torch.mean(cmlp.GC().float())))

#             # ---- 早停逻辑 ----
#             if mean_loss < best_loss:
#                 best_loss = mean_loss
#                 best_it = it
#                 best_model = deepcopy(cmlp)
#             elif best_it is not None and (it - best_it) == lookback * check_every:
#                 # 如果在 lookback*check_every 次迭代内没有提升, 早停
#                 if verbose:
#                     print('Stopping early')
#                 break

#     # ========== 恢复为最佳模型参数 ==========
#     restore_parameters(cmlp, best_model)

#     return train_loss_list

def train_model_ista(cmlp, X, lr=1e-3, max_iter=5000, lam=0, lam_ridge=0, penalty='H',
                     lookback=5, check_every=100, verbose=1, max_grad_norm=1.0):
    """
    用 ISTA 训练 cMLP，改进了数值稳定性，避免 Loss 为 NaN。
    
    Args:
        cmlp: cMLP 模型
        X: 输入张量，shape=(batch, T, p)
        lr: 学习率
        max_iter: 最大迭代次数
        lam: 非平滑正则参数
        lam_ridge: Ridge 正则参数
        penalty: 正则类型 ('GL', 'GSGL', 'H')
        lookback: 早停步数
        check_every: 每隔多少步记录一次 loss
        verbose: 打印日志
        max_grad_norm: 梯度裁剪阈值
    Returns:
        train_loss_list: 训练损失列表
    """

    # --- 权重初始化 ---
    for m in cmlp.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    train_loss_list = []

    # 早停参数
    best_it = 0
    best_loss = float('inf')
    best_model = None

    # 初始 loss
    loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1]) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
    smooth = loss + ridge

    for it in range(max_iter):
        # 梯度清零
        cmlp.zero_grad()
        smooth.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(cmlp.parameters(), max_norm=max_grad_norm)

        # ISTA 更新
        with torch.no_grad():
            for param in cmlp.parameters():
                param -= lr * param.grad

        # proximal 更新
        if lam > 0:
            for net in cmlp.networks:
                W = net.layers[0].weight
                # 防止除零
                norm = torch.norm(W, dim=(0, 2), keepdim=True)
                if penalty == 'GL':
                    W.data = ((W / torch.clamp(norm, min=(lr * lam + 1e-8))) *
                              torch.clamp(norm - (lr * lam), min=0.0))
                elif penalty == 'GSGL':
                    norm1 = torch.norm(W, dim=0, keepdim=True)
                    W.data = ((W / torch.clamp(norm1, min=(lr * lam + 1e-8))) *
                              torch.clamp(norm1 - (lr * lam), min=0.0))
                    norm2 = torch.norm(W, dim=(0, 2), keepdim=True)
                    W.data = ((W / torch.clamp(norm2, min=(lr * lam + 1e-8))) *
                              torch.clamp(norm2 - (lr * lam), min=0.0))
                elif penalty == 'H':
                    hidden, p_tmp, lag_tmp = W.shape
                    for i_lag in range(lag_tmp):
                        norm_h = torch.norm(W[:, :, :(i_lag+1)], dim=(0, 2), keepdim=True)
                        W.data[:, :, :(i_lag+1)] = ((W.data[:, :, :(i_lag+1)] /
                                                     torch.clamp(norm_h, min=(lr * lam + 1e-8))) *
                                                    torch.clamp(norm_h - (lr * lam), min=0.0))
                else:
                    raise ValueError('unsupported penalty: %s' % penalty)

        # 重新计算 smooth loss
        loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1]) for i in range(p)])
        ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
        smooth = loss +  ridge   

        # 每 check_every 次记录
        if (it + 1) % check_every == 0:
            nonsmooth = sum([regularize(net, lam, penalty) for net in cmlp.networks])
            mean_loss = (smooth + nonsmooth) / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(f"{'-'*10}Iter = {it+1}{'-'*10}")
                print(f"Loss = {mean_loss}")
                print('Variable usage = %.2f%%' % (100 * torch.mean(cmlp.GC().float())))

            # 早停
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) >= lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # 恢复最佳模型
    restore_parameters(cmlp, best_model)

    return train_loss_list



def train_unregularized(cmlp, X, lr, max_iter, lookback=5, check_every=100,
                        verbose=1):
    '''Train model with Adam and no regularization.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        pred = cmlp(X[:, :-1])
        loss = sum([loss_fn(pred[:, :, i], X[:, lag:, i]) for i in range(p)])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        cmlp.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    return train_loss_list



def save_training_outputs(train_loss_list, train_mse_list=None, model=None, model_name="cMLP"):
    """
    保存训练日志和最佳模型

    Args:
        train_loss_list: list[Tensor] or list[float]，训练损失
        train_mse_list: list[Tensor] or list[float]，训练 MSE（可选）
        model: nn.Module，最佳模型
        model_name: str，模型名字，用于文件命名
    """
    # 确保目录存在
    log_dir = "./log"
    model_dir = "./result/model"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== 保存日志 =====
    log_path = os.path.join(log_dir, f"{model_name}_trainlog_{timestamp}.csv")
    with open(log_path, "w") as f:
        if train_mse_list is not None:
            f.write("iter,loss,mse\n")
            for i, (loss, mse) in enumerate(zip(train_loss_list, train_mse_list)):
                f.write(f"{(i+1)}, {float(loss)}, {float(mse)}\n")
        else:
            f.write("iter,loss\n")
            for i, loss in enumerate(train_loss_list):
                f.write(f"{(i+1)}, {float(loss)}\n")
    print(f"[INFO] 训练日志已保存到 {log_path}")

    # ===== 保存模型 =====
    if model is not None:
        model_path = os.path.join(model_dir, f"{model_name}_best_{timestamp}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] 最优模型已保存到 {model_path}")