import numpy as np
from scipy.integrate import odeint


def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]           # 变量数 p
    lag = beta.shape[1] // p    # 滞后阶数 L

    # 构造伴随矩阵的下半部分，用于 shift 滞后向量
    # 维度: (p*(lag-1)) × (p*lag)
    bottom = np.hstack(
        (np.eye(p * (lag - 1)),      # 左边是单位阵块
         np.zeros((p * (lag - 1), p))# 右边补零
         ))
    
    # 拼接得到完整的伴随矩阵 (p*lag × p*lag)
    beta_tilde = np.vstack((beta, bottom))
    # 计算伴随矩阵的特征值，取最大模
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))

    # 判断是否非平稳（最大特征值超出阈值）
    nonstationary = max_eig > radius
    if nonstationary:
         # 如果不平稳，则整体缩小系数，递归检查直到平稳
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta


def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # 初始化 Granger 因果矩阵和基础系数矩阵（自回归对角线先设为 beta_value）
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value
    # 计算每行要增加的非零项数量（除自身外）
    num_nonzero = max(0, int(p*sparsity)-1)
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1
    # 横向拼接 lag 份相同的系数矩阵，得到 p × (p*lag) 的 beta
    beta = np.hstack([beta for _ in range(lag)])    # 仍为每个滞后复制同一组系数
    beta = make_var_stationary(beta)                # 调整系数，确保系统平稳

    # Generate data.
    burn_in = 100   # burn-in 步数（用于消除初始条件的影响）
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))  # 高斯白噪声误差项
    X = np.zeros((p, T + burn_in))  # 存放模拟的序列
    X[:, :lag] = errors[:, :lag]    # 初始化前 lag 步为随机噪声

    # 主循环：生成序列
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += + errors[:, t-1]

    return X.T[burn_in:], beta, GC


def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC
