# 生成 TC-cMLP 论文示例图片（图 1 为方法示意，图 2-4 为模拟数据示例，300 dpi）
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
})


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", path)


def example_tag(fig):
    # 标注该图为模拟数据示例，避免与真实实验输出混淆
    fig.text(0.995, 0.005, "Example figure — simulated data", ha="right", va="bottom",
             fontsize=7, color="0.45", style="italic")


# ---------------------------------------------------------------- 图 1 方法框架
def fig1_framework():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, fc="#eef3fb", ec="#31527a", lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc=fc, ec=ec, lw=lw, zorder=2))

    def arrow(p0, p1, color="#31527a", ls="-", lw=1.3):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                     color=color, linestyle=ls, lw=lw, zorder=1,
                                     shrinkA=0, shrinkB=0))

    def note(text, xy, xytext):
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=8.5, color="#7a4a00",
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color="#b07800", lw=1.1,
                                    linestyle="--", shrinkA=2, shrinkB=2))

    # A: 多通道 iEEG
    box(2, 16, 14, 14)
    ax.text(9, 27.5, "Multi-channel", ha="center", va="center", fontsize=9)
    ax.text(9, 25.3, "iEEG", ha="center", va="center", fontsize=9)
    tt = np.linspace(0, 1, 220)
    sig = 0.55*np.sin(2*np.pi*7*tt) + 0.3*np.sin(2*np.pi*17*tt + 1.2) + 0.15*np.sin(2*np.pi*31*tt)
    ax.plot(2.9 + tt*12.2, 20.2 + sig*1.7, color="#31527a", lw=0.8, zorder=3)

    # B: 重叠滑动窗口
    box(21, 16, 17, 14)
    ax.text(29.5, 27.5, "Overlapping sliding", ha="center", va="center", fontsize=9)
    ax.text(29.5, 25.3, "windows (5 s, 80%)", ha="center", va="center", fontsize=9)
    for i in range(4):
        ax.add_patch(Rectangle((23.2 + i*3.1, 17.8), 5.6, 3.4, fill=False,
                               ec="#31527a", lw=0.9, alpha=0.45 + 0.18*i, zorder=3))

    # C: 窗口专属第一层（容器 + 三个子框）
    box(43, 6, 20, 34, fc="#f7f9fd")
    ax.text(53, 42.6, "Window-specific 1st layers", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
    for (yy, lab) in [(29.5, r"$W_i^{(1,1)}$"), (22.0, r"$W_i^{(1,2)}$"), (7.5, r"$W_i^{(1,W)}$")]:
        box(45.5, yy, 15, 6, fc="white")
        ax.text(53, yy + 3, lab, ha="center", va="center", fontsize=9.5)
    ax.text(53, 18.6, "\u22ee", ha="center", va="center", fontsize=13, color="#31527a")
    ax.text(53, 15.4, "\u22ee", ha="center", va="center", fontsize=13, color="#31527a")

    # D: 共享后续层
    box(69, 16, 16, 14)
    ax.text(77, 25.6, "Shared layers", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(77, 22.6, r"($\Theta_i = W_{2:L,i}$)", ha="center", va="center", fontsize=9)
    for k in range(3):
        ax.add_patch(Rectangle((72 + k*3.4, 17.6), 2.6, 2.6, fill=False,
                               ec="#31527a", lw=0.9, alpha=0.4 + 0.2*k, zorder=3))

    # E: 逐窗口 causal matrix 序列
    ax.text(99.5, 42.6, "Causal matrices", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
    mini = rng.random((6, 6)) * 0.25
    mini[1, 2] = 0.9
    mini[3, 0] = 0.75
    mini[4, 2] = 0.85
    mini2 = mini.copy()
    mini2[2, 3] = 0.8
    mini3 = mini2.copy()
    mini3[0, 4] = 0.7
    for (x0, M, lab) in [(90, mini, r"$C^{(1)}$"), (98, mini2, r"$C^{(2)}$"), (107, mini3, r"$C^{(W)}$")]:
        ax.imshow(M, cmap="magma", extent=(x0, x0 + 6, 27, 33), interpolation="nearest",
                  vmin=0, vmax=1, zorder=3, aspect="auto")
        ax.text(x0 + 3, 25.6, lab, ha="center", va="center", fontsize=9.5)
    ax.text(103.5, 30, "\u2026", ha="center", va="center", fontsize=13)
    # temporal regularization 作用位置：相邻矩阵之间
    ax.plot([96.6, 97.4], [33.8, 33.8], color="#b07800", lw=1.1, ls="--", zorder=4)
    ax.plot([105.6, 106.4], [33.8, 33.8], color="#b07800", lw=1.1, ls="--", zorder=4)

    # F: 动态 outflow
    box(95, 4, 18, 12)
    ax.text(104, 14.2, "Dynamic outflow", ha="center", va="center", fontsize=9.5, fontweight="bold")
    tt2 = np.linspace(0, 1, 80)
    ramp = 1.0 / (1.0 + np.exp(-(tt2 - 0.52) * 18))
    ax.plot(96.5 + tt2 * 15, 7.2 + ramp * 4.2, color="#31527a", lw=1.1, zorder=3)
    ax.plot([103.5, 103.5], [6.0, 12.6], color="#999999", lw=0.8, ls=":", zorder=3)

    # G: SOZ 排名
    box(69, 4, 18, 12)
    ax.text(78, 12.2, "SOZ ranking", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(78, 9.6, "(top 18% channels)", ha="center", va="center", fontsize=8.5)
    for k, frac in enumerate([1.0, 0.86, 0.72, 0.55, 0.4]):
        ax.plot([72.5, 72.5 + 11 * frac], [6.4 + k * 1.15, 6.4 + k * 1.15],
                color="#31527a", lw=1.4 if k < 2 else 1.0,
                alpha=1.0 if k < 2 else 0.55, zorder=3)

    # 主流程箭头
    arrow((16.2, 23), (20.8, 23))
    arrow((38.2, 23), (42.8, 23))
    arrow((63.2, 23), (68.8, 23))
    arrow((85.2, 23), (89.2, 29.5))
    arrow((107, 26.8), (104.5, 16.4))
    arrow((94.8, 10), (87.4, 10))
    # 窗口专属第一层到共享层的连接
    for yy in [32.5, 25.0, 10.5]:
        arrow((60.7, yy), (68.8, 23), lw=1.0, color="#7d9cc4")

    # 正则化标注
    note("Group sparsity", (53, 5.6), (53, 2.2))
    note("Weight decay", (77, 15.8), (77, 12.6))
    note("Temporal\nregularization", (101.5, 34.2), (101.5, 38.2))

    save(fig, "fig1_method_framework.png")


# ---------------------------------------------------------------- 图 2 synthetic
def fig2_synthetic():
    p = 10
    # 真实网络 A：稀疏随机
    A = np.zeros((p, p))
    edges = [(0, 2), (0, 5), (1, 3), (1, 6), (2, 4), (3, 5), (4, 6),
             (5, 8), (6, 9), (2, 7), (7, 9), (1, 8)]
    for (i, j) in edges:
        A[i, j] = rng.uniform(0.45, 1.0)
    np.fill_diagonal(A, 0.0)
    # 网络 B：增加连接、删除连接、改变强度
    B = A.copy()
    B[2, 5] = rng.uniform(0.5, 0.9)
    B[8, 0] = rng.uniform(0.5, 0.9)
    B[4, 1] = rng.uniform(0.5, 0.9)
    B[3, 5] = 0.0
    B[6, 9] = 0.0
    B[0, 2] = A[0, 2] * 0.5
    B[5, 8] = min(1.0, A[5, 8] * 1.4)

    # 两种方法在第 5 个窗口（网络 A 阶段）的估计
    est_cmlp = np.clip(A + 0.28 * rng.standard_normal(A.shape), 0, None)
    est_tc = np.clip(A + 0.06 * rng.standard_normal(A.shape), 0, None)

    # 相邻窗口变化量 D(w)
    w = np.arange(2, 21)
    d_cmlp = 0.85 + 0.28 * rng.standard_normal(19)
    d_cmlp[9] += 0.45
    d_wo = 0.26 + 0.07 * rng.standard_normal(19)
    d_wo[9] += 1.05
    d_tc = 0.055 + 0.015 * rng.standard_normal(19)
    d_tc[9] += 2.3
    d_tc[10] += 0.35
    d_cmlp = np.clip(d_cmlp, 0, None)
    d_wo = np.clip(d_wo, 0, None)
    d_tc = np.clip(d_tc, 0, None)

    fig = plt.figure(figsize=(11, 9.8))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.92],
                  hspace=0.5, wspace=0.28)

    def mat_ax(row, col, M, title, cbar=False):
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=1.15, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("source channel")
        ax.set_ylabel("target channel")
        if cbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="causal score")
        return ax

    mat_ax(0, 0, A, "(a) True network A (windows 1\u201310)")
    mat_ax(0, 1, B, "(a) True network B (windows 11\u201320)", cbar=True)
    mat_ax(1, 0, est_cmlp, "(b) cMLP estimate, window 5")
    mat_ax(1, 1, est_tc, "(c) TC-cMLP estimate, window 5", cbar=True)

    axd = fig.add_subplot(gs[2, :])
    axd.plot(w, d_cmlp, "o-", color="#7f7f7f", ms=4, lw=1.2, label="cMLP (per-window)")
    axd.plot(w, d_wo, "s-", color="#4a7bb7", ms=4, lw=1.2,
             label="TC-cMLP w/o temporal reg.")
    axd.plot(w, d_tc, "D-", color="#c4453c", ms=4, lw=1.4, label="TC-cMLP")
    axd.axvline(10.5, color="k", lw=1.0, ls="--")
    axd.text(10.8, 2.42, "true change point", fontsize=8.5, va="top")
    axd.set_xlabel("window index $w$")
    axd.set_ylabel(r"$D^{(w)} = \| C^{(w)} - C^{(w-1)} \|_1$")
    axd.set_title("(d) Window-to-window network change", fontsize=10)
    axd.set_xlim(1.4, 20.6)
    axd.legend(fontsize=8.5, frameon=False, loc="upper left")
    axd.grid(alpha=0.25, lw=0.5)

    example_tag(fig)
    save(fig, "fig2_synthetic_recovery.png")


# ---------------------------------------------------------------- 图 3 iEEG 与因果矩阵
def fig3_hup():
    fs = 100
    t_rel = np.arange(-80, 80, 1 / fs)
    n = len(t_rel)
    p = 50
    soz = np.arange(20, 29)  # 9 个 SOZ channel

    # 模拟 50 通道 iEEG：preictal 低幅背景，onset 后 SOZ channel 出现大幅节律放电
    data = np.zeros((p, n))
    smooth = np.ones(5) / 5.0
    for ch in range(p):
        x = np.convolve(rng.standard_normal(n), smooth, mode="same")
        x = x / np.std(x)
        if ch in soz:
            amp = np.where(t_rel < 0, 1.0, 7.0)
            osc = np.sin(2 * np.pi * (8.0 + 0.02 * t_rel) * t_rel)
            data[ch] = amp * (0.6 * x + 0.9 * osc)
        else:
            amp = np.where(t_rel < 5, 1.0, 2.2)
            osc = 0.5 * np.sin(2 * np.pi * (5.0 + 0.05 * ch) * t_rel + ch)
            data[ch] = amp * (x + 0.25 * osc)
    data = data / np.max(np.abs(data)) * 0.9

    # 逐窗口 causal matrix（preictal 与 ictal 代表性窗口）
    def make_matrix(ictal):
        M = rng.random((p, p)) * 0.12
        np.fill_diagonal(M, 0.0)
        if ictal:
            for j in soz:
                rows = rng.random(p) < 0.6
                M[rows, j] = rng.uniform(0.45, 1.0, int(rows.sum()))
            sub = M[np.ix_(soz, soz)] + rng.uniform(0.2, 0.5, (len(soz), len(soz)))
            M[np.ix_(soz, soz)] = np.clip(sub, 0, 1)
        else:
            for j in soz:
                rows = rng.random(p) < 0.25
                M[rows, j] = rng.uniform(0.15, 0.35, int(rows.sum()))
        return M

    M_pre = make_matrix(ictal=False)
    M_ict = make_matrix(ictal=True)

    fig = plt.figure(figsize=(12, 10.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.15],
                  hspace=0.42, wspace=0.3)

    # 上：叠加波形
    axw = fig.add_subplot(gs[0, :])
    dec = 2
    offset = 2.2
    for ch in range(p):
        axw.plot(t_rel[::dec], data[ch, ::dec] * 1.1 + ch * offset,
                 color="#c4453c" if ch in soz else "#31527a",
                 lw=0.32, alpha=0.95 if ch in soz else 0.6, zorder=2 if ch in soz else 1)
    axw.axvline(0, color="k", lw=1.0, ls="--")
    axw.text(1.2, 0.98, "seizure onset", transform=axw.get_xaxis_transform(),
             fontsize=8.5, va="top")
    axw.set_xlabel("Time relative to seizure onset (s)")
    axw.set_ylabel("channel index (stacked)")
    axw.set_title("Simulated multi-channel iEEG (example patient, 50 channels)", fontsize=10)
    axw.set_xlim(-80, 80)
    axw.set_yticks([0, 10, 20, 30, 40, 49])
    handles = [Line2D([], [], color="#c4453c", lw=1.4, label="clinical SOZ channels"),
               Line2D([], [], color="#31527a", lw=1.4, label="other channels")]
    axw.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")

    # 下：两个代表性窗口的 causal matrix
    for col, (M, title) in enumerate([(M_pre, "Preictal window (t \u2248 \u221240 s)"),
                                      (M_ict, "Ictal window (t \u2248 +30 s)")]):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("source channel")
        ax.set_ylabel("target channel")
        # SOZ 列（source 方向）以红色框标出
        ax.add_patch(Rectangle((19.5, -0.5), 9, p, fill=False, ec="#c4453c", lw=1.6, zorder=4))
        if col == 1:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="causal score")
    fig.get_axes()[2].text(28.6, 51.6, "SOZ\ncolumns", color="#c4453c", fontsize=8.5,
                           ha="center", va="bottom")

    example_tag(fig)
    save(fig, "fig3_ieeg_causal_matrices.png")


# ---------------------------------------------------------------- 图 4 动态 outflow
def fig4_outflow():
    t = np.linspace(-78, 77, 156)  # 与 160 s / 5 s 窗口 / 80% overlap 对应
    p = 50
    soz = np.arange(20, 29)

    methods = [
        ("cMLP (per-window)", 0.30, 2.6),
        ("TC-cMLP w/o temporal reg.", 0.10, 2.9),
        ("TC-cMLP", 0.035, 3.3),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    for ax, (name, sigma, lvl) in zip(axes, methods):
        ax.axvspan(0, 80, color="0.87", zorder=0)
        base = rng.uniform(0.35, 0.75, p)
        for ch in range(p):
            if ch in soz:
                pre = rng.uniform(0.7, 1.0)
                ramp = 1.0 / (1.0 + np.exp(-(t - 8.0) / 4.0))
                curve = pre + (lvl - pre) * ramp
            else:
                curve = np.full_like(t, base[ch])
            curve = np.clip(curve + rng.normal(0, sigma, len(t)), 0, None)
            if ch in soz:
                ax.plot(t, curve, color="#c4453c", lw=1.0, alpha=0.9, zorder=3)
            else:
                ax.plot(t, curve, color="#4a7bb7", lw=0.7, alpha=0.35, zorder=2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Time relative to seizure onset (s)")
        ax.set_xlim(-80, 80)
        ax.grid(alpha=0.2, lw=0.5)
    axes[0].set_ylabel("Causal outflow")
    axes[0].set_ylim(0, 4.6)
    handles = [Line2D([], [], color="#c4453c", lw=1.4, label="clinical SOZ channels"),
               Line2D([], [], color="#4a7bb7", lw=1.4, label="other channels"),
               Rectangle((0, 0), 1, 1, fc="0.87", ec="none", label="ictal period")]
    axes[0].legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")

    fig.subplots_adjust(wspace=0.08)
    example_tag(fig)
    save(fig, "fig4_dynamic_outflow.png")


if __name__ == "__main__":
    fig1_framework()
    fig2_synthetic()
    fig3_hup()
    fig4_outflow()
    print("all figures generated.")
