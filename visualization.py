import shutil
import matplotlib
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from typing import Dict, Sequence, List, Union


def plt_font_download():

    # 设置字体文件的URL和目标文件路径
    font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
    target_font_path = os.path.join(
        matplotlib.get_data_path(), "fonts/ttf/SimHei.ttf")

    # 下载字体文件
    urllib.request.urlretrieve(font_url, target_font_path)

    # 找到matplotlibrc文件的路径
    matplotlibrc_path = matplotlib.matplotlib_fname()
    # 获取matplotlibrc文件所在的文件夹路径
    matplotlibrc_dir = os.path.dirname(matplotlibrc_path)

    # 找到fonts/ttf目录的路径
    fonts_dir = os.path.join(matplotlib.get_data_path(), "fonts/ttf")

    # 删除缓存文件夹中的内容
    cache_dir = matplotlib.get_cachedir()
    shutil.rmtree(cache_dir, ignore_errors=True)

    print("字体文件已下载到:", target_font_path)
    print("matplotlibrc文件所在目录:", matplotlibrc_dir)
    print("删除缓存文件夹:", cache_dir)


def plt_init():
    """
    linux 下找不到字体导致中文乱码的解决方法
    https://zhuanlan.zhihu.com/p/449589031?utm_id=0&wd=&eqid=ba9336d3000002100000000664814425
    """
    # import matplotlib
    # # print("font", matplotlib.matplotlib_fname())
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def cvrt_smooth(xs: np.ndarray, weight=0.75):
    """
    平滑处理，类似tensorboard的smoothing函数。
    returns:
        xs_smoothed:与 xs 长度一致的滑动平均数组
    """
    assert len(np.shape(xs)) == 1
    xs = np.ravel(xs)
    n = xs.shape[0]
    xs_smoothed = np.empty(n, float)
    xs_smoothed[0] = x_smoothed = xs[0]
    for i in range(1, n):
        x = xs[i]
        x_smoothed = x+weight*(x_smoothed-x)
        xs_smoothed[i] = x_smoothed
    return xs_smoothed


def algos_returns2df(
    xs: np.ndarray,
    ys_origin: Dict[str, np.ndarray],
    xlabel="episode",
    ylabel="returns",
    algo_colname="algoname",  # 新增1列用于区分不同算法
    smooth_weight=0.99,  # 平滑系数,越大曲线越平滑
):
    """
    将一组算法训练数据转为df
    origin_returns:键为算法名,不建议使用中文;值为同一算法下训练的累计回报数组;
    """
    assert isinstance(xs, np.ndarray)
    assert len(xs.shape) == 1
    n = xs.shape[0]
    assert len(ys_origin) > 0  # 算法结果非空
    assert n > 0  # 长度检查
    data = ys_origin.values()
    assert all([isinstance(vs, np.ndarray)for vs in data])  # 类型一致性检查
    assert all([len(vs.shape) for vs in data])  # 维度一致性检查,只支持一维数组
    assert all([n == vs.shape[0] for vs in ys_origin.values()])  # 长度一致性检查

    df = []
    for i_algo, kv in enumerate(ys_origin.items()):
        algo_name, returns_orgin = kv
        returns_smoothed = cvrt_smooth(returns_orgin, weight=smooth_weight)
        df_i1 = pd.DataFrame({
            xlabel: xs,
            ylabel: returns_orgin,
            algo_colname: algo_name})
        df_i2 = pd.DataFrame({
            xlabel: xs,
            ylabel: returns_smoothed,
            algo_colname: algo_name})
        df_i = pd.concat((df_i1, df_i2))  # 合并的顺序不能调换
        df.append(df_i)
        # print(df_i)

    df = pd.concat(df)
    return df


def rl_returns2fig(
    episode_steps: Sequence[Union[int, float]] = None,  # 对局编号列
    algos_returns: Dict[str, np.ndarray] = None,  # 算法
    xlabel="Episode",  # X轴标签
    ylabel="Returns",  # Y轴标签
    algo_colname="Algorithm",  # 算法分类legends标题
    smooth_weight=0.75,  # 平滑系数
    font_size=18,  # 字体大小
    title: str = "RL-Algorithms",  # 绘图区标题
    fig_out_filename: str = None,  # 图像输出文件名,若路径不合法则不输出
):
    """
    将一组算法的训练结果输入,创建一个新窗口绘图
    origin_returns:键为算法名,不建议使用中文;值为同一算法下训练的累计回报数组;
    """
    if algos_returns is None:  # 演示程序
        n = 100
        gen_bias = 0.05
        gen_gain = 5e-1

        def demo_make_line(
            n: int = n,
            bias: float = gen_bias,
            gain: float = gen_gain,
            gain2: float = 0.2
        ):
            dxs = np.random.rand(n)
            xs = np.zeros_like(dxs)
            x = xs[0]
            for i, dx in enumerate(dxs):
                x = xs[i] = x+gain*(dx+bias)
            return 1-np.exp(-xs)+dxs*gain2

        algos_returns = dict([
            ("PPO", demo_make_line()),
            ("DDPG", demo_make_line()),
        ])
    else:
        # 获取单个算法的实验数据长度
        n = np.prod(np.shape(algos_returns.values[0]))
        pass

    if episode_steps is None:
        episode_steps = np.arange(n)  # 默认的时间步

    # 数据预处理,将原始数据按算法分类制成3列表格
    df = algos_returns2df(
        xs=episode_steps,
        ys_origin=algos_returns,
        ylabel=ylabel,
        xlabel=xlabel, algo_colname=algo_colname,
        smooth_weight=smooth_weight)
    # print(df)

    # 开始绘图
    plt.rcParams['font.size'] = font_size  # 字体大小设置

    fig = plt.figure(dpi=300)  # 新建窗口
    ax = fig.gca()  # 设置绘图区
    sns.lineplot(df, x=xlabel, y=ylabel, hue=algo_colname, ax=ax)  # 绘图
    ax.set_title(title)  # 设置标题
    fig.tight_layout()  # 重新紧凑布局

    # 输出图像到文件
    if fig_out_filename:
        try:
            fig.savefig(fig_out_filename)
            print("fig>>", fig_out_filename)
        except:
            pass
    return fig


if __name__ == "__main__":
    fig = rl_returns2fig()
    plt.show()
    pass
