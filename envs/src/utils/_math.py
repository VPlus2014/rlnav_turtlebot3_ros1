from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Deque, Sequence
import numpy as np
from numpy.typing import NDArray


def mod_range(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, dtype=np.float_
) -> NDArray[np.floating]:
    np.broadcast()
    a = np.asarray(a, dtype=dtype)
    b = b - a
    x = x - a
    x = np.mod(x, b, where=b != 0)
    x = (a + x).astype(dtype)
    return x


def affine_comb(
    a: np.ndarray, b: np.ndarray, t: np.ndarray, dtype=np.float_
) -> NDArray[np.floating]:
    """计算仿射组合 (1-t)*a+t*b"""
    return (a + np.multiply(t, np.subtract(b, a))).astype(dtype)


class SmoothAve(object):
    def __init__(self, smooth=0.618, maxlen=64):
        assert smooth is not None
        self.__data: Deque[float] = deque(maxlen=maxlen)
        self.reset(smooth)
        return

    def reset(self, smooth: float = None):
        assert math.isfinite(smooth)

        self.__data.clear()
        self.count = 0  # 样本容量
        self.mean = 0.0  # 滑动平均
        self.last = 0.0  # 上一样本
        self.smooth = float(smooth)
        self._w = 1 - self.smooth
        return

    def append(self, val: float):
        val = float(val)
        assert math.isfinite(val)
        if self.count == 0:
            self.mean = self.last = val
        else:
            self.mean += self._w * (val - self.mean)
        self.__data.append(val)
        self.last = val
        self.count += 1

    def __str__(self):
        return (
            "{"
            + f"mean:{self.mean:0.6f},last:{self.last:0.6f},count:{self.count}"
            + "}"
        )


def rewards2gain(rewards: Sequence[float], gamma: float, pow_start=0):
    gain = float(np.sum(rewards))
    n = np.shape(rewards)[0]
    gammas = np.logspace(pow_start, pow_start + n - 1, n, base=gamma)
    gain_d = float(np.dot(rewards, gammas))
    return gain, gain_d


def init_seed(
    seed: int = None,
    init_numpy=True,
    init_torch=True,
):
    if seed is None:
        seed = random.randint(0, 0x7FFFFFFF)
    random.seed(seed)

    if init_numpy:
        np.random.seed(seed)
    if init_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception as e:
            print(e)

    return seed


# @dataclass
# class CircleF(object):
#     x: float = 0.0
#     y: float = 0.0
#     r: float = 0.0


# @dataclass
# class RectF(object):
#     left: float = 0.0
#     top: float = 0.0
#     width: float = 0.0
#     height: float = 0.0

#     @property
#     def right(self):
#         assert self.width >= 0, f"{self}.width is negative"
#         return self.left + self.width

#     @property
#     def bottom(self):
#         assert self.height >= 0, f"{self}.height is negative"
#         return self.top + self.height


# def dist1D(x: float, left: float, right: float) -> float:
#     """计算x到闭区间[left,right]的距离"""
#     if left > right:
#         right, left = left, right
#     if x < left:
#         return left - x
#     if x > right:
#         return x - right
#     return 0


# def dist2D(x: float, y: float, rect: RectF) -> float:
#     """计算x到闭区间[left,right]的距离"""
#     dx = dist1D(x, rect.left, rect.right)
#     dy = dist1D(y, rect.top, rect.bottom)
#     return math.hypot(dx, dy)


# def collide1D(a1: float, b1: float, a2: float, b2: float):
#     """判断闭区间[a1,b1],[a2,b2]是否相交"""
#     if a1 > b1:
#         a1, b1 = b1, a1
#     if a2 > b2:
#         a2, b2 = b2, a2
#     return a1 <= b2 and a2 <= b1


# def collideRectCircle(rect: RectF, circle: CircleF):
#     """判断闭矩形和闭圆是否相交"""
#     r = circle.r
#     assert r >= 0
#     # 计算欧氏投影距离
#     d = dist2D(circle.x, circle.y, rect)
#     return d <= r


# def collideRect(r1: RectF, r2: RectF):
#     """判断闭矩形是否相交"""
#     cx = collide1D(r1.left, r1.right, r2.left, r2.right)
#     cy = collide1D(r1.top, r1.bottom, r2.top, r2.bottom)
#     return cx and cy
