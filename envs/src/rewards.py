from dataclasses import dataclass
import math
from math import inf
from typing import Any, Callable, Dict, List, Union

import numpy as np


@dataclass
class BoundedReward(object):
    name: str
    """名称"""
    low: float
    """单步奖励下界(越紧越好)"""
    high: float
    """单步奖励上界(越紧越好)"""
    func: Callable[..., float] = lambda *args, **kwargs: 0
    """奖励函数"""
    gamma: float = 0.99
    """折扣因子, 范围 [0,1], 决定价值函数界的估计"""
    steps_tol: int = 100
    """系统经过这个步数后必然进入终止状态, 决定价值函数界的估计"""
    is_terminal: bool = False
    """是否终局奖励"""
    weight: float = 1
    """权重"""
    priori_returns_low: float = -inf
    """回报函数下界(先验)"""
    priori_returns_high: float = inf
    """回报函数上界(先验)"""

    def __check(self):
        objname = f"{type(self).__name__}(name={self.name})"
        etc = ["priori_returns_low", "priori_returns_high"]
        for attrn, attrv in self.__dict__.items():
            if isinstance(attrv, float) and attrn not in etc:
                assert math.isfinite(attrv), f"{objname}.{attrn} is not finite"
        assert self.low <= self.high, Exception(
            objname, "expected low<=high, but get", (self.low, self.high)
        )
        assert 0 <= self.gamma <= 1, (
            f"expected {objname}.gamma in [0,1], but get",
            self.gamma,
        )
        steps_tol = self.steps_tol
        assert steps_tol > 0 and (
            isinstance(steps_tol, int)
        ), f"{objname}.horizon must be positve integer"
        assert self.priori_returns_low <= self.priori_returns_high, (
            f"{objname} expected returns_low<=returns_high, but get",
            (
                self.priori_returns_low,
                self.priori_returns_high,
            ),
        )
        return

    def vf_bounds(self):
        r"""
        重新估计回报函数界
        注意,min/max 不是仿射算子, 记 G_{0,i}:\omega\mapsto\sum_{t=1}^T \gamma^{t-1} r_t
        \sum_i w_i min_\omega G_{0,i}(\omega)
        \leq min_\omega \sum_i w_i G_{0,i}(\omega)
        \leq \sum_i w_i G_{0,i}(\omega)
        \leq max_\omega \sum_i w_i G_{0,i}(\omega)
        \sum_i w_i max_\omega G_{0,i}(\omega)
        """
        self.__check()
        gamma = self.gamma
        steps_tol = self.steps_tol
        gN = math.pow(gamma, steps_tol)
        if gamma < 1:
            gSum2N = (1 - gN) / (1 - gamma)
        else:
            gSum2N = steps_tol

        # 价值函数界估计
        rew_lb = self.low
        rew_ub = self.high
        if self.is_terminal:  # 终局奖励
            v_lb = (gN if rew_lb >= 0 else 1) * rew_lb
            v_ub = (1 if rew_ub >= 0 else gN) * rew_ub
        else:  # 非终局奖励
            v_lb = (1 if rew_lb >= 0 else gSum2N) * rew_lb
            v_ub = (gSum2N if rew_ub >= 0 else 1) * rew_ub
        v_lb = max(v_lb, self.priori_returns_low)
        v_ub = min(v_ub, self.priori_returns_high)
        assert v_lb <= v_ub
        return v_lb, v_ub

    def df_dict(self) -> Dict[str, Any]:
        """转为结构化数据表的字典"""
        d = {k: v for k, v in self.__dict__.items() if not callable(v)}
        return d

    def __str__(self):
        msg = []
        for attrn, attrv in self.__dict__.items():
            if callable(attrv):
                continue
            attrvstr = attrv
            if isinstance(attrv, float):
                attrvstr = f"{attrv:+0.04g}"
            elif isinstance(attrv, str):
                attrvstr = "'" + attrv + "'"
            msg.append(f"{attrn}={attrvstr}")
        v_lb, v_ub = self.vf_bounds()
        msg.append(f"vf_low={v_lb:+0.04g}")
        msg.append(f"vf_high={v_ub:+0.04g}")
        return ",".join(msg)
