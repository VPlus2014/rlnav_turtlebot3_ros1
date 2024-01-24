from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    batch_size: int = 32
    trn_toal_hours: int = 12  # 预计训练总时长(h)
    trn_steps_hours: int = 20  # 单轮训练时长(m)

    task_secs_tol = 60.0  # 搜索任务的时限,超时则失败
    max_episode_steps_secs = 20.0  # 单局仿真时限(s),不是任务总时限
    trn_total_hours = 12.0  # 预计训练总时长(h)
    trn_steps_mins = 30.0  # 单轮训练时长(m)

    algoname: str
    algo_model_name: str  # 算法预训练模型
    gamma: float = 1 - 1e-2
    seed: Optional[int] = None
    robot_model_name: str  # 机器人模型
    env_stage: int
