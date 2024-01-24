from functools import partial
import logging
import math
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import Schedule
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter

from dataclasses import dataclass
import rospy
import envs
import os


PYTORCH_MODEL_EXT = ".pt"
__DIRNAME = os.path.dirname(__file__)


class RM_scheduler_maker(object):
    def __init__(
        self,
        initial_value: float,
        init_n: Union[int, float] = 2,
    ) -> None:
        assert initial_value > 0
        assert init_n >= 1
        self.__initial_value = float(initial_value)
        self.__init_n = init_n
        self.reset()

    def reset(self):
        self.__n = 0
        self.__update()
        return self.__schedule

    def step(self) -> Schedule:
        self.__n += 1
        self.__update()
        return self.__schedule

    @property
    def lr(self):
        return self.__r

    def __update(self):
        self.__r = 1 / (1 + self.__n / self.__init_n) * self.__initial_value

    def __schedule(self, progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return self.__r * progress_remaining


def linearScheduler(init_value: float, term_value: float = 0):
    span = init_value - term_value

    def __schedule(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return term_value + progress_remaining * span

    return __schedule


def algo_get_model_name(model_fname: str):
    fn = os.path.splitext(model_fname)[0]
    return fn + PYTORCH_MODEL_EXT


def algo_train(
    algoname="ppo",
    disc_action_size=3,  # 离散动作空间大小(不要是偶数,不然没法表示零控制量)
    seed: Optional[int] = None,
    #
    epochs=50,  # 轮数,每轮依次执行一次 训练,验证,保存模型
    trn_steps=10000,  # 每轮训练交互步数
    eval_episodes=100,  # 每轮验证局数
    batch_size=64,
    max_episode_steps=int(30 / 0.2),  # 每局最多决策几步,用于截断对局时长
    gamma=1 - 1e-2,
    #
    learning_rate=1e-4,
    algo_model_name="test_params.pt",  # 预训练模型文件名
    device: torch.device = "cuda",
    #
    robot_model_name="waffle",
    launch_fname="turtlebot3_stage4server_3.launch",
    gui=True,
    use_obs_norm=True,
    use_rew_norm=True,
):
    #
    #
    seed = envs.utils.init_seed(seed=seed)

    use_train = trn_steps > 0
    use_eval = eval_episodes > 0
    algoname_ = algoname.lower()

    # 设置调试信息
    runs_dir = os.path.join(__DIRNAME, "runs")  # 任务工作区

    task_time_stamp = envs.utils.datetime2str(fmt="%Y%m%d_%H%M%S")  # 任务启动时间戳
    task_dir = os.path.join(runs_dir, f"{algoname_}_{task_time_stamp}")

    main_log_fn = os.path.join(task_dir, "main.log")
    main_logger = logging.getLogger(os.path.split(__DIRNAME)[1]).getChild("train")
    envs.utils.logger_quickReset(
        main_logger.name, level=logging.INFO, log_fn=main_log_fn, to_stdout=True
    )
    main_logger.info(f"batch_size:{batch_size}")
    main_logger.info(f"max_episode_steps{max_episode_steps}")

    # 日志目录
    envs_log_dir = os.path.join(task_dir)
    trn_log_dir = os.path.join(task_dir, "log_train")
    eval_log_dir = os.path.join(task_dir, "log_eval")
    models_dir = os.path.join(task_dir, "saved_models")
    for _ in [envs_log_dir, trn_log_dir, eval_log_dir]:
        os.makedirs(_, exist_ok=True)

    # 根据算法选择动作空间类型
    if algoname_ in ["ppo", "dqn"]:
        disc_action = True
    elif algoname_ in ["sac", "ddpg", "td3"]:
        disc_action = False
    else:
        raise NotImplementedError(f"unsupported algoname '{algoname}'")

    # 设置 gym 仿真环境
    kwargs4env = dict(
        config=envs.gymenv_core.Env_stage_core_init_args(
            launch_fname=launch_fname,
            launch_args=envs.gymenv_core.Turtlebot3_world_launch_args(
                y_pos=0.6,
                model=robot_model_name,
                paused=True,
                gui=gui,
                headless=not gui,
                debug=False,
            ),
            horizon=max_episode_steps,
            gamma=gamma,
            log_dir=envs_log_dir,
        ),
        seed=seed,
    )
    if disc_action:
        kwargs4env["action_size"] = disc_action_size
        env = envs.gym_.make(
            envs.gym_register.GymEnvID_stage_3_disc_v1,
            max_episode_steps=max_episode_steps,
            **kwargs4env,
        )
    else:
        env = envs.gym_.make(
            envs.gym_register.GymEnvID_stage_3_cont_v1,
            max_episode_steps=max_episode_steps,
            **kwargs4env,
        )

    from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward

    if use_obs_norm:
        env = NormalizeObservation(env)
    main_logger.info(f"use NormalizeObservation={use_obs_norm}")
    if use_rew_norm:
        env = NormalizeReward(env, gamma=gamma)
    main_logger.info(f"use NormalizeReward={use_obs_norm},gamma={gamma}")

    scheduler = linearScheduler(learning_rate, learning_rate * 1e-1)
    scheduler = learning_rate

    # 设置算法
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    kwargs4algo = dict(
        policy="MlpPolicy",
        env=env,
        batch_size=batch_size,
        learning_rate=scheduler,
        verbose=1,
        device=device,
        seed=seed,
        gamma=gamma,
    )
    if algoname_ == "ppo":
        algo_cls = sb3.PPO
        kwargs4algo["n_steps"] = max_episode_steps
    elif algoname_ == "sac":
        algo_cls = sb3.SAC
    elif algoname_ == "td3":
        algo_cls = sb3.TD3
    elif algoname_ == "ddpg":
        algo_cls = sb3.DDPG
    elif algoname_ == "dqn":
        algo_cls = sb3.DQN
    else:
        raise NotImplementedError
    algo = algo_cls(**kwargs4algo)

    # 日志记录器, 这两个只记录表格数据
    trn_sb3logger = sb3_logger.configure(trn_log_dir, ["stdout", "csv", "tensorboard"])
    eval_sb3logger = sb3_logger.configure(
        eval_log_dir, ["stdout", "csv", "tensorboard"]
    )
    algo.set_logger(trn_sb3logger)

    # 加载预训练模型
    if algo_model_name:
        algo_model_name = algo_get_model_name(algo_model_name)
        model_in_fn = envs.utils.path_find_with_interaction(
            root=runs_dir, posfix=algo_model_name
        )
    else:
        model_in_fn = ""
    if model_in_fn and os.path.exists(model_in_fn):
        try:
            algo.load(model_in_fn, device=device)
            main_logger.info(f"[{algoname_}] model<<'{model_in_fn}'")
        except Exception as e:
            traceback.print_exc()
            main_logger.info(f"[{algoname_}] failed to load '{model_in_fn}'")
    else:
        main_logger.warning(f"[{algoname_}] no such model file '{algo_model_name}'")
        pass

    env = Monitor(env)
    try:
        for epoch in range(1, 1 + epochs):
            # 训练
            if use_train:
                main_logger.info(f"[{algoname_}] training {epoch}/{epochs}")
                algo.learn(
                    total_timesteps=trn_steps,
                    reset_num_timesteps=False,
                )

                # 保存
                os.makedirs(models_dir, exist_ok=True)
                save_ts = envs.utils.datetime2str(fmt="%Y%m%d_%H%M%S")
                model_out_fn = f"{algoname_}_{task_time_stamp}_E{epoch}_{save_ts}"
                model_out_fn = os.path.join(
                    models_dir,
                    algo_get_model_name(model_out_fn),
                )
                algo.save(model_out_fn)
                main_logger.info(f"[{algoname_}] model>>{model_out_fn}")

            if use_eval:
                logger = eval_sb3logger
                logger.info(f"[{algoname_}] evaluating {epoch}/{epochs}")
                # 验证
                episode_rewards, episode_lengths = evaluate_policy(
                    algo,
                    env,
                    n_eval_episodes=eval_episodes,
                    deterministic=True,
                    return_episode_rewards=True,
                )
                tmp_meta: Dict[str, List[float]] = {
                    "eval/episode_rewards": episode_rewards,
                    "eval/episode_lengths": episode_lengths,
                }
                for itemname, vals in tmp_meta.items():
                    logger.record(f"{itemname}/mean", np.mean(vals))
                    logger.record(f"{itemname}/std", np.std(vals))
                logger.dump(epoch)
                pass

    except Exception as e:
        traceback.print_exc()
        pass

    trn_sb3logger.close()  # 关闭日志输出
    eval_sb3logger.close()
    env.close()  # 关闭仿真
    main_logger.info("train end")
    return


def dataclass2str(obj: object):
    expr = ",".join(f"{atn}={atv}" for atn, atv in obj.__dict__.items())
    expr = f"{type(obj).__name__}({expr})"
    return expr


def main():
    run_as_node = __name__ == "__main__"

    if run_as_node:
        envs.bash.ros_restart()
        node_id = envs.utils.ros_valid_node_id(__file__)
        rospy.init_node(node_id)
    # 自定义参数
    max_episode_steps_secs = 20.0  # 单局仿真时长(s)
    trn_total_hours = 4.0  # 预计训练总时长(h)
    trn_steps_mins = 30.0  # 单轮训练时长(m)
    #
    env_fps = 5  # 实测帧率
    batch_size_base = 4
    epochs = max(round(trn_total_hours * 60 / trn_steps_mins), 1)
    trn_steps = int(env_fps * (60 * trn_steps_mins))
    max_episode_steps = (
        max(int(max_episode_steps_secs * env_fps) // batch_size_base, 1)
        * batch_size_base
    )
    batch_size = max_episode_steps
    print("batch_size", batch_size)
    print("max_episode_steps", max_episode_steps)
    print("max_episode_steps in sec", max_episode_steps * env_fps)
    assert batch_size <= 4096, ("batch_size too large", batch_size)
    algo_train(
        algoname="ppo",
        robot_model_name="waffle",
        algo_model_name="ppo_20240120_045724_E15_20240120_103145",
        gamma=1 - 1e-3,
        seed=8,
        epochs=epochs,
        trn_steps=trn_steps,
        eval_episodes=10,
        learning_rate=2e-4,
        batch_size=batch_size,
        max_episode_steps=max_episode_steps,
        launch_fname="turtlebot3_stage4server_4.launch",
        gui=True,
        use_obs_norm=True,
        use_rew_norm=True,
    )
    #

    if run_as_node:
        rospy.signal_shutdown("end")
        envs.bash.killall()
    return


if __name__ == "__main__":
    main()
