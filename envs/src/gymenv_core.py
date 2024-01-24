import abc
from copy import deepcopy
import os
import sys
import subprocess

import math
from math import pi, inf
from typing import Any

import numpy as np
from numpy.typing import NDArray
import time
import pandas as pd

from numpy import ndarray

from geometry_msgs.msg import Twist, Point, Pose, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import DeleteModel
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import logging
from .respawnGoal import Respawn
from .utils import *
from .robot_description import *
from .rewards import *
from .gazebo_srv import *
import gymnasium as gym_
from stable_baselines3.common import logger as sb3_logger


@dataclass
class EnvInfo(object):
    """
    状态转移中得到的信息
    """

    ang_vel: float = 0.0
    """决策时给出的角速度 rad/s"""

    lin_vel: float = 0.0
    """决策时给出的线速度 rad/s"""

    ang_vel_real: float = 0.0
    """决策时实际的角速度 rad/s"""

    lin_vel_real: float = 0.0
    """决策时实际的线速度 rad/s"""

    pos_x: float = 0.0
    """来自里程计的位置 x 坐标"""

    pos_y: float = 0.0
    """来自里程计的位置 y 坐标"""

    yaw: float = 0.0
    """偏航角 值域 [-pi,pi)"""

    obst_collision: bool = False
    """是否与障碍物碰撞(仅从雷达数据判断)"""

    obst_dist: float = 0.0
    """最近的障碍物的距离"""

    obst_angle_idx: int = 0
    """最近的障碍物所在方位(对应雷达编号)"""

    obst_angle: float = 0.0
    """最近的障碍物所在方位(rad),值域 [0,2pi)]"""

    obst_too_close: bool = False
    """距离障碍物是否过近"""

    goal_get: bool = False
    """是否抵达 状态转移前目标 """

    goal_dist: float = 0.0
    """状态转移后机器人 与 状态转移前目标 的距离"""

    goal_dist_old: float = 0.0
    """状态转移前机器人 与 状态转移前目标 的距离"""

    goal_yaw_err: float = 0.0
    """与 状态转移前目标 的视线角误差,绕z轴右手方向为正,值域 [-pi,pi)"""

    def to_dict(self, copy=True) -> Dict[str, Union[int, bool, float]]:
        d = self.__dict__
        return deepcopy(d) if copy else d

    def from_dict(self, info: Dict[str, Any]):
        for k, v in info.items():
            if k not in self.__dict__.keys():
                continue
            self[k] = type(self[k])(v)

    def __str__(self):
        ss = [f"{k}: {v}" for k, v in self.__dict__.items()]
        return "{" + ("\n".join(ss)) + "}"

    def round(self, ndigits: int):
        for k, v in self.__dict__.items():
            if isinstance(v, float):
                setattr(self, k, round(v, ndigits))


@dataclass
class TurtlebotObservationSpace:
    scan_ranges: gym_.spaces.Box
    goal_yaw_err: gym_.spaces.Box
    goal_dist: gym_.spaces.Box
    obst_angle: gym_.spaces.Box
    obst_range_min: gym_.spaces.Box
    time_left: gym_.spaces.Box


@dataclass
class TurtlebotObservation:
    """观测值结构体"""

    scan_ranges: NDArray[np.floating]
    """雷达测距数组(扫描一圈), 尺寸 ()"""
    goal_yaw_err: NDArray[np.floating]
    """与目标中心的角误差,值域 [-pi,pi), 尺寸 ()"""
    goal_dist: NDArray[np.floating]
    """与目标中心的距离, 尺寸 ()"""
    obst_angle: NDArray[np.floating]
    """最近的障碍物所在方位角, 尺寸 ()"""
    obst_range_min: NDArray[np.floating]
    """最近的障碍物的距离, 尺寸 ()"""
    time_left: NDArray[np.floating]
    """剩余时间(s), 尺寸 ()"""


@dataclass
class TurtlebotActionSpace:
    ang_vel: gym_.spaces.Box
    lin_vel: gym_.spaces.Box


@dataclass
class TurtlebotAction:
    ang_vel: NDArray[np.floating]
    """角速度,绕z轴旋转右手定则方向为负, 单位 rad/s"""
    lin_vel: NDArray[np.floating]
    """线速度,单位 m/s"""


def cal_intersection_len(
    width: float,
    height: float,
    n_angles: int,
    angle_min: float,
    angle_max: float,
) -> np.ndarray:
    thres = np.zeros(n_angles)
    angles = np.linspace(angle_min, angle_max, n_angles)
    coss = np.cos(angles)
    sins = np.sin(angles)
    cz = coss == 0
    sz = sins == 0
    r1 = np.abs(np.where(cz, inf, width) / np.where(cz, 1, coss))
    r2 = np.abs(np.where(sz, inf, height) / np.where(sz, 1, sins))
    thres = np.where(r1 < r2, r1, r2) * 0.5
    assert np.isfinite(thres).all()
    return thres


def _pose2str(x: float, y: float, yaw: float):
    return f"x={x:+0.04g},y={y:+0.04g},yaw={yaw:+0.04g}"


@dataclass
class Turtlebot3_world_launch_args:
    model: Optional[str] = None
    x_pos: Optional[float] = None
    y_pos: Optional[float] = None
    z_pos: Optional[float] = None
    paused: bool = False
    use_sim_time: bool = True
    gui: bool = True  # True->不显示图形界面
    headless: bool = False  # True->不渲染图形,用于服务器仿真
    debug: bool = False


@dataclass
class Env_stage_core_init_args:
    launch_fname: str = "turtlebot3_stage_4.launch"
    launch_args: Optional[Turtlebot3_world_launch_args] = None
    sim_dt: float = 0.2
    """交互周期,单位 sec,但是被雷达参数限死,暂不改"""
    steps_tol: int = int(60.0 / 0.2)
    """任务总时限,超时则失败(终止状态)"""
    horizon: Optional[int] = None
    """控制视界=单局仿真步数限制,超时则截断(可能非终止状态),默认为 任务总时限"""
    gamma: float = 1 - 1e-2
    dtype: np.dtype = np.float32
    log_dir: Optional[str] = None  # 日志根目录


class Env_stage_core(gym_.Env):
    ANGULAR_VEL_MAX = 1.5  # 最大角速度 rad/s
    LINEAR_VEL_MAX = 0.15  # 最大线速度 m/s

    # 雷达测距参数范围可以从机器人模型描述文件读取
    SCAN_RANGE_VALID_MIN = 0.120000  # 有效测距的最小值
    SCAN_RANGE_VALID_MAX = 3.5  # 测距的最大值

    # 预处理后测距数据的范围
    SCAN_RANGE_EXT_MIN = 0.0
    SCAN_RANGE_EXT_MAX = SCAN_RANGE_VALID_MAX + 0.1

    SCAN_RANGE4INF = SCAN_RANGE_EXT_MIN
    SCAN_RANGE4NAN = SCAN_RANGE_EXT_MAX

    SCAN_ANGLE_MIN = 0.0
    SCAN_ANGLE_SUP = SCAN_ANGLE_MIN + 2 * pi

    # 航向角范围 (atan2 的值域)
    YAW_MIN = -pi
    YAW_SUP = YAW_MIN + 2 * pi

    def __init__(
        self,
        config: Optional[Env_stage_core_init_args] = None,
        seed: Optional[int] = None,
    ):
        """
        目前只支持 0.200 (机器人描述文件里限制了雷达更新频率)
        """
        if config is None:
            config = Env_stage_core_init_args()
        else:
            config = deepcopy(config)
        if config.launch_args is None:
            config.launch_args = Turtlebot3_world_launch_args()
        self._config = config

        super().__init__()
        super().reset(seed=seed)  # 初始化随机数发生器

        self.__n_episodes = 0  # 对局数
        self.__episode_steps: int = 0  # 决策步数

        debug = config.launch_args.debug
        self.__init_log(log_dir=config.log_dir, debug=debug)  # 初始化日志
        self.__init_debug_tags()
        try:
            stage = int(os.path.splitext(config.launch_fname)[0].split("_")[-1])
            self.loginfo(f"stage={stage}")
            self._stage = stage
        except:
            raise NotImplementedError(
                f"unsupported launch file '{config.launch_fname}'"
            )

        # 从 .world 读取的配置
        world_max_step_size = 0.001  # 最大仿真步长
        world_real_time_factor = 1

        # 命令 $(rospack find gazebo_ros) 可找到 empty_world.launch
        self.__use_pause = config.launch_args.paused
        self.__sim_dt = config.sim_dt
        """交互周期(sec)"""

        # 用于启动 launch 文件的模型后缀
        model_suffix = turtlebot3_model_suffix(config.launch_args.model)
        # 用于运行时修改状态的模型名
        self._bot_model_name = turtlebot3_model_name(model_suffix)
        # 读取模型描述文件
        self._bot_model_desc_fn = turtlebot3_urdf_fname(model_suffix)
        self._read_description_first_only()
        # 碰撞体积参数(依赖模型描述)
        if model_suffix == "burger":
            self._bot_collision_size_x = 0.140
            self._bot_collision_size_y = 0.140
        if model_suffix in ["waffle", "waffle_pi"]:
            self._bot_collision_size_x = 0.266
            self._bot_collision_size_y = 0.266
        else:
            raise NotImplementedError(f"unsupported arg model '{model_suffix}'")
        self.SCAN_ANGLE_MIN = self._scan_attrs.min_angle
        self.SCAN_ANGLE_MAX = self._scan_attrs.max_angle
        # 各方向上的碰撞距离阈值(基于雷达测距)
        self.SCAN_RANGES4COLLISION = cal_intersection_len(
            width=self._bot_collision_size_x * 1.10,
            height=self._bot_collision_size_y * 1.10,
            n_angles=self.laserScan_ranges_dim(),
            angle_min=self.SCAN_ANGLE_MIN,
            angle_max=self.SCAN_ANGLE_MAX,
        )
        self.SCAN_RANGE_THRES4COLLISION: float = self.SCAN_RANGES4COLLISION.max()
        self.loginfo(
            f"SCAN_RANGE_THRES for COLLISION:{self.SCAN_RANGE_THRES4COLLISION:0.04g}"
        )
        # 机器人(launch 文件参数)
        self._ic_x_pos = config.launch_args.x_pos  # x_pos
        self._ic_y_pos = config.launch_args.y_pos  # y_pos

        self._random_init_pose = True
        self._reset_easy_mode = True
        self.__resetState_period = 10  # 同一组(机器人位姿,目标区位姿)重复局数
        self.__resetState_count = 0  # 有效重复计数
        #
        self._respawn_goal = Respawn(
            stage=stage, logger_name=self._logger4ros_suffix, debug=debug
        )
        self.__resetGoal = True  # 是否要在下一次 reset 中重新生成目标区
        self.DIST_THRES4REACH = (
            min(
                self._respawn_goal._collision_size_x,
                self._respawn_goal._collision_size_y,
            )
            * 0.8
            + min(
                self._bot_collision_size_x,
                self._bot_collision_size_y,
            )
        ) * 0.5
        self.loginfo(f"SCAN_RANGE_THRES for REACH:{self.DIST_THRES4REACH:0.04g}")

        # launch 文件转为绝对路径
        self._launch_fname = path_find_with_interaction(
            path_parent(__file__), config.launch_fname
        )
        assert self._launch_fname is not None, (
            "no such launch file",
            f"'{config.launch_fname}'",
            "in",
            f"'{path_parent(__file__)}'",
        )
        # 依赖 ros 的部分
        self.gazebo_open()  # 启动仿真,每一个对象都会启动一个 gazebo 模拟器
        # gazebo 启动后再设置 ros 服务

        self.pub_cmd_vel = rospy.Publisher("cmd_vel", Twist, queue_size=5)
        self.pub_model_pose4nav = rospy.Publisher("initialpose", Pose, queue_size=1)
        self.pub_model_state = rospy.Publisher(
            "/gazebo/set_model_state", ModelState, queue_size=1
        )
        # self.sub_odom = rospy.Subscriber("odom", Odometry, self.getOdometry)
        rospy.on_shutdown(self.close)  # 注册回调函数,节点关闭前先关闭仿真

        # 状态信息
        # 机器人几何中心坐标(来自里程计)
        self.pos_x: float = 0.0
        self.pos_y: float = 0.0

        # 机器人偏航角,值域 [-pi,pi)
        self.yaw: float = 0.0
        # 航向角误差=目标方位角-偏航角,值域 [-pi,pi)
        self.yaw_err = 0.0

        # 奖励函数信息
        self.__steps_tol = config.steps_tol
        """搜索步数上限(首停时界),决策超过这个步数则为终止状态"""
        if config.horizon is None:
            config.horizon = config.steps_tol
        self.__horizon = min(config.horizon, self.__steps_tol)
        """控制视界=单局仿真最大决策步数,超时则截断(区别于首停时界)"""
        self.__gamma = float(config.gamma)
        self.__check_horizon()

        self.__rew_norm = True
        self.__rew_norm_no_bias = True
        self.__rew_norm_centralize = True
        self.__init_rewards()

        # gym 内容
        # 观测空间
        self.observation_space = gym_.spaces.Dict(
            TurtlebotObservationSpace(
                scan_ranges=gym_.spaces.Box(
                    low=self.SCAN_RANGE_EXT_MIN * np.ones(self.laserScan_ranges_dim()),
                    high=self.SCAN_RANGE_EXT_MAX * np.ones(self.laserScan_ranges_dim()),
                    shape=(self.laserScan_ranges_dim(),),
                    seed=self.np_random,
                ),
                goal_yaw_err=gym_.spaces.Box(
                    low=self.YAW_MIN,
                    high=self.YAW_SUP,
                    dtype=config.dtype,
                    seed=self.np_random,
                ),
                goal_dist=gym_.spaces.Box(
                    low=self.SCAN_RANGE_EXT_MIN,
                    high=self.SCAN_RANGE_EXT_MAX,
                    dtype=config.dtype,
                    seed=self.np_random,
                ),
                obst_angle=gym_.spaces.Box(
                    low=self.SCAN_ANGLE_MIN,
                    high=self.SCAN_ANGLE_MAX,
                    dtype=config.dtype,
                    seed=self.np_random,
                ),
                obst_range_min=gym_.spaces.Box(
                    low=self.SCAN_RANGE_EXT_MIN,
                    high=self.SCAN_RANGE_EXT_MAX,
                    dtype=config.dtype,
                    seed=self.np_random,
                ),
                time_left=gym_.spaces.Box(
                    low=0,
                    high=self.__steps_tol * self.__sim_dt,
                    dtype=config.dtype,
                    seed=self.np_random,
                ),
            ).__dict__,
            seed=self.np_random,
        )

        # 动作空间，解算前的 [角速度,线速度] 范围
        self.action_space = gym_.spaces.Dict(
            TurtlebotActionSpace(
                ang_vel=gym_.spaces.Box(
                    -self.ANGULAR_VEL_MAX,
                    self.ANGULAR_VEL_MAX,
                    dtype=config.dtype,
                    seed=self.np_random,
                ),
                lin_vel=gym_.spaces.Box(
                    0,
                    self.LINEAR_VEL_MAX,
                    dtype=config.dtype,
                    seed=self.np_random,
                ),
            ).__dict__,
            seed=self.np_random,
        )
        self.loginfo(f"config={config}")  # 打印最终配置
        self.__is_initialized = True
        return

    def make_launch_cmd(self):
        """
        将初始化参数设置为 launch 命令
        """
        cmd: List[str] = ["roslaunch", self._launch_fname]
        # bool 参数不区分大小写
        for k, v in self._config.launch_args.__dict__.items():
            if v is not None:
                cmd.append(f"{k}:={v}")
        return cmd

    def __check_init(self):
        try:
            assert self.__is_initialized
        except:
            raise Exception(f"{self.objid()}.__init__ is never called")

    def __check_reset(self):
        try:
            assert self.__is_reset
        except:
            raise Exception(f"{self.objid()}.reset is not called")

    DEFAULT_LOG_DIR = os.path.join(path_parent(__file__, 3), "log")
    LOG_FOLDER_ROS = "ros"

    def __init_log(self, log_dir: str, debug: bool):
        self.__debug = debug = bool(parse_optional(debug, False))
        if log_dir is None:
            log_dir = self.DEFAULT_LOG_DIR
        objid = self.objid()
        assert os.path.isdir(log_dir), f"{objid}:'{log_dir}' must be a directory"
        # 约定:每个环境对象占用一个文件夹,文件夹名由 对象ID 确定
        self._log_dir = _log_dir = os.path.join(log_dir, objid)

        level = logging.DEBUG if debug else logging.INFO
        self._log_dir4ros = os.path.join(_log_dir, self.LOG_FOLDER_ROS)

        # 主要负责一般部分的信息记录
        self._logger4main_logdir = os.path.join(_log_dir, "main.log")
        self._logger4main = logging.getLogger(objid)
        self._logger4main.setLevel(level)

        # ros 运行期间接口日志 !!!必须在主程序 ros 节点初始化后再初始化
        self._logger4ros_logdir = os.path.join(_log_dir, "ros.log")
        logger_rosout = logging.getLogger("rosout")
        self._logger4ros_suffix = objid
        self._logger4ros = logger_rosout.getChild(self._logger4ros_suffix)
        self._logger4ros.setLevel(level)

        # 记录奖励分量信息
        self._logger4retcomps_logdir = os.path.join(_log_dir, "episode_rewards")
        os.makedirs(self._logger4retcomps_logdir, exist_ok=True)
        self._logger4retcomps = sb3_logger.configure(
            self._logger4retcomps_logdir, ["csv", "tensorboard"]
        )

        self.__reset_log()

    @property
    def log_dir(self):
        return self._log_dir

    def __close_log(self):
        logger_clearHandlers(self._logger4main)
        logger_clearHandlers(self._logger4ros)
        self._logger4retcomps.close()

    def __reset_log(self):
        n_episode = self.__n_episodes
        #
        logger = self._logger4main
        log_fn = self._logger4main_logdir
        logger_quickReset(
            logger.name,
            level=logger.level,
            fmt="[%(levelname)s][%(name)s] %(asctime)s "
            + f"Episode{n_episode} "
            + "Step%(step)s: "
            + "%(message)s",
            log_fn=log_fn,
            to_stderr=False,
            to_stdout=True,
        )
        #
        logger = self._logger4ros
        log_fn = self._logger4ros_logdir
        logger_quickReset(
            logger.name,
            level=logger.level,
            fmt="[%(levelname)s][%(name)s] %(asctime)s "
            + f"Episode{n_episode}: "
            + "%(message)s",
            log_fn=log_fn,
            to_stderr=False,
            to_stdout=False,
        )
        return

    def _log(self, level: int, msg: str):
        kwargs = {}
        kwargs["step"] = self.__episode_steps
        self._logger4main.log(level, msg, extra=kwargs)

    def logdebug(self, msg: str):
        self._log(logging.DEBUG, msg)

    def loginfo(self, msg: str):
        self._log(logging.INFO, msg)

    def logwarn(self, msg: str):
        self._log(logging.WARNING, msg)

    def logerr(self, msg: str):
        self._log(logging.ERROR, msg)

    def __init_debug_tags(self):
        self.debug_msgs_pause = False
        self.debug_msgs_reward = False
        self.debug_mdgs_returns = True
        self.debug_msgs_reset = False
        self.debug_msgs_step = False
        self.debug_msgs_close = True
        self.debug_msgs_info = False
        self.debug_control_auto = True

    def get_robot_model_state(self):
        return gazebo_get_ModelState(
            model_name=self._bot_model_name,
            logger_name=self._logger4ros_suffix,
        )

    def update_real_pose(self):
        """更新机器人的真实位姿"""
        self.logdebug(f"get real pose for model {self._bot_model_name}")
        ms = self.get_robot_model_state()
        self.pos_x_real = ms.pose.position.x
        self.pos_y_real = ms.pose.position.y
        return ms.pose

    def _pose_parse(self, pose: Pose):
        x = pose.position.x
        y = pose.position.y
        ori = pose.orientation
        _, _, yaw = euler_from_quaternion(
            [ori.x, ori.y, ori.z, ori.w], axes=self.EULER_AXES
        )
        return x, y, yaw

    def _show_pose(self, pose: Pose = None):
        if pose is None:
            pose = self.update_real_pose()
        x, y, yaw = self._pose_parse(pose)
        model_name = self._bot_model_name
        self.logdebug(model_name + " real " + _pose2str(x, y, yaw))
        self.logdebug(
            model_name + " odom " + _pose2str(self.pos_x, self.pos_y, self.yaw)
        )

    EULER_AXES = "sxyz"

    def resetOdometry(self, init_pose: Pose = None):
        """同步初始位姿到导航系统"""
        if init_pose is None:
            init_pose = self.update_real_pose()
        self.pub_model_pose4nav.publish(init_pose)

    def sim_set_pose(self, x: float = None, y: float = None, yaw: float = None):
        """自定义位姿,如果参数不设置,则沿用旧的位姿"""
        old_pose = self.update_real_pose()

        # 修改机器人的位置
        new_pose = deepcopy(old_pose)
        new_pos = new_pose.position
        if x is not None:
            new_pos.x = x
        if y is not None:
            new_pos.y = y
        new_pos.z = 0

        if yaw is not None:
            new_ori = new_pose.orientation
            (
                new_ori.x,
                new_ori.y,
                new_ori.z,
                new_ori.w,
            ) = quaternion_from_euler(0, 0, yaw, axes=self.EULER_AXES)

        new_state = ModelState()
        new_state.model_name = self._bot_model_name
        new_state.pose = new_pose

        # 设置机器人的初始状态
        rospy.wait_for_service(self.pub_model_state.name)
        self.pub_model_state.publish(new_state)
        return

    def getOdometry(self, odom: Odometry = None):
        """
        读取里程计信息
        """
        if odom is None:
            odom = rospy.wait_for_message("odom", Odometry)

        self.pos_x, self.pos_y, yaw = self._pose_parse(odom.pose.pose)
        goal_x, goal_y = self._respawn_goal.getPosition()
        goal_angle = math.atan2(
            goal_y - self.pos_y,
            goal_x - self.pos_x,
        )  # 对目标的视线角

        self.yaw = yaw
        self.yaw_err = goal_angle - yaw

    def getGoalDistace(self):
        """获取此时 目标点 与 里程计位置 距离"""
        goal_x, goal_y = self._respawn_goal.getPosition()
        goal_distance = math.hypot(
            goal_x - self.pos_x,
            goal_y - self.pos_y,
        )
        return goal_distance

    def _read_description_first_only(self):
        """
        读取模型描述文件,获取传感器部分参数
        """
        try:
            if self.__model_description_loaded:
                return
        except:  #  未定义
            self.__model_description_loaded = False
            pass

        self._scan_attrs = Turtlebot3_LaserScan_Attrs()
        self._scan_attrs.parse_file(self._bot_model_desc_fn)

        self.__model_description_loaded = True
        return

    def state_dict(self) -> Dict[str, Any]:
        """"""
        raise NotImplementedError
        return

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raise NotImplementedError
        return

    def laserScan_ranges_dim(self):
        """
        更新激光雷达数据维数(只在首次调用时读取模型描述文件)
        """
        self._read_description_first_only()
        return self._scan_attrs.ranges_dim

    def _get_LaserScan(self) -> LaserScan:
        """读取激光雷达实时信息"""

        scan: LaserScan = None
        while scan is None:
            try:
                assert not rospy.is_shutdown(), "rospy shutdown"
                scan = rospy.wait_for_message("scan", LaserScan, timeout=10.0)
                # 注意,光雷会按照模型描述文件设置的更新频率来获取数据,并按照对应的频率阻塞到返回
            except Exception as e:
                raise e

        return scan

    def get_obs(self) -> Tuple[TurtlebotObservation, bool, EnvInfo]:
        """
        计算新观测
        returns:
            next_obs: TurtlebotObservation
            terminated: bool
            info: EnvInfo
        """
        # assert not gazebo_is_paused()
        terminated = False
        collision = False
        info = EnvInfo()
        next_obs = TurtlebotObservation(**self.observation_space.sample())

        # 给出的控制量
        info.ang_vel = self.ang_vel
        info.lin_vel = self.lin_vel
        # 实际产生的控制量
        info.ang_vel_real, info.lin_vel_real = self.get_real_cmd()

        # 获取雷达观测
        scan_data = self._get_LaserScan()

        scan_ranges = np.asarray(scan_data.ranges, self.observation_space.dtype)
        assert scan_ranges.shape[0] == self.laserScan_ranges_dim()

        # 处理异常值为有限值
        tvs_inf = np.isinf(scan_ranges)
        tvs_nan = np.isnan(scan_ranges)
        scan_ranges = np.where(tvs_nan, self.SCAN_RANGE4NAN, scan_ranges)
        scan_ranges = np.where(tvs_inf, self.SCAN_RANGE4INF, scan_ranges)

        # 搜索正常测距数据中的最有可能发生碰撞的方位
        ranges_ext = np.where(tvs_inf | tvs_nan, inf, scan_ranges)
        collision = (ranges_ext <= self.SCAN_RANGES4COLLISION).any()
        too_close = (ranges_ext <= self.SCAN_RANGES4COLLISION + 0.3).any()
        obst_angle_idx = np.argmin(ranges_ext - self.SCAN_RANGES4COLLISION).item()
        obst_angle = (
            scan_data.angle_min + obst_angle_idx * scan_data.angle_increment
        )  # 最容易碰撞的方位角
        obst_angle = mod_range(
            obst_angle, self.SCAN_ANGLE_MIN, self.SCAN_ANGLE_SUP
        ).item()
        obst_range_min = float(scan_ranges[obst_angle_idx])  # 最近的距离

        # 状态转移前位置 与 状态转移前目标 的距离
        dist_old2old_goal = self.__goal_dist

        # 状态转移后位置 与 状态转移前目标 的距离
        self.updateGoalPosition(regen=False)
        self.getOdometry()
        dist_cur2old_goal = self.__goal_dist = self.getGoalDistace()

        old_goal_get = False
        # 终止条件判断
        # 碰撞障碍(原环境比较严重的 bug 在于只能通过雷达测距自行判断碰撞,这取决于雷达分辨率)
        if collision:
            self.loginfo("Collision!!!")
            terminated = True
        elif dist_cur2old_goal < self.DIST_THRES4REACH:  # 触碰目标区域中心 & 无墙体碰撞
            self.loginfo("Goal!!!")
            terminated = True
            old_goal_get = True
        elif self.__time_left <= 0:
            self.loginfo("Time up!!!")
            self.__time_left = 0
            terminated = True

        info.pos_x = self.pos_x
        info.pos_y = self.pos_y
        info.yaw = mod_range(self.yaw, self.YAW_MIN, self.YAW_SUP).item()
        info.obst_collision = collision
        info.obst_angle_idx = obst_angle_idx
        info.obst_angle = obst_angle
        info.obst_too_close = too_close
        info.obst_dist = obst_range_min
        info.goal_dist = dist_cur2old_goal
        info.goal_dist_old = dist_old2old_goal
        info.goal_get = old_goal_get
        info.goal_yaw_err = mod_range(self.yaw_err, self.YAW_MIN, self.YAW_SUP).item()

        next_obs.scan_ranges[:] = scan_ranges
        next_obs.obst_range_min.itemset(obst_range_min)
        next_obs.obst_angle.itemset(obst_angle)
        next_obs.goal_dist.itemset(dist_cur2old_goal)
        next_obs.goal_yaw_err.itemset(info.goal_yaw_err)
        next_obs.time_left.itemset(self.__time_left)

        info.round(4)
        if self.debug_msgs_info:
            self.logdebug(f"info={info}")
        self.info = info
        return next_obs, terminated, info

    def get_real_cmd(self):
        """读取真实的控制量"""
        self._logger4ros.debug(f"get real cmd for {self._bot_model_name}")

        ms = self.get_robot_model_state()
        ang_vel = ms.twist.angular.z
        lin_vel = ms.twist.linear.x

        self.logdebug(f"ang_vel={ang_vel:+0.04g},lin_vel={lin_vel:+0.04g}")
        return ang_vel, lin_vel

    def __check_horizon(self):
        """
        0<horizon<=steps_tol<inf,
        0<=gamma<=1
        """
        steps_tol = self.__steps_tol
        assert isinstance(steps_tol, int) and 0 < steps_tol < inf, Exception(
            f"steps_tol must be integer in (0,inf), but get {steps_tol}"
        )
        horizon = self.__horizon
        assert isinstance(horizon, int) and 0 < horizon <= steps_tol, Exception(
            f"horizon must be integer in (0,steps_tol], but get {horizon}"
        )
        gamma = self.__gamma
        assert 0 <= gamma <= 1, Exception(f"gamma must be in [0,1], but get {gamma}")

    def __init_rewards(self):
        self.__check_horizon()
        steps_tol = self.__steps_tol
        horizon = self.__horizon
        gamma = self.__gamma
        secs_tol = steps_tol * self.__sim_dt
        self.loginfo(f"secs_tol:{secs_tol}")
        self.loginfo(f"steps_tol:{steps_tol}")
        self.loginfo(f"horizon:{horizon}")
        self.loginfo(f"gamma:{gamma}")
        # 暂不支持 rho
        rho = ...  # \max_\pi \max_{s\notin S_*,t} \Pr(S_{t+N}\notin S_*|S_t=s,\pi)
        #
        w4cost_dt = 1
        w4cost_angv = 10
        w4cost_heading = 0.5 * w4cost_angv
        w4cost_linv = 0
        #
        _w4cost = sum([w4cost_dt, w4cost_angv, w4cost_heading, w4cost_linv])
        self.loginfo(f"original one-step non-terminal costs bound:{_w4cost}")
        #
        gN = gamma**steps_tol
        if gamma < 1:
            gSum2N = (1 - gN) / (1 - gamma)
        else:
            gSum2N = steps_tol
        w4reach = _w4cost * gSum2N / gN * 2.0  # loan奖励
        w4collision = w4reach * gN
        self.SCAN_RANGE_THRES4SAFETY = self.SCAN_RANGE_THRES4COLLISION * 2.0
        w4collision_soft = w4collision / steps_tol * 3
        #
        region_diam = self._respawn_goal.roominfo.ROOM_INNER_LENGTH * math.sqrt(2)
        motion_diam = min(region_diam, self.LINEAR_VEL_MAX * secs_tol)
        self.motion_diam = motion_diam  # m
        self.loginfo(f"motion_diam:{motion_diam:0.04g}")
        rew_dr_bound = self.LINEAR_VEL_MAX * self.__sim_dt * 1.5
        rets_dr_bound = motion_diam  # 不含终端值
        w4dr = w4reach / rets_dr_bound  # 加权后累计收益约等于首达奖励
        #
        self.__reward_functions = [
            BoundedReward(
                name="goal_reach",
                func=self._reward_goal_reach,
                low=0,
                high=1,
                is_terminal=True,
                weight=w4reach,
            ),
            BoundedReward(
                name="delta_dist",
                func=self._reward_goal_dr,
                low=-rew_dr_bound,
                high=rew_dr_bound,
                weight=w4dr,
                priori_returns_low=-rets_dr_bound,
                priori_returns_high=rets_dr_bound,
            ),
            BoundedReward(
                name="time_elapse",
                func=self._reward_time_elapse,
                low=-1,
                high=-1,
                weight=w4cost_dt,
            ),
            BoundedReward(
                name="collision",
                func=self._reward_collision,
                low=-1,
                high=0,
                is_terminal=True,
                weight=w4collision,
            ),
            BoundedReward(
                name="obst_dist",
                func=self._reward_obstacle_dist,
                low=-1,
                high=0,
                weight=w4collision_soft,
            ),
            BoundedReward(
                name="heading_err",
                func=self._reward_heading_err,
                low=-1,
                high=0,
                weight=w4cost_heading,
            ),
            BoundedReward(
                name="ang_vel",
                func=self._reward_angv,
                low=-1,
                high=0,
                weight=w4cost_angv,
            ),
            BoundedReward(
                name="lin_vel",
                func=self._reward_linv,
                low=0,
                high=1,
                weight=w4cost_linv,
            ),
        ]

        def _rf_ordkey(rf: BoundedReward):
            return int(rf.is_terminal)

        self.__reward_functions = sorted(self.__reward_functions, key=_rf_ordkey)
        for rf in self.__reward_functions:
            rf.gamma = gamma
            rf.steps_tol = steps_tol

        self.show_rewards_info(self.__reward_functions)

        # 函数界估计
        ## 奖励函数界 (按是否终局奖励划分)
        rw_lb_nonterm = 0.0
        rw_ub_nonterm = 0.0
        rw_lb_term = 0.0
        rw_ub_term = 0.0
        for rf in self.__reward_functions:
            rlb = rf.low * rf.weight
            rub = rf.high * rf.weight
            if rf.is_terminal:
                rw_lb_term += rlb
                rw_ub_term += rub
            else:
                rw_lb_nonterm += rlb
                rw_ub_nonterm += rub
        # 综合奖励函数
        wrf_nonterm = BoundedReward(
            name="non-terminal rewards",
            low=rw_lb_nonterm,
            high=rw_ub_nonterm,
            gamma=gamma,
            steps_tol=steps_tol,
            is_terminal=False,
        )
        wrf_term = BoundedReward(
            name="terminal rewards",
            low=rw_lb_term,
            high=rw_ub_term,
            gamma=gamma,
            steps_tol=steps_tol,
            is_terminal=True,
        )
        self.show_rewards_info([wrf_nonterm, wrf_term])

        ## 奖励函数界
        vf_lb_term, vf_ub_term = wrf_term.vf_bounds()
        ### !!!非终局奖励的价值函数界是用奖励函数界简单估计的,非常松,需要修正
        vf_lb_nonterm, vf_ub_nonterm = wrf_nonterm.vf_bounds()

        vf_lb = vf_lb_nonterm + vf_lb_term
        vf_ub = vf_ub_nonterm + vf_ub_term
        assert vf_lb < vf_ub, (f"expected vf_lb < vf_ub, but get", (vf_lb, vf_ub))

        for k, v in (
            ("rf_lb_nonterm", wrf_nonterm.low),
            ("rf_ub_nonterm", wrf_nonterm.high),
            ("rf_lb_term", wrf_term.low),
            ("rf_ub_term", wrf_term.high),
            ("vf_lb_nonterm", vf_lb_nonterm),
            ("vf_ub_nonterm", vf_ub_nonterm),
            ("vf_lb_term", vf_lb_term),
            ("vf_ub_term", vf_ub_term),
            ("vf_lb", vf_lb),
            ("vf_ub", vf_ub),
        ):
            s = f"{v:+0.06e}" if isinstance(v, float) else str(v)
            self.loginfo(f"{k}:\t{s}")

        # 注意,因为该归一化只保证价值函数满足 \forall s, LB_V\leq V(s)\leq HB_V
        # 但是累计回报(不含边界)可能越界
        if self.__rew_norm_no_bias:
            self._REWARD_NORM_BIAS = 0
            self._REWARD_NORM_WEIGHT = (abs(vf_ub) + abs(vf_lb)) * 0.5
            self.loginfo("REWARD_NORM_WEIGHT = (abs(vf_ub) + abs(vf_lb))*0.5")
        else:
            if not self.__rew_norm_centralize:
                # 玄学版! 按 bias=(1-gamma)*LB_V 会导致收益曲线形状恰好反相!
                self._REWARD_NORM_BIAS = vf_lb * (1 - gamma)
                self._REWARD_NORM_WEIGHT = vf_ub - vf_lb
                self.loginfo("REWARD_NORM_BIAS = vf_lb * (1 - gamma)")
                self.loginfo("REWARD_NORM_WEIGHT = vf_ub - vf_lb")
                self.loginfo("expected to rescale value function in [0,1]")
            else:
                # 改进版,中心化, 但是引入偏差还是会导致两种回报曲线有部分不一致性
                self._REWARD_NORM_BIAS = (vf_ub + vf_lb) * 0.5 * (1 - gamma)
                self._REWARD_NORM_WEIGHT = (vf_ub - vf_lb) * 0.5
                self.loginfo("REWARD_NORM_BIAS = (vf_ub + vf_lb) * 0.5 * (1 - gamma)")
                self.loginfo("REWARD_NORM_WEIGHT = (vf_ub - vf_lb) * 0.5")
                self.loginfo("expected to rescale value function in [-1,1]")
        assert self._REWARD_NORM_WEIGHT > 0, (
            f"_REWARD_NORM_WEIGHT must be positive"
            f"but get {self._REWARD_NORM_WEIGHT}"
        )
        for k, v in (
            ("REWARD_NORM_BIAS", self._REWARD_NORM_BIAS),
            ("REWARD_NORM_WEIGHT", self._REWARD_NORM_WEIGHT),
            ("rew_norm_no_bias", self.__rew_norm_no_bias),
            ("rew_norm_centralize", self.__rew_norm_centralize),
        ):
            s = f"{v:+0.06e}" if isinstance(v, float) else str(v)
            self.loginfo(f"{k}:\t{s}")

        # 原始奖励信号列
        self.__collected_origin_rewards: NDArray[np.floating] = np.zeros(
            (len(self.__reward_functions), horizon)
        )
        # 加权求和&归一化的奖励信号列
        self.__collected_final_rewards: NDArray[np.floating] = np.zeros(horizon)

    def __del_rewards(self):
        self.__collected_origin_rewards = None
        self.__collected_final_rewards = None

    def _log_episode_rewards(self):
        log_ud_uw = False
        log_d_uw = False
        log_ud_w = False

        title = "episode_rewards"
        title_comp = f"{title}/components"
        title_total = f"{title}/total"
        _word_d = "discounted"
        _word_ud = f"un{_word_d}"
        _word_w = "weighted"
        _word_uw = f"un{_word_w}"
        _word_f = f"final"
        words_o = f"{_word_ud}/{_word_uw}"
        words_d = f"{_word_d}/{_word_uw}"
        words_w = f"{_word_ud}/{_word_w}"
        words_dw = f"{_word_d}/{_word_w}"
        words_f = f"{_word_ud}/{_word_f}"
        words_df = f"{_word_d}/{_word_f}"
        del _word_w, _word_uw, _word_d, _word_ud, _word_f
        #
        df: Dict[str, float] = {}
        gain_o = 0
        gain_d = 0
        gain_w = 0
        gain_wd = 0
        gamma = self.__gamma
        t = self.__episode_steps  #
        ts = slice(t)
        gain_ud_f, gain_d_f = rewards2gain(self.__collected_final_rewards[ts], gamma)
        for i, rf in enumerate(self.__reward_functions):
            rfname = rf.name
            weight = rf.weight
            rews_i = self.__collected_origin_rewards[i, ts]
            gain_ud_uw_i, gain_d_uw_i = rewards2gain(rews_i, gamma)
            gain_ud_w_i = gain_ud_uw_i * weight
            gain_d_w_i = gain_d_uw_i * weight

            gain_o += gain_ud_uw_i
            gain_d += gain_d_uw_i
            gain_w += gain_ud_w_i
            gain_wd += gain_d_w_i

            if log_ud_uw:
                df[f"{title_comp}/{words_o}/{rfname}"] = gain_ud_uw_i
            if log_d_uw:
                df[f"{title_comp}/{words_d}/{rfname}"] = gain_d_uw_i
            if log_ud_w:
                df[f"{title_comp}/{words_w}/{rfname}"] = gain_ud_w_i

            df[f"{title_comp}/{words_dw}/{rfname}"] = gain_d_w_i

        if log_ud_uw:
            df[f"{title_total}/{words_o}"] = gain_o
        if log_d_uw:
            df[f"{title_total}/{words_d}"] = gain_d
        if log_ud_w:
            df[f"{title_total}/{words_w}"] = gain_w

        df[f"{title_total}/{words_dw}"] = gain_wd

        if log_ud_w:
            df[f"{title_total}/{words_f}"] = gain_ud_f
        df[f"{title_total}/{words_df}"] = gain_d_f

        logger = self._logger4retcomps
        for k, v in df.items():
            logger.record(k, v)
        logger.dump(self.__n_episodes)
        return df

    def show_rewards_info(self, rfs: List[BoundedReward]):
        msg = [f"reward funtions:"]
        data = [rf.df_dict() for rf in rfs]
        df = pd.DataFrame(data)
        msg.append(
            df.to_string(
                index=False,  # 隐藏行索引
                max_rows=None,  # 显示所有行
                max_cols=None,  # 显示所有列
                line_width=math.inf,  # 不自动换行
            )
        )
        self.loginfo("\n".join(msg))

    def _get_reward(
        self,
        next_obs: TurtlebotObservation,
        terminated: bool,
        info: EnvInfo,
    ) -> float:
        """根据状态,动作,终止标志,信息计算所有奖励信号"""
        t = self.__episode_steps - 1
        reward = 0.0
        for i, rf in enumerate(self.__reward_functions):
            rlb = rf.low
            rub = rf.high
            r_orgin = rf.func(next_obs, terminated, info)  # 原始奖励信号
            if not (rlb <= r_orgin <= rub):
                self.logwarn(
                    f"reward '{rf.name}' {r_orgin:+0.04g} out of bound [{rlb:+0.04g},{rub:+0.04g}]"
                )
            reward += r_orgin * rf.weight

            self.__collected_origin_rewards[i, t] = r_orgin

        if self.__rew_norm:
            reward_n = (reward - self._REWARD_NORM_BIAS) / self._REWARD_NORM_WEIGHT
            if self.__debug:
                self.logdebug(f"normlized reward:{reward:+0.04g}->{reward_n:+0.04g}")
            reward = reward_n

        self.__collected_final_rewards[t] = reward
        return reward

    def _reward_goal_reach(
        self,
        next_obs: TurtlebotObservation,
        terminated: bool,
        info: EnvInfo,
    ):
        """到达奖励,值域 {0,1}"""
        return float(info.goal_get)

    def _reward_time_elapse(
        self,
        next_obs: TurtlebotObservation,
        terminated: bool,
        info: EnvInfo,
    ):
        """耗时奖励,值域 {-1}"""
        return -1.0

    def _reward_collision(
        self,
        next_obs: TurtlebotObservation,
        terminated: bool,
        info: EnvInfo,
    ):
        """碰撞奖励,值域 {-1,0}"""
        return -float(info.obst_collision)

    def _reward_goal_dr(
        self,
        next_obs: TurtlebotObservation,
        terminated: bool,
        info: EnvInfo,
    ):
        r"""
        距离变化奖励(基于折扣问题势能形式的 shaping reward)
        min_{a_0} \gamma^T d(s_T) \iff max_{a_0} d(s_t)-\gamma^T d(s_T)

        分解为  (1-\gamma)|LOS|+\gamma\Delta|LOS|
                =(1-\gamma)|LOS|+\gamma\Delta V_{close} \Delta t
        """
        shaping_rew = info.goal_dist_old - self.__gamma * info.goal_dist
        return shaping_rew

    def _reward_obstacle_dist(
        self,
        next_obs: TurtlebotObservation,
        terminated: bool,
        info: EnvInfo,
    ) -> float:
        """
        离障碍物过近的惩罚,值域 [-1,0]
        """
        thres = self.SCAN_RANGE_THRES4SAFETY
        use_hinge = True
        if use_hinge:
            cost = max(0, 1.0 - info.obst_dist / thres)  # hinge 损失
        else:
            cost = info.obst_dist <= thres
        return -cost

    def _reward_angv(
        self,
        next_obs: TurtlebotObservation,
        terminated: bool,
        info: EnvInfo,
    ) -> float:
        """角速度奖励,值域 [-1,0]"""

        goal_yaw_err = info.goal_yaw_err
        ang_sat = pi / 4
        k = self.ANGULAR_VEL_MAX / ang_sat  # 比例系数
        ang_vel_target = np.clip(
            k * goal_yaw_err, -self.ANGULAR_VEL_MAX, self.ANGULAR_VEL_MAX
        )
        # self.set_cmd(ang_vel_target, self.LINEAR_VEL_MAX) # 测试用比例导引给出的控制量
        ang_vel = info.ang_vel
        err = (ang_vel_target - ang_vel) / (2 * self.ANGULAR_VEL_MAX)
        reward = -(err**2)
        return reward

    def _reward_heading_err(
        self,
        next_obs: TurtlebotObservation,
        terminated: bool,
        info: EnvInfo,
    ) -> float:
        """角误差奖励,值域 [-1,1]"""

        goal_yaw_err = info.goal_yaw_err
        err = goal_yaw_err / pi
        reward = -(err**2)
        return reward

    def _reward_linv(
        self,
        next_obs: TurtlebotObservation,
        terminated: bool,
        info: EnvInfo,
    ) -> float:
        """线速度奖励,值域 [0,1]"""
        return info.lin_vel / self.LINEAR_VEL_MAX

    def updateGoalPosition(self, regen=False) -> Tuple[float, float]:
        """更新目标区位置 / 重新生成目标区(不依赖机器人实时位置)"""
        if regen:
            regions_must_collide_any = []
            regions_must_collide_none = []
            if self._stage != 4:
                ic_bot_pos_x = self._ic_x_pos
                ic_bot_pos_y = self._ic_y_pos

                r1 = max(self.DIST_THRES4REACH * 2, self.motion_diam * 0.1)
                regions_must_collide_none.append(
                    SimplestCircle(ic_bot_pos_x, ic_bot_pos_y, r1)
                )
                if self._reset_easy_mode:
                    r2 = max(r1, self.motion_diam * 0.5)
                    # 必须与圆有接触
                    regions_must_collide_any.append(
                        SimplestCircle(ic_bot_pos_x, ic_bot_pos_y, r2)
                    )
            if self.__debug:
                self.logdebug(
                    "\n".join(
                        (
                            f"going to regen goal in with ",
                            f"regions_must_collide_any={np.asarray(regions_must_collide_any)}",
                            f"regions_must_collide_none={np.asarray(regions_must_collide_none)}",
                        )
                    )
                )
            pos = self._respawn_goal.regen(
                rng=self.np_random,
                regions_must_collide_any=regions_must_collide_any,
                regions_must_collide_none=regions_must_collide_none,
            )
            self.loginfo("new goal position: x{:+0.04g}, y{:0.04g}".format(*pos))
        else:
            pos = self._respawn_goal.getPosition(update=True)
        return pos

    def sim_reset_world(self):
        gazebo_reset(logger_name=self._logger4ros_suffix)

    def sim_unpause(self):
        gazebo_unpause(logger_name=self._logger4ros_suffix)

    def sim_pause(self):
        gazebo_pause(logger_name=self._logger4ros_suffix)

    def objid(self):
        """以英文字母开头,只包含数字、英文、下划线和小数点,小数点前后有至少一个数字、英文、下划线"""
        try:
            return self.__objname
        except:
            name = f"{type(self).__name__}.x{hex(id(self)).upper()}"
            self.__objname = name
        return self.__objname

    def new_safe_circle(self, x: float, y: float, scale: float = 1):
        """
        计算一个机器人在所有方向扫过的圆,圆心是机器人几何中心
        """
        diam = math.hypot(self._bot_collision_size_x, self._bot_collision_size_y)
        r = diam * scale * 0.5
        rc = SimplestCircle(x, y, r)
        return rc

    def __genNewRobotPose(self, goal_x: float, goal_y: float):
        """根据给定的目标区位置产生合理的机器人初始位姿"""
        regions_must_collide_any = []
        regions_must_collide_none = []
        if self._stage != 4:
            r1 = max(self.DIST_THRES4REACH * 2, self.motion_diam * 0.1)
            regions_must_collide_none.append(SimplestCircle(goal_x, goal_y, r1))
            if self._reset_easy_mode:
                r2 = max(r1, min(self.motion_diam * 0.5))
                # 必须与圆有接触
                regions_must_collide_any.append(SimplestCircle(goal_x, goal_y, r2))
            if self.__debug:
                self.logdebug(
                    " ".join(
                        f"gen new position for {self._bot_model_name} with"
                        f"regions_must_collide_any={np.asarray(regions_must_collide_any)}"
                        f"regions_must_collide_none={np.asarray(regions_must_collide_none)}"
                    )
                )

        new_x, new_y = self._respawn_goal.genNewPosition(
            rng=self.np_random,
            obj2place=self.new_safe_circle(0, 0, scale=1.1),
            regions_must_collide_any=regions_must_collide_any,
            regions_must_collide_none=regions_must_collide_none,
        )
        new_yaw = affine_comb(self.YAW_MIN, self.YAW_SUP, self.np_random.random())

        # 设置机器人初始位姿
        self.sim_set_pose(x=new_x, y=new_y, yaw=new_yaw)

        self.loginfo(self._bot_model_name + " init " + _pose2str(new_x, new_y, new_yaw))
        return

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        seed: 环境第一次初始化必须指定, 若设为 None 则表示延续上一轮的随机种子
        """
        self.__n_episodes += 1
        self.__episode_steps = 0  # 重置仿真步数
        self.__reset_log()
        self.__check_init()

        super().reset(seed=seed, options=options)  # 重新初始化伪随机数发生器

        loops = 0
        while True:
            loops += 1
            self.loginfo(f"{self.objid()}.reset {loops}")

            # 重置剩余时间
            self.__time_left = self.__steps_tol * self.__sim_dt

            # 重置控制量
            self.ang_vel = 0.0
            self.lin_vel = 0.0
            try:
                self.sim_unpause()  # 继续,此后处理 gazebo 的逻辑

                self.sim_reset_world()  # 世界重置

                # 更新目标区
                goal_x, goal_y = self.updateGoalPosition(
                    regen=self.__resetGoal
                )  # 重新生成/刷新实时位置
                self.__resetGoal = False

                # 自定义机器人初始位姿(依赖目标区位置)
                if self._random_init_pose:
                    self.__genNewRobotPose(goal_x, goal_y)
                    self.__time_left *= affine_comb(1, 0.5, self.np_random.random())

                # 获取真实位姿
                real_pose = self.update_real_pose()
                self.resetOdometry(real_pose)  # 重置里程计
                self.getOdometry()  # 读取里程计

                if self.__debug:
                    # 显示实际初始位姿(校对初始位置是否设置成功)
                    self._show_pose(real_pose)

                self.__goal_dist = self.getGoalDistace()  # 更新距离
                self.loginfo(f"new goal dist {self.__goal_dist:0.04g}")

                # 状态转移&观测
                init_obs, terminated, info = self.get_obs()

                if self.__use_pause:
                    self.sim_pause()  # 暂停

            except Exception as e:
                msg = f"{self.objid()}.reset failed with error: {e}"
                self.logerr(msg)
                self._logger4ros.error(msg)
                raise e

            if not terminated:
                break
            self.loginfo("emit the terminal state")
        #
        assert not terminated, (self.objid(), init_obs, terminated, info)
        self.__is_reset = True
        return init_obs.__dict__, info.__dict__

    def set_cmd(self, ang_vel: float, lin_vel: float):
        vel_cmd = Twist()
        vel_cmd.angular.z = ang_vel  # 角速度 rad/s
        vel_cmd.linear.x = lin_vel  # 线速度 m/s
        self._last_cmd = vel_cmd  # 记录
        try:
            self.pub_cmd_vel.publish(vel_cmd)  # 写入控制量
        except rospy.ROSException as e:
            self.logwarn(f"failed to set command: {e}")
            raise e

    def step(
        self,
        action: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        @param action: 动作指令 (角速度,线速度)
        returns:
            next_state: np.ndarray, 雷达测距数据

            reward: float 奖励信号

            terminated: bool 是否为正常终止状态

            info: Dict 其他信息,包括碰撞事件
        """
        self.__check_init()
        self.__check_reset()
        self.__time_left -= self.__sim_dt
        self.__episode_steps += 1
        # 到达控制仿真时限制/仿真异常中断
        truncated = self.__episode_steps >= self.__horizon
        try:
            # 1.解算动作为仿真环境接受的控制量
            action_ = TurtlebotAction(**action)
            self.ang_vel = action_.ang_vel.item()
            self.lin_vel = action_.lin_vel.item()
            if self.debug_control_auto:
                self.set_cmd(self.ang_vel, self.lin_vel)  # 写入控制量

            self.sim_unpause()  # 继续,此后处理 gazebo 的逻辑

            # 2. 状态转移&观测
            next_obs, terminated, info = self.get_obs()

            if self.debug_msgs_step:
                self._show_pose()

            if self.__use_pause:  # 暂停
                self.sim_pause()
        except Exception as e:
            msg = "\n".join(
                f"step failed with error: {e}",
                f"action:{action}",
            )
            self.logerr(msg)
            self._logger4ros.error(msg)
            raise e

        # 3. 计算奖励信号
        reward = self._get_reward(next_obs, terminated, info)

        done = terminated or truncated
        if done:
            self.__is_reset = False
            self._log_episode_rewards()
        return next_obs.__dict__, reward, terminated, truncated, info.__dict__

    def __init_gazebo(self):
        try:
            self._simulator
        except Exception as e:  # 属性未初始化
            self._simulator = None

    def gazebo_open(self) -> bool:
        """如果 launch 文件已启动则无操作, 否则启动 gazebo"""
        self.__init_gazebo()
        if self._simulator is None:
            cmd = self.make_launch_cmd()
            msg = "simulator launch command:\n" + (" ".join(str(arg) for arg in cmd))
            self.loginfo(msg)
            self._logger4ros.info(msg)
            #
            # 使用当前环境变量创建一个副本,修改环境变量传入子进程
            sys_env = dict(os.environ)
            sys_env["ROS_LOG_DIR"] = self._log_dir4ros  # ros 日志重定向
            #
            self._simulator = subprocess.Popen(cmd, env=sys_env)
            time.sleep(5.0)
            #
            msg = "simulator is ready"
            self.loginfo(msg)
            self._logger4ros.info(msg)
        return

    def gazebo_close(self, wait4terminate=True) -> bool:
        """关闭gazebo仿真"""
        self.__init_gazebo()
        if self._simulator is not None:
            msg = "simulator is closing"
            self._logger4ros.info(msg)
            self.loginfo(msg)

            self.sim_unpause()  # 解除暂停

            self._respawn_goal.close()  # 删除目标区模型

            gazebo_delete_model(model_name=self._bot_model_name)  # 删除机器人模型

            self._simulator.terminate()  # 关闭 gazebo 客户端
            if wait4terminate:
                rst = self._simulator.wait()  # 等待(会卡死)
            self._simulator = None
        return

    def close(self):
        self.gazebo_close()  # 关闭客户端
        self.__close_log()

    def __del__(self):
        self.close()
        self.__del_rewards()


if __name__ == "__main__":
    raise Exception(f"不允许直接运行 {__file__}")
