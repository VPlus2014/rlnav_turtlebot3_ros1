#!/usr/bin/env python
import logging

from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple
from numpy.typing import NDArray

import numpy as np
import rospy
import random
import time
import os
from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
    SpawnModelResponse,
    DeleteModelResponse,
)
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates, WorldState, ModelState
from geometry_msgs.msg import Pose
from .utils import *
from .gazebo_srv import *

_SupportedRegion = Union[SimplestCircle, SimplestRectangle]


class SquareRoomInfo4stage:
    def __init__(self, stage: int):
        self.WALL_LENGTH = 4.0
        self.WALL_WIDTH = 0.25
        self.OBSTACLES_NAMED_GEOM: Dict[str, _SupportedRegion] = {
            "obstacle1": SimplestCircle(0.6, 0.6, 0.15),
            "obstacle2": SimplestCircle(0.6, -0.6, 0.15),
            "obstacle3": SimplestCircle(-0.6, -0.6, 0.15),
            "obstacle4": SimplestCircle(-0.6, 0.6, 0.15),
        }
        self.ROOM_INNER_LENGTH = self.WALL_LENGTH - 2 * self.WALL_WIDTH
        self.ROOM_INNER_X_MIN = -self.ROOM_INNER_LENGTH * 0.5
        self.ROOM_INNER_X_MAX = self.ROOM_INNER_LENGTH * 0.5
        self.ROOM_INNER_Y_MIN = -self.ROOM_INNER_LENGTH * 0.5
        self.ROOM_INNER_Y_MAX = self.ROOM_INNER_LENGTH * 0.5
        pass


def too_closed(x1, y1, x2, y2, dx, dy):
    return abs(x1 - x2) <= dx and abs(y1 - y2) <= dy


def any_collide(r: _SupportedRegion, regions: List[_SupportedRegion]):
    for region in regions:
        if geom_collide2D(r, region):
            return True
    return False


class Respawn:
    """
    管理目标区的生成,修复了一些bug,现在在构造期间不依赖 ros, 所有服务代理临时创建
    """

    def __init__(self, stage: int, logger_name: str, debug: bool = True):
        self._typename = type(self).__name__
        self.__debug = debug
        self.rosout_suffix = logger_name

        self.modelPath = os.path.join(
            os.path.dirname(__file__),
            "turtlebot3_simulations",
            "turtlebot3_gazebo",
            "models",
            "turtlebot3_square",
            "goal_box",
            "model.sdf",
        )
        self.model_name = "goal"  # 读取描述文件可得
        with open(self.modelPath, "r") as f:
            self.model_sdf = f.read()
        self.loginfo(f"{self._typename} model '{self.model_name}'<<'{self.modelPath}'")
        # 模型尺寸信息,查阅 $modelPath 文件
        self._collision_size_x = 0.5
        self._collision_size_y = 0.5

        self._stage = stage
        self.roominfo = SquareRoomInfo4stage(stage=stage)

        # 初始位置
        self._init_goal_x = 0.6
        self._init_goal_y = 0.0

        # 当前位姿
        self._goal_pose = Pose()
        self.setPosition(self._init_goal_x, self._init_goal_y)
        self._collision_box = self.new_collision_box(
            self._init_goal_x,
            self._init_goal_y,
        )
        self.__init_generate_rules()

        self._model_exist = False  # 模型是否已加载
        return

    def model_exist(self):
        return gazebo_check_ModelState(self.model_name, logger_name=self.rosout_suffix)

    def new_collision_box(self, x: float = None, y: float = None, scale: float = 1):
        goal_x, goal_y = self.getPosition()
        if x is None:
            x = goal_x
        if y is None:
            y = goal_y
        sx = self._collision_size_x * scale
        sy = self._collision_size_y * scale
        rc = SimplestRectangle.from_gazebo_gemotry(x, y, sx, sy)
        return rc

    def checkModel(self, ms: ModelStates):
        model_exist = False
        for i, model_name in enumerate(ms.name):
            if model_name == self.model_name:
                model_exist = True
                break
        self._model_exist = model_exist
        return model_exist

    def logdebug(self, msg: str):
        rospy.logdebug(msg, logger_name=self.rosout_suffix)

    def loginfo(self, msg: str):
        rospy.loginfo(msg, logger_name=self.rosout_suffix)

    def logwarn(self, msg: str):
        rospy.logwarn(msg, logger_name=self.rosout_suffix)

    def logerr(self, msg: str):
        rospy.logerr(msg, logger_name=self.rosout_suffix)

    def respawnModel(self):
        self.loginfo(f"respawn model '{self.model_name}'")
        proxy = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        target = f"{self._typename} service {proxy.resolved_name} for {self.model_name}"
        pose = self._goal_pose
        if not self.model_exist():
            if self.__debug:
                self.logdebug(f"{target} is waiting")
            proxy.wait_for_service()
            rsp: SpawnModelResponse = proxy(
                self.model_name, self.model_sdf, "robotos_name_space", pose, "world"
            )
            if not rsp.success:
                msg = f"{target} failed: {rsp.status_message}"
                self.logwarn(msg)

            self.loginfo(
                "Goal init position: {:+0.4f}, {:+0.4f}".format(
                    pose.position.x,
                    pose.position.y,
                )
            )

            self._wait_exist(target=True)
            time.sleep(0.5)

            if self.__debug:
                real_pos = gazebo_get_ModelState(
                    model_name=self.model_name,
                    logger_name=self.rosout_suffix,
                ).pose.position
                self.logdebug(
                    "Goal real position: {:+0.4f}, {:+0.4f}".format(
                        real_pos.x, real_pos.y
                    )
                )

        assert self.model_exist(), f"model {self.model_name} not exist after sespawn"
        return

    def moveModel(self, x: float, y: float):
        raise NotImplementedError("不支持直接移动位置,显示有问题")
        new_ms = ModelState()
        new_ms.model_name = self.model_name
        new_ms.pose = deepcopy(self._goal_pose)
        new_ms.pose.position.x = x
        new_ms.pose.position.y = y
        print(new_ms)

        try:
            srv_ = None
            self.loginfo(f"{self._typename} service {srv_.resolved_name} is waiting")
            srv_.wait_for_service()
            resp = srv_(new_ms)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return

    def _wait_exist(self, target: bool, echo=True):
        if echo:
            self.logdebug(f"waiting model_exist={target}")
        if self.model_exist() != target:
            time.sleep(0.010)

    def deleteModel(self):
        if self.model_exist():
            self.loginfo(f"delete model '{self.model_name}'")
            gazebo_delete_model(self.model_name)
            self._wait_exist(False)
            time.sleep(0.5)

        assert not self.model_exist(), f"model '{self.model_name}' exist after delete!"
        return

    def __update_tabu_regions(self):
        x, y = self.getPosition(update=False)
        obj = self.__tabu_regions[self.model_name]
        if self.__debug:
            msg = [
                f"move tabu region '{self.model_name}'",
                f"from\t{obj}",
            ]
        geom_move_center(obj, x, y, inplace=True)
        if self.__debug:
            msg.append(f"to\t{obj}")
            self.logdebug("\n".join(msg))

    def __init_generate_rules(self):
        tabu_regions: Dict[str, _SupportedRegion] = {}

        # stage<=4: 禁止与障碍物碰撞
        for k, v in self.roominfo.OBSTACLES_NAMED_GEOM.items():
            tabu_regions[k] = v
        # 禁止与当前位置碰撞
        tabu_regions[self.model_name] = self._collision_box

        self.__tabu_regions = tabu_regions

        # 预设目标点位置
        self.__candidate_xys4stage4: List[Tuple[float, float]] = [
            (0.6, 0),
            (1.9, -0.5),
            (0.5, -1.9),
            (0.2, 1.5),
            (-0.8, -0.9),
            (-1, 1),
            (-1.9, 1.1),
            (0.5, -1.5),
            (2, 1.5),
            (0.5, 1.8),
            (0, -1),
            (-0.1, 1.6),
            (-2, -0.8),
        ]
        # 当前所选的索引
        self.__candidate_index = 0
        return

    def genNewPosition(
        self,
        rng: np.random.Generator,
        obj2place: _SupportedRegion,
        regions_must_collide_any: Optional[List[_SupportedRegion]] = None,
        regions_must_collide_none: Optional[List[_SupportedRegion]] = None,
        update_index=False,
        max_attempts: int = 10000,
    ) -> Tuple[float, float]:
        r"""
        在给定约束下随机生成一个可行位置,返回几何中心坐标
        @param regions_must_collide_any: 若为 None|[] 则表示没有限制, 非空则表示必须与其中至少一个区域碰撞
        @param regions_must_collide_none: 若为 None|[] 则表示没有限制, 非空则表示不能与其中任何一个区域碰撞
        @param max_attempts: 最大随机次数, 超过则报错
        """
        if regions_must_collide_any is None:
            regions_must_collide_any = []
        if regions_must_collide_none is None:
            regions_must_collide_none = []
        params = (
            ("obj2place", obj2place),
            ("regions_must_collide_any", regions_must_collide_any),
            ("regions_must_collide_none", regions_must_collide_none),
        )

        #
        obj2place = deepcopy(obj2place)
        bbxmin, bbymin, bbxmax, bbymax = obj2place.bounds
        bbwidth_half = (bbxmax - bbxmin) * 0.5
        bbheight_half = (bbymax - bbymin) * 0.5
        # 简单的可行域检查
        is_in_square_room = self._stage != 4
        if is_in_square_room:
            cx_min = self.roominfo.ROOM_INNER_X_MIN + bbwidth_half
            cy_min = self.roominfo.ROOM_INNER_Y_MIN + bbheight_half
            cx_max = self.roominfo.ROOM_INNER_X_MAX - bbwidth_half
            cy_max = self.roominfo.ROOM_INNER_Y_MAX - bbheight_half
            if cx_min > cx_max or cy_min > cy_max:
                msg = (
                    f"{self.genNewPosition.__name__} with {params} failed:"
                    f"obj2place {obj2place} is too large to place in the room with "
                )
                msg += ", ".join(
                    f"cx_min={cx_min}",
                    f"cy_min={cy_min}",
                    f"cx_max={cx_max}",
                    f"cy_max={cy_max}",
                )
                self.logerr(msg)
                raise Exception(msg)

        index = self.__candidate_index
        for itr in range(max_attempts):
            position_invalid = False

            # 1. 产生位置
            if is_in_square_room:
                new_x = affine_comb(cx_min, cx_max, rng.random()).item()
                new_y = affine_comb(cy_min, cy_max, rng.random()).item()
                geom_move_center(obj2place, new_x, new_y, inplace=True)
                #
                for region_name, region in self.__tabu_regions.items():
                    if geom_collide2D(region, obj2place):
                        position_invalid = True
                        if self.__debug:
                            self.logdebug(
                                f"itr {itr}, collide with {region_name}:{region}"
                            )
                        break
            else:
                index = rng.choice(len(self.__candidate_xys4stage4))
                if self.__candidate_index == index:  # 重复位置
                    position_invalid = True
                    continue
                new_x, new_y = self.__candidate_xys4stage4[index]
            #
            if position_invalid:
                continue

            # 2. 对额外约束进行检查
            if len(regions_must_collide_any) and not any_collide(
                obj2place, regions_must_collide_any
            ):
                if self.__debug:
                    self.logdebug(
                        f"itr {itr}, not collide with regions_must_collide_any"
                    )
                position_invalid = True
                continue
            if len(regions_must_collide_none) and any_collide(
                obj2place, regions_must_collide_none
            ):
                if self.__debug:
                    self.logdebug(f"itr {itr}, collide with regions_must_collide_none")
                position_invalid = True
                continue

            if not position_invalid:
                break
        else:
            msg = f"{self.genNewPosition.__name__} out of time limit {max_attempts} with params={params}"
            self.logerr(msg)
            raise Exception(msg, params, max_attempts)
        #
        if update_index:
            self.__candidate_index = index
        return new_x, new_y

    def setPosition(self, x: float, y: float):
        self._goal_pose.position.x = x
        self._goal_pose.position.y = y

    def getPosition(self, update=False):
        if update:
            if self.__debug:
                self.logdebug(
                    f"{self._typename} is reading current position"
                    f"for model {self.model_name}"
                )
            ms = gazebo_get_ModelState(self.model_name, logger_name=self.rosout_suffix)
            self._goal_pose = ms.pose  # 设置位姿
            self.__update_tabu_regions()  # 更新碰撞
        pos = self._goal_pose.position
        return pos.x, pos.y

    def regen(
        self,
        rng: np.random.Generator,
        regions_must_collide_any: Optional[List[_SupportedRegion]] = None,
        regions_must_collide_none: Optional[List[_SupportedRegion]] = None,
        max_attempts: int = 10000,
    ):
        if self.__debug:
            self.logdebug(
                f"{self._typename} generating random position for model {self.model_name}"
            )
        # 产生新位置
        goal_x, goal_y = self.genNewPosition(
            rng=rng,
            obj2place=self.new_collision_box(scale=1.1),
            regions_must_collide_any=regions_must_collide_any,
            regions_must_collide_none=regions_must_collide_none,
            update_index=True,
            max_attempts=max_attempts,
        )
        self.setPosition(goal_x, goal_y)

        self.deleteModel()
        self.respawnModel()

        self.__update_tabu_regions()
        return goal_x, goal_y

    def close(self):
        if not rospy.is_shutdown():
            self.deleteModel()

    def __del__(self):
        self.close()
