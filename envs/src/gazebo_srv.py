# 在这里处理基于 ros api 的交互流程
from copy import deepcopy
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import *
from gazebo_msgs.msg import *
from .utils import _logging


LOGGER_NAME_ROS_SERVER = rospy.service.logger.name  # "rospy.service"
LOGGER_NAME_ROSOUT = "rosout"

LEVEL2LOGGING_LEVEL = {
    _logging.DEBUG: _logging.DEBUG,
    _logging.INFO: _logging.INFO,
    _logging.WARNING: _logging.WARNING,
    _logging.ERROR: _logging.ERROR,
    _logging.CRITICAL: _logging.CRITICAL,
    #
    rospy.DEBUG: _logging.DEBUG,
    rospy.INFO: _logging.INFO,
    rospy.WARN: _logging.WARNING,
    rospy.ERROR: _logging.ERROR,
    rospy.FATAL: _logging.CRITICAL,
    #
    "debug": _logging.DEBUG,
    "info": _logging.INFO,
    "warning": _logging.WARNING,
    "error": _logging.ERROR,
    "critical": _logging.CRITICAL,
}

LOGGING_LEVEL2ROSPY_LOG_FUNC = {
    _logging.DEBUG: rospy.logdebug,
    _logging.INFO: rospy.loginfo,
    _logging.WARNING: rospy.logwarn,
    _logging.ERROR: rospy.logerr,
    _logging.CRITICAL: rospy.logfatal,
}


def rosout_child_quickReset(
    suffix: str,
    level: Optional[Union[int, str]] = _logging.INFO,
    fmt: str = _logging.DEFAULT_LOG_FORMAT,
    log_fname: str = None,
    append=True,
) -> _logging.Logger:
    if level is not None:
        level = LEVEL2LOGGING_LEVEL[level]
    logger = _logging.logger_quickReset(
        name=f"{LOGGER_NAME_ROSOUT}.{suffix}",
        level=level,
        fmt=fmt,
        log_fn=log_fname,
        append=append,
        to_stdout=False,
        to_stderr=False,
    )
    return logger


import logging


def run_proxy(
    proxy: rospy.ServiceProxy,
    proxy_args: List[Any] = [],
    proxy_kwargs: Dict[str, Any] = {},
    wait_service=True,
    wait_timeout: float = None,
    logger_name: str = None,  # rosout child logger suffix
):
    # if logger_name:
    #     handlers = logging.getLogger(f"rosout").getChild(logger_name).handlers
    #     print("{")
    #     for hd in handlers:
    #         print(hd)
    #     print("}")
    #     pass
    target_name = f"service {proxy.resolved_name}"
    suf = []
    if len(proxy_args):
        suf.append(f"args={proxy_args}")
    if len(proxy_kwargs):
        suf.append(f"kwargs={proxy_kwargs}")
    if len(suf):
        target_name += " with " + (",".join(suf))
    try:
        assert not rospy.is_shutdown(), "rospy is shutdown"
        if wait_service:
            rospy.logdebug(f"{target_name} is waiting", logger_name=logger_name)
            proxy.wait_for_service(wait_timeout)  # 等待
        resp = proxy(*proxy_args, **proxy_kwargs)
        return resp
    except rospy.ServiceException as e:
        msg = f"{target_name} failed:{e}"
        rospy.logerr(msg, logger_name=logger_name)
        raise rospy.ServiceException(msg)


def gazebo_reset(wait_timeout: float = None, logger_name: str = None):
    proxy = rospy.ServiceProxy("gazebo/reset_simulation", Empty)
    run_proxy(proxy=proxy, wait_timeout=wait_timeout, logger_name=logger_name)


def gazebo_pause(logger_name: str = None):
    """仿真暂停,警告:暂停后所有依赖 wait_for_message 的服务都会陷入等待"""

    proxy = rospy.ServiceProxy("gazebo/pause_physics", Empty)
    run_proxy(proxy=proxy, wait_service=False, logger_name=logger_name)


def gazebo_unpause(
    proxy: rospy.ServiceProxy = None,
    logger_name: str = None,
):
    proxy = rospy.ServiceProxy("gazebo/unpause_physics", Empty)
    run_proxy(proxy=proxy, wait_service=False, logger_name=logger_name)


def gazebo_get_ModelStates(wait_timeout: float = None, logger_name: str = None):
    target = "service gazebo/model_states"
    rospy.logdebug(f"{target} is waiting")
    try:
        ms: ModelStates = rospy.wait_for_message(
            "gazebo/model_states", ModelStates, timeout=wait_timeout
        )
    except rospy.ServiceException as e:
        msg = f"{target} failed: {e}"
        rospy.logerr(msg, logger_name=logger_name)
    return ms


def gazebo_get_ModelState(
    model_name: str,
    wait_timeout: float = None,
    logger_name: str = None,
):
    proxy = rospy.ServiceProxy("gazebo/get_model_state", GetModelState)
    target_name = f"service {proxy.resolved_name} for model_name '{model_name}'"
    req = GetModelStateRequest()
    req.model_name = model_name
    try:
        resp: GetModelStateResponse = run_proxy(
            proxy,
            proxy_args=[req],
            wait_service=True,
            wait_timeout=wait_timeout,
            logger_name=logger_name,
        )
    except rospy.ServiceException as e:
        msg = f"{target_name} failed: {e}"
        rospy.logerr(msg, logger_name=logger_name)
        raise rospy.ServiceException(msg)
    ms = ModelState()
    ms.model_name = model_name
    ms.pose = deepcopy(resp.pose)
    ms.twist = deepcopy(resp.twist)
    return ms


def gazebo_get_WorldProperties(
    wait_timeout: float = None, logger_name: str = None
) -> GetPhysicsPropertiesResponse:
    proxy = rospy.ServiceProxy("/gazebo/get_physics_properties", GetPhysicsProperties)
    return run_proxy(
        proxy, wait_service=True, wait_timeout=wait_timeout, logger_name=logger_name
    )


def gazebo_is_paused(wait_timeout: float = 2.0, logger_name: str = None):
    try:
        pause = gazebo_get_WorldProperties(
            wait_timeout=wait_timeout, logger_name=logger_name
        ).pause
    except rospy.ServiceException as e:
        rospy.logwarn(e)
        pause = True
    return pause


def gazebo_check_ModelProperties(
    model_name: str,
    wait_timeout: float = None,
    logger_name: str = None,
) -> bool:
    proxy = rospy.ServiceProxy("/gazebo/get_model_properties", GetModelProperties)
    target = f"service '{proxy.resolved_name}' for model_name '{model_name}'"

    req = GetModelPropertiesRequest()
    req.model_name = model_name
    try:
        proxy.wait_for_service(wait_timeout)
        resp: GetModelPropertiesResponse = proxy(req)

        print(f"body_names:{np.asarray(resp.body_names)}")
        print(f"canonical_body_name:{np.asarray(resp.canonical_body_name)}")
        print(f"geom_names:{np.asarray(resp.geom_names)}")
        print(f"child_model_names:{np.asarray(resp.child_model_names)}")
        print(f"joint_names:{np.asarray(resp.joint_names)}")
        print(f"is_static:{resp.is_static}")
        print(f"status_message:{resp.status_message}")
        print(f"parent_model_name:{resp.parent_model_name}")

        # Check if the model has links
        if resp.success:
            # Check if all components are loaded for each link
            for link in resp.body_names:
                components_loaded = "visual" in link and "collision" in link
                if not components_loaded:
                    # At least one link is not fully loaded
                    rospy.logwarn(
                        f"Not all components are loaded for link {link}",
                        logger_name=logger_name,
                    )
                    return False

            # All links have all components loaded
            rospy.loginfo(
                f"All components are loaded for the model {model_name}",
                logger_name=logger_name,
            )
            return True
        else:
            rospy.logerr(f"{target} failed", logger_name=logger_name)
            return False

    except rospy.ServiceException as e:
        rospy.logerr(f"{target} failed", logger_name=logger_name)
        return False


def gazebo_check_ModelState(
    model_name: str,
    wait_timeout: float = None,
    logger_name: str = None,
):
    ms = gazebo_get_ModelStates(wait_timeout=wait_timeout, logger_name=logger_name)
    for name in ms.name:
        if name == model_name:
            return True
    return False


def gazebo_delete_model(
    model_name: str,
    wait_timeout: float = None,
    logger_name: str = None,
):
    proxy = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    # 注意:这个过程只是提交了一个删除模型的请求,并不会阻塞到后续的清理工作完成
    run_proxy(
        proxy=proxy,
        proxy_args=[model_name],
        wait_timeout=wait_timeout,
        logger_name=logger_name,
    )
