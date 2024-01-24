import os
import subprocess
import time
from .utils._path import path_find, path_parent


def ros_launch_turtlebot3_gazebo_stage(
    stage: int, model: str = None
) -> subprocess.Popen:
    """
    用子进程打开 turtlebot3_stage_*.launch 文件
    @param stage: 可选 1,2,3,4
        详见 /turtlebot3_simulations/turtlebot3_gazebo/launch/ 下的文件设定
    @param model: 可选 burger,waffle,waffle_pi, 若为 None 则使用环境变量 $TURTLEBOT3_MODEL
        详见 /turtlebot3/turtlebot3_description/urdf/
        本参数是否可用取决于 launch 文件中 robot_description 是否使用 model,
        例如 stage 1,2 均只使用 burger
    @return: ...
    @rtype: Popen
    """
    fn_target = f"turtlebot3_stage_{stage}.launch"

    flaunch = path_find(path_parent(__file__), fn_target)[0]
    # flaunch = os.path.join(
    #     parent_dir(__file__),
    #     "turtlebot3_simulations",
    #     "turtlebot3_gazebo",
    #     "launch",
    #     f"turtlebot3_stage_{stage}.launch",
    # )
    # 为避免歧义,使用绝对路径
    cmd_launch = [
        "roslaunch",
        flaunch,
    ]
    if model is not None:  # 否则读取环境变量 $TURTLEBOT3_MODEL 作为机器人模型名
        if stage in [1, 2] and model != "burger":
            print(f"{flaunch} 不支持修改模型名 {model}")
        cmd_launch.append(f"model:={model}")  # 机器人模型

    sp = subprocess.Popen(cmd_launch)
    print("$", " ".join(cmd_launch))
    time.sleep(5.0)
    return sp


def killall_gazebo():
    killall(process=["gzclient", "gzserver", "roslaunch"])


def killall(
    process=[
        "rosmaster",
        "rosout",
        "roslaunch",
        "gzclient",
        "gzserver",
        "nodelet",
        "robot_state_publisher",
    ],
    sudo=True,
):
    process = list(set(process))
    assert len(process) > 1
    cmd = ["sudo"] if sudo else []
    cmd.extend(["killall", "-9"])
    cmd.extend(process)
    cmd = " ".join(cmd)
    print("$", cmd)
    os.system(cmd)
    time.sleep(1.0)
    return


def ros_restart():
    killall()
    time.sleep(1.0)
    os.system("roscore &")
    time.sleep(3.0)
