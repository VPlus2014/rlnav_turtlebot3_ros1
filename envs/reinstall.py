import os
import re
import shutil
from typing import List


def catkin_ws_clean(dir_catkin_ws: str, folders: List[str]):
    # 删除 devel,build
    for dir_ in [os.path.join(dir_catkin_ws, dir_) for dir_ in folders]:
        cmd = f"rm -r {dir_}"
        print("$", cmd)
        try:
            shutil.rmtree(dir_)
        except Exception as e:
            print("failed to delete", dir_, e)

    dir_src = os.path.join(dir_catkin_ws, "src")
    if not os.path.exists(dir_src):
        cmd = f"mkdir -p {dir_src}"
        print("$", cmd)
        os.makedirs(dir_src, exist_ok=True)
    return


def main():
    dir_catkin_ws = os.path.dirname(os.path.realpath(__file__))  # 待编译的包路径
    folder_devel = "devel"
    folder_build = "build"
    folder_src = "src"

    # 检查 src
    if folder_src not in os.listdir(dir_catkin_ws):
        print(f"warning: there is not fold '{folder_src}' in '{dir_catkin_ws}'")
        fn_src = os.path.join(dir_catkin_ws, folder_src)
        arg2acc = "YES"
        rst = input(f"input {arg2acc} to create {fn_src}:")
        if rst != arg2acc:
            print("Installation cancelled")
            return
        os.mkdir(fn_src)

    # 定义环境变量名
    varname_turtlebot_model = "TURTLEBOT3_MODEL"  # turtlebot3 默认的模型名
    varname_env_dir = "TURTLEBOT3_CATKIN_WS"  # 自定义 turtlebot3 包

    bashval_dir_catkin_ws = '~/"{}"'.format(
        dir_catkin_ws.lstrip(os.path.expanduser("~"))
    )
    bashval_dir_setup = os.path.join("$" + varname_env_dir, folder_devel, "setup.bash")
    vars_meta = [
        (varname_turtlebot_model, "burger"),
        (varname_env_dir, bashval_dir_catkin_ws),
    ]
    cmd_source_setup = f"source {bashval_dir_setup}"

    fn_bashrc = os.path.expanduser(os.path.join("~", ".bashrc"))
    with open(fn_bashrc) as f:
        data = f.read()

    cmds2append = []
    # 删除原定义的环境变量对应的行
    for varname, target_val in vars_meta:
        pattern = rf"\n*\s*export {varname}\s*=\s*[^\n]+\n*"
        data = re.sub(pattern, "\n", data)  # 删除
        cmds2append.append(f"export {varname}={target_val}")

    data = data.replace(cmd_source_setup, "\n")
    cmds2append.append(cmd_source_setup)
    # 添加到末尾
    cmds2append = "\n".join(cmds2append)
    data += f"\n\n{cmds2append}\n"

    # 缩减连续换行
    while True:
        data2 = data.replace("\n\n\n", "\n\n")
        if len(data2) == len(data):
            break
        data = data2

    print(
        f"{fn_bashrc} will be changed as follow:", "```", data, "```\n", sep="\n"
    )  # 预览内容

    arg2acc = "YES"
    rst = input(f"input {arg2acc} to accept change:")
    if rst != arg2acc:
        print("change refused")
        return

    catkin_ws_clean(dir_catkin_ws, folders=[folder_build, folder_devel])  # 删除旧目录

    with open(fn_bashrc, "w") as f:  # 重新写入
        f.write(data)
        print(fn_bashrc, "is changed")

    def hint_cmds(cmds: List[str], title: str):
        print("\n" + title)
        print("`" * 3, *cmds, "`" * 3, sep="\n")

    hint_cmds(
        cmds=[
            f"source ~/.bashrc",
            f"echo ${varname_turtlebot_model}",
            f"echo ${varname_env_dir}",
        ],
        title="input the lines in a terminal to update and test:",
    )
    return
    hint_cmds(
        cmds=[
            f"source ~/.bashrc",
            f"cd ${varname_env_dir}",
            f"catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3",
        ],
        title="input the lines in a terminal to rebuild:",
    )
    return


if __name__ == "__main__":
    main()
