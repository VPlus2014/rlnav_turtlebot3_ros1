# 说明

## 0. 安装前处理

为方便后续操作，运行 ``[项目文件夹]/envs/reinstall.py`` 来自定义环境变量：

``TURTLEBOT3_CATKIN_WS`` 为 ``[项目文件夹]/envs`` 表示仿真环境部分的目录；
```TURTLEBOT3_MODEL``` 表示 turtlebot3 仿真默认的机器人模型；

该程序会将这些环境变量自动添加到 ```~/.bash``` 中，它等效于每次启动终端时使用如下命令

```bash
export TURTLEBOT3_MODEL=burger
export TURTLEBOT3_CATKIN_WS=~/"uv_turtlebot3_1/envs"
source $TURTLEBOT3_CATKIN_WS/devel/setup.bash
```

其中 ```source $TURTLEBOT3_CATKIN_WS/devel/setup.bash``` 作用是让一些 ROS* 开头的命令可以使用。

在旧终端 ```source ~/.bashrc``` /打开新终端，用如下命令测试环境变量是否成功定义

```bash
echo $TURTLEBOT3_MODEL
echo $TURTLEBOT3_CATKIN_WS
```

## 1. 安装ROS

本项目基于 ros1 ，版本为 noetic

<https://blog.csdn.net/zyh821351004/article/details/130180172>

### 1.1  
#### 1.1.1 鱼香ROS大佬的一键安装

```bash
wget http://fishros.com/install -O fishros && . fishros
```

### 1.1.2 mamba

像 conda 一样管理多个环境

<https://robostack.github.io/GettingStarted.html#__tabbed_1_1>


### 1.2 rosdep 初始化

```bash
sudo rosdep init
rosdep update
```

#### 1.2.x 初始化失败

因为墙的问题导致不能使用rosdep init ，参考 <https://www.guyuehome.com/35408> 安装使用 rosdepc ，然后再初始化

```bash
sudo pip install rosdepc
sudo rosdep init
sudo rosdep update
```

## 2. 项目依赖与编译

### 2.0 下载 turtlebot3 必要包

```bash
cd $TURTLEBOT3_CATKIN_WS/src
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
```

后续在此基础上改了模型描述文件

### 2.1 自动下载项目的 ros 依赖项

下载 ``$TURTLEBOT3_CATKIN_WS/src/`` 中的ros依赖项

```bash
cd $TURTLEBOT3_CATKIN_WS
rosdep install -i --from-path src --rosdistro noetic -y
```

### 2.2 手动下载其他依赖项

ros noetic 的其他依赖项

```bash
sudo apt-get install ros-noetic-tf
sudo apt-get install ros-noetic-gazebo-*
sudo apt-get install rviz # 好用的可视化工具
```

### 2.3 下载 python 依赖

```bash
pip install em # 编译依赖
pip install empy # 编译依赖
pip install catkin_pkg # 编译依赖
pip install rospkg # ROS 编译依赖
pip install transformations # 坐标变换依赖
pip install defusedxml
pip install netifaces
pip install seaborn # 绘图需要
pip install stable-baselines3[extra] # RL训练需要
```

没有这些依赖项后面没法编译通过。

### 2.4 编译环境

#### 2.4.1 安装编译工具

ros1 用 catkin

```bash
sudo apt-get install python-catkin-tools
```

参考
<https://blog.csdn.net/u014603518/article/details/127717928>
<https://blog.csdn.net/qq_29912325/article/details/130260527>

#### 2.4.2 重新编译

删除 build、devel 文件夹(由cmake自动生成，不要在这里放非自动生成的代码、数据!)

```bash
cd $TURTLEBOT3_CATKIN_WS
rm -r ./build ./devel

catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.8
```

- ``-DPYTHON_EXECUTABLE=/usr/bin/python3`` 使用系统自带的 python3.8 ，这可确保编译不报错。如果只用 ```catkin_make``` ，则使用 conda 当前 python 环境，可能出现不兼容问题，有奇怪的报错内容比如 ```Invoking "make -j16 -l16" failed``` 和 ```Invoking "cmake" failed``` ，后续模型无法正常加载。
- 自动生成的关键脚本 ```$TURTLEBOT3_CATKIN_WS/devel/setup.sh``` 中有多处将 ```~``` 展开，导致项目迁移到其他机器后不能直接运行，所以需要重新编译。

## 3. 准备启动 gazebo 仿真

### 3.1 必要项设置

在终端启动仿真前需要设置的一些临时变量，也可写入 ```~/.bashrc``` 中作为默认设置，避免重复敲命令。

#### 3.1.1 机器人模型申明

选择其中一种机器人模型申明，用以设定打开后模型的样子

```bash
export TURTLEBOT3_MODEL=burger
export TURTLEBOT3_MODEL=waffle
export TURTLEBOT3_MODEL=waffle_pi
```

#### 3.1.2 设置 ROS 工作区

```bash
source $TURTLEBOT3_CATKIN_WS/devel/setup.bash 
```

作用是让一些 ROS* 开头的命令可以正常使用，比如搜索当前 ```$TURTLEBOT3_CATKIN_WS``` 下的各种模型描述文件。

- 注意，如果把多个项目的这种 setup 指令写入 ```~/.bashrc``` ，在后续启动仿真前宏指令搜索 launch、xacro 等文件时容易出现歧义。

##### 检查

```bash
echo $ROS_PACKAGE_PATH
```

显示结果可能有多个路径，以冒号分隔，只要包含 ```$TURTLEBOT3_CATKIN_WS/src``` 就表明设置成功

##### 卸载当前包

```bash
unset ROS_PACKAGE_PATH
```

### 3.2 非必要项设置

#### 3.2.1 日志重定向

<https://zhuanlan.zhihu.com/p/378175365?utm_id=0>

环境变量 ```ROS_LOG_DIR``` 和 ```ROS_HOME``` 默认为空，此时 ros 日志会输出到 ```~/.ros/log``` 。若设置为非空目录，则 ros 日志输出到 ```$ROS_LOG_DIR``` ，例如

```bash
export ROS_LOG_DIR=$TURTLEBOT3_CATKIN_WS/log
```

### 3.3 打开 roscore

如果使用 roslaunch 启动仿真，则会自动启动 roscore ，可跳过此步。
启动 rosmaster 需要使用 roscore 指令，该指令除了启动 rosmaster ，还会一并启动 rosout 和 parameter server ，前者主要负责日志输出，后者则是参数服务器。
初始化 ros 节点前必须保证 roscore 是启动的。

```bash
roscore # 常规启动，占用一个终端
roscore & # 后台启动，不占用终端但是没法 ctrl+c 中断
```

#### 异常处理

仿真异常中断时，需要杀死 ros 进程才能重新启动 roscore

```bash
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient
```

### 3.4 启动 Gazebo 仿真

打开新终端，输入命令启动 Gazabo ，通过 roslaunch 命令启动 src 文件夹中特定节点，例如启动 ```turtlebot3_stage_2.launch``` 。

- 相对路径形式。格式为 ```roslaunch [package] [filename]``` ，会在 ```$ROS_PACKAGE_PATH``` 下按深度优先遍历查找文件

```bash
roslaunch turtlebot3_gazebo turtlebot3_stage_2.launch
```

- 绝对路径形式。当同一个 launch 文件名出现在 ```$ROS_PACKAGE_PATH``` 中的多处，那么上一种调用会有歧义，提示你用绝对路径来消除歧义。

```bash
roslaunch $TURTLEBOT3_CATKIN_WS/src/envs/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_stage_2.launch
```

### 3.5 [非必要]人工介入控制

打开新终端，用键盘控制机器人，该键盘控制是依靠按“W A D X”键增加速度，按S制动。

```bash
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

## 4 训练&评估

```bash
export RL_PROJ="$TURTLEBOT3_CATKIN_WS/.."
```

### 4.1 运行训练程序

```bash
python $TURTLEBOT3_CATKIN_WS/src/rlalgos/TD3/train_stage_2_td3.py
```

每次运行结果保存在 $TURTLEBOT3_CATKIN_WS/results/Env_stage_2_cont_action/TD3/[时间戳] 下
 models 是模型参数文件夹
 rewards 是平均回报数据
 log 是其他训练数据，包括 Q 值， TD 误差等，用tensorboard查看

tensorboard 每一次训练的数据记录在 ```$TURTLEBOT3_CATKIN_WS/log/DDPG/[日期]``` 下
模型参数读写在 ```$TURTLEBOT3_CATKIN_WS/src/rlalgos/saved_models/DDPG``` 下

### 4.2 训练过程可视化

训练过程中使用了 SummaryWritter ，使用 tensorboard 在 <http://localhost:6006/> 查看训练过程，占用一个终端

```bash
tensorboard --logdir tensorboard --logdir $TURTLEBOT3_CATKIN_WS/../log/
```

#### 正则表达式速查

- 无折扣累计回报 ```episode_rewards/[^/]+/undiscounted/unweighted/```
- 无折扣加权累计回报 ```episode_rewards/[^/]+/undiscounted/weighted/```
- 折扣累计回报 ```episode_rewards/[^/]+/discounted/unweighted/```
- 加权折扣累计回报 ```episode_rewards/[^/]+/discounted/weighted/```
- 最终折扣累计回报 ```episode_rewards/[^/]+/discounted/final```
- 最终无折扣累计回报 ```episode_rewards/[^/]+/undiscounted/final```

- 只看加权、归一化的折扣累计回报 ```episode_rewards/total/discounted/(weighted|final)$```
- 只看加权折扣回报下除线速度惩罚(目前不控制线速度所以无奖励)的所有项,以及最终的无折扣累计收益

```bash
episode_rewards/[^/]+/(discounted/(final|weighted)($|/(?!lin_vel$).*))|(undiscounted/final)
```
