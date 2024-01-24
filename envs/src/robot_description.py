from dataclasses import dataclass
import os
from typing import Union, Dict, Tuple
from .utils import path_parent, path_find_with_interaction
import xml.etree.ElementTree as ET


def element_to_dict(element: ET.Element):
    raise NotImplementedError
    _T = Dict[str, Union[str, Dict]]
    rst: _T = {}
    tag = element.tag
    for child in element:
        rst[tag][child.tag] = element_to_dict(child)
    return rst

    if element.text is not None:
        tag_val = element.text
    else:
        tag_val = {}
        for child in element:
            child_result = element_to_dict(child)
            tag_val[child.tag] = child_result
            # if child.tag in tag_val:
            #     if isinstance(tag_val[child.tag], list):
            #         result[tag][child.tag].append(child_result)
            #     else:
            #         result[tag][child.tag] = [
            #             result[tag][child.tag],
            #             child_result,
            #         ]
            # else:
            #     result[tag].update(child_result)
    rst[tag] = tag_val
    return rst


@dataclass
class Turtlebot3_LaserScan_Attrs:
    ranges_dim: int = 0
    min_angle: float = 0.0
    max_angle: float = 0.0
    update_rate: float = 0.0

    def ranges_dim_setter(self, val):
        self.ranges_dim = type(self.ranges_dim)(val)

    def min_angle_rad_setter(self, val):
        self.min_angle = type(self.min_angle)(val)

    def max_angle_rad_setter(self, val):
        self.max_angle = type(self.max_angle)(val)

    def update_rate_setter(self, val):
        self.update_rate = type(self.update_rate)(val)

    def __str__(self) -> str:
        return " ".join(
            f"{k}: {type(v).__name__} = {v}" for k, v in self.__dict__.items()
        )

    def parse_file(self, fname: str):
        """
        从描述文件获取光雷参数
        """
        # 默认的解码器不能正确读取所有模型参数
        parser: ET.XMLParser = None
        tree = ET.parse(fname, parser)
        root = tree.getroot()

        meta = {
            "samples": self.ranges_dim_setter,
            "min_angle": self.min_angle_rad_setter,
            "max_angle": self.max_angle_rad_setter,
            "update_rate": self.update_rate_setter,
        }

        for elem in root.iter():
            for k, setter in meta.items():
                if elem.tag == k:
                    setter(elem.text)
                    break
        return self


__turtlebot3_model_title = "turtlebot3_"


def turtlebot3_model_suffix(model_name: str = None):
    """获取模型名,若 model_name=None, 则取为 $TURTLEBOT3_MODEL"""
    if model_name is None:
        model_name = os.environ["TURTLEBOT3_MODEL"]  # 读取环境变量定义的模型名
    if model_name.startswith(__turtlebot3_model_title):
        model_name = model_name[len(__turtlebot3_model_title) :]
    return model_name


def turtlebot3_model_name(suffix: str = None):
    suffix = turtlebot3_model_suffix(suffix)
    model_name = __turtlebot3_model_title + suffix
    return model_name


def turtlebot3_urdf_fname(model: str = None) -> str:
    """获取模型描述文件名,
    等价于命令 $(find turtlebot3_description)/urdf/turtlebot3_$(arg model).urdf.xacro"""
    # 实际上需要分析 launch 文件对模型文件的依赖关系
    model = turtlebot3_model_name(model)
    target_fname = f"{model}.gazebo.xacro"
    root_dir = path_parent(__file__)
    fn_description = path_find_with_interaction(root_dir, target_fname)
    assert (
        len(fn_description) > 0
    ), f"failed to find description file {target_fname} in {root_dir}"
    return fn_description
