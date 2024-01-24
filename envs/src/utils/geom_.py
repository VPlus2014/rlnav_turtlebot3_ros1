from copy import deepcopy
from dataclasses import dataclass
import math
from typing import List, Tuple, Union
import numpy as np


def __sortTuple(x, y):
    if x > y:
        return y, x
    return x, y


# shapely 难用的一匹


class SimplestCircle(object):
    def __init__(self, cx: float, cy: float, radius: float):
        self.cx = float(cx)
        self.cy = float(cy)
        self.radius = radius

    def __str__(self):
        return (
            f"{type(self).__name__}"
            + "("
            + (",".join(f"{atn}={atv}" for atn, atv in self.__dict__.items()))
            + ")"
        )

    @property
    def radius(self):
        return self.__radius

    @radius.setter
    def radius(self, radius: float):
        assert (
            0 < radius < math.inf
        ), f"radius must be positve finite number, but get {radius}"
        self.__radius = float(radius)

    @property
    def bounds(self):
        """Returns minimum bounding region (minx, miny, maxx, maxy)"""
        x = self.cx
        y = self.cy
        r = self.radius
        return (x - r, y - r, x + r, y + r)


class SimplestRectangle(object):
    def __init__(self, xmin: float, ymin: float, width: float, height: float):
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.width = width
        self.height = height

    def __str__(self):
        d = {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "width": self.width,
            "height": self.height,
        }
        return (
            f"{type(self).__name__}"
            + "("
            + (",".join(f"{atn}={atv}" for atn, atv in d.items()))
            + ")"
        )

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, val: float):
        assert 0 < val < math.inf, f"width must be positve finite number, but get {val}"
        self.__width = float(val)

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, val: float):
        assert (
            0 < val < math.inf
        ), f"height must be positve finite number, but get {val}"
        self.__height = float(val)

    @property
    def xmax(self) -> float:
        return self.xmin + self.__width

    @property
    def ymax(self) -> float:
        return self.ymin + self.__height

    @property
    def exterior(self) -> List[Tuple[float, float]]:
        x1 = self.xmin
        y1 = self.ymin
        x2 = self.xmax
        y2 = self.ymax
        xys = (
            (x1, y1),
            (x1, y2),
            (x2, y2),
            (x2, y1),
        )
        return xys

    @property
    def bounds(self):
        """Returns minimum bounding region (minx, miny, maxx, maxy)"""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @staticmethod
    def from_bounds(xmin: float, ymin: float, xmax: float, ymax: float):
        xmin, xmax = __sortTuple(xmin, xmax)
        ymin, ymax = __sortTuple(ymin, ymax)
        rect = SimplestRectangle(xmin, ymin, xmax - xmin, ymax - ymin)
        return rect

    @staticmethod
    def from_gazebo_gemotry(cx: float, cy: float, size_x: float, size_y: float):
        rect = SimplestRectangle(cx - size_x * 0.5, cy - size_y * 0.5, size_x, size_y)
        return rect


def geom_dist1D(x, a, b) -> float:
    a, b = __sortTuple(a, b)
    if x < a:
        return a - x
    if x > b:
        return x - b
    return 0


def geom_dist2D(x: float, y: float, rect: SimplestRectangle) -> float:
    dx = geom_dist1D(x, rect.xmin, rect.xmax)
    dy = geom_dist1D(y, rect.ymin, rect.ymax)
    return math.hypot(dx, dy)


def geom_collide1D(a1: float, b1: float, a2: float, b2: float):
    a1, b1 = __sortTuple(a1, b1)
    a2, b2 = __sortTuple(a2, b2)
    return bool(a1 <= b2 and a2 <= b1)


def geom_contains1D(a1: float, b1: float, a2: float, b2: float):
    a1, b1 = __sortTuple(a1, b1)
    a2, b2 = __sortTuple(a2, b2)
    return a1 <= a2 and b2 <= b1


def geom_collide_Circle(
    lhs: SimplestCircle,
    rhs: SimplestCircle,
):
    return math.hypot(lhs.cx - rhs.cx, lhs.cy - rhs.cy) <= lhs.radius + rhs.radius


def geom_collide_Circle_Rect(
    circle: SimplestCircle,
    rect: SimplestRectangle,
):
    return geom_dist2D(circle.cx, circle.cy, rect) <= circle.radius


def geom_contains_Circle_Rect(
    circle: SimplestCircle,
    rect: SimplestRectangle,
):
    cx = circle.cx
    cy = circle.cy
    r = circle.radius
    for x, y in rect.exterior:
        if math.hypot(cx - x, cy - y) > r:
            return False
    return True


def geom_contains_Rect_Circle(
    rect: SimplestRectangle,
    circle: SimplestCircle,
):
    """rect contains circle?"""
    cx = circle.cx
    cy = circle.cy
    r = circle.radius
    return (rect.xmin + r <= cx <= rect.xmax - r) and (
        rect.ymin + r <= cy <= rect.ymax - r
    )


def geom_contains_Rect(
    lhs: SimplestRectangle,
    rhs: SimplestRectangle,
):
    """lhs contains rhs?"""

    return geom_contains1D(lhs.xmin, lhs.xmax, rhs.xmin, rhs.xmax) and geom_contains1D(
        lhs.ymin, lhs.ymax, rhs.ymin, rhs.ymax
    )


def geom_contains_Circle(
    lhs: SimplestCircle,
    rhs: SimplestCircle,
):
    """lhs contains rhs?"""

    return math.hypot(lhs.cx - rhs.cx, lhs.cy - rhs.cy) + rhs.radius <= lhs.radius


def geom_collide2D(
    lhs: Union[SimplestRectangle, SimplestCircle],
    rhs: Union[SimplestRectangle, SimplestCircle],
):
    if isinstance(lhs, SimplestRectangle):
        if isinstance(rhs, SimplestRectangle):
            return geom_collide1D(
                lhs.xmin, lhs.xmax, rhs.xmin, rhs.xmax
            ) and geom_collide1D(lhs.ymin, lhs.ymax, rhs.ymin, rhs.ymax)
        if isinstance(rhs, SimplestCircle):
            return geom_collide_Circle_Rect(rhs, lhs)
        raise NotImplementedError(f"unsupported rhs={rhs}")
    elif isinstance(lhs, SimplestCircle):
        if isinstance(rhs, SimplestRectangle):
            return geom_collide_Circle_Rect(lhs, rhs)
        if isinstance(rhs, SimplestCircle):
            return geom_collide_Circle(lhs, rhs)
        raise NotImplementedError(f"unsupported rhs={rhs}")
    raise NotImplementedError(f"unsupported lhs={lhs}")


def geom_contains2D(
    lhs: Union[SimplestRectangle, SimplestCircle],
    rhs: Union[SimplestRectangle, SimplestCircle],
):
    if isinstance(lhs, SimplestRectangle):
        if isinstance(rhs, SimplestRectangle):
            return geom_contains_Rect(lhs, rhs)
        if isinstance(rhs, SimplestCircle):
            return geom_contains_Rect_Circle(lhs, rhs)
        raise NotImplementedError(f"unsupported rhs={rhs}")
    elif isinstance(lhs, SimplestCircle):
        if isinstance(rhs, SimplestRectangle):
            return geom_contains_Circle_Rect(lhs, rhs)
        if isinstance(rhs, SimplestCircle):
            return geom_contains_Circle(lhs, rhs)
        raise NotImplementedError(f"unsupported rhs={rhs}")
    raise NotImplementedError(f"unsupported lhs={lhs}")


def geom_move_center(obj, cx: float = 0, cy: float = 0, inplace=True):
    if isinstance(obj, SimplestCircle):
        if not inplace:
            obj = deepcopy(obj)
        obj.cx = cx
        obj.cy = cy
    elif isinstance(obj, SimplestRectangle):
        if not inplace:
            obj = deepcopy(obj)
        obj.xmin = cx - obj.width * 0.5
        obj.ymin = cy - obj.height * 0.5
    else:
        raise NotImplementedError(f"unsupported obj={obj}")
    return obj
