import os
from typing import List, Optional


def path_parent(fn: str, iters: int = 1):
    assert iters >= 0
    fn = os.path.realpath(fn)
    for _ in range(iters):
        fn = os.path.split(fn)[0]
    return fn


def __path_findall(
    root: str,
    posfix: str,
    skip_dir: bool,
    return_first: bool,
) -> List[str]:
    # assert os.path.exists(root_dir)
    rst = []
    root_is_dir = os.path.isdir(root)
    if root.endswith(posfix) and not (skip_dir and root_is_dir):
        rst.append(root)
        if return_first:
            return rst
    if root_is_dir:
        for pfn in os.listdir(root):
            child = os.path.join(root, pfn)
            rst_c = __path_findall(child, posfix, skip_dir, return_first)
            if len(rst_c):
                rst.extend(rst_c)
                if return_first:
                    return rst
    return rst


def path_find(
    root: str,
    posfix: str,
    skip_dir=True,  # 结果不含目录
    return_first=False,  # 找到1个停止
    allow_empty_posfix=False,
) -> List[str]:
    root = root.rstrip(os.sep)
    posfix = posfix.strip(os.sep)
    if not os.path.exists(root):
        print(f"warning: root '{root}'", "is not a existing filename")
        return []
    if len(posfix) == 0 and not allow_empty_posfix:
        raise Exception("empty posfix is not allowed")
    return __path_findall(root, posfix, skip_dir=skip_dir, return_first=return_first)


def path_find_with_interaction(
    root: str,
    posfix: str,
    skip_dir=True,
    return_first=False,
    allow_empty_posfix=False,
) -> Optional[str]:
    rst = path_find(
        root,
        posfix,
        skip_dir=skip_dir,
        return_first=return_first,
        allow_empty_posfix=allow_empty_posfix,
    )
    n_rst = len(rst)
    if n_rst == 1:
        return rst[0]
    if n_rst > 1:
        msg = [f"[{i}]:\t{fn}" for i, fn in enumerate(rst)]
        msg.append(f"{n_rst} results, input a number to select:")
        msg = "\n".join(msg)
        while True:
            ans = input(msg)
            try:
                i = int(ans)
                fn = rst[i]
                print(f"select '{fn}'")
                return fn
            except Exception as e:
                print("invalid input")
    else:
        print((f"warning: no such file '{posfix}' in '{root}'"))
        return None
