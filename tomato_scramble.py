import math
import threading
from collections import OrderedDict
import numpy as np


# 模块级缓存：相同尺寸的图片共享同一条 Gilbert 曲线（LRU 淘汰）
_curve_cache: OrderedDict = OrderedDict()
_curve_lock = threading.Lock()
_CACHE_MAX = 8


def _build_gilbert2d(width, height):
    """生成 Gilbert 2D 空间填充曲线的位置数组（迭代式栈模拟，避免 Python 递归开销）"""
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(f"width 和 height 必须为正整数，收到: width={width}, height={height}")
    positions = np.empty(width * height, dtype=np.int64)
    pos = 0

    # 用显式栈代替递归
    if width >= height:
        stack = [(0, 0, width, 0, 0, height)]
    else:
        stack = [(0, 0, 0, height, width, 0)]

    while stack:
        x, y, ax, ay, bx, by = stack.pop()
        w = abs(ax + ay)
        h = abs(bx + by)
        dax = int(math.copysign(1, ax)) if ax != 0 else 0
        day = int(math.copysign(1, ay)) if ay != 0 else 0
        dbx = int(math.copysign(1, bx)) if bx != 0 else 0
        dby = int(math.copysign(1, by)) if by != 0 else 0

        if h == 1:
            for i in range(w):
                positions[pos] = x + y * width
                pos += 1
                x += dax
                y += day
            continue

        if w == 1:
            for i in range(h):
                positions[pos] = x + y * width
                pos += 1
                x += dbx
                y += dby
            continue

        ax2 = ax // 2
        ay2 = ay // 2
        bx2 = bx // 2
        by2 = by // 2

        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)

        if 2 * w > 3 * h:
            if (w2 & 1) == 1 and w > 2:
                ax2 += dax
                ay2 += day
            # 注意入栈顺序与递归调用相反（后进先出）
            stack.append((x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by))
            stack.append((x, y, ax2, ay2, bx, by))
        else:
            if (h2 & 1) == 1 and h > 2:
                bx2 += dbx
                by2 += dby
            stack.append((
                x + (ax - dax) + (bx2 - dbx),
                y + (ay - day) + (by2 - dby),
                -bx2, -by2,
                -(ax - ax2), -(ay - ay2)
            ))
            stack.append((x + bx2, y + by2, ax, ay, bx - bx2, by - by2))
            stack.append((x, y, bx2, by2, ax2, ay2))

    return positions


def _get_curve(width, height):
    """获取 Gilbert 曲线位置数组，带 LRU 缓存（线程安全）"""
    key = (width, height)
    with _curve_lock:
        if key in _curve_cache:
            _curve_cache.move_to_end(key)  # 命中时移到末尾，标记为最近使用
            return _curve_cache[key]
    # 构建曲线在锁外执行，避免长时间持锁
    curve = _build_gilbert2d(width, height)
    with _curve_lock:
        if key not in _curve_cache:  # double-check，防止重复构建
            if len(_curve_cache) >= _CACHE_MAX:
                _curve_cache.popitem(last=False)
            _curve_cache[key] = curve
        else:
            _curve_cache.move_to_end(key)
            curve = _curve_cache[key]
    return curve


class TomatoScramble:
    """Tomato Scramble - Gilbert 2D curve + golden ratio offset (optimized)"""

    def __init__(self, pixels, width, height, key=1.0):
        # 参数边界校验
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"width 和 height 必须为正整数，收到: width={width}, height={height}")
        if not isinstance(key, (int, float)) or not math.isfinite(key) or key <= 0:
            raise ValueError(f"key 必须为有限正数，收到: {key}")
        expected_count = width * height
        if len(pixels) != expected_count:
            raise ValueError(f"pixels 长度 ({len(pixels)}) 与 width*height ({expected_count}) 不一致")

        self.pixels = pixels
        self.width = width
        self.height = height
        self.pixel_count = expected_count
        self.offset = round((math.sqrt(5) - 1) / 2 * self.pixel_count * key) % self.pixel_count

    def encrypt(self):
        pos = _get_curve(self.width, self.height)
        offset = self.offset
        pc = self.pixel_count

        # 直接用 numpy 数组操作，全程不转 tuple
        px = np.asarray(self.pixels, dtype=np.uint8).reshape(pc, -1)
        new_pixels = np.empty_like(px)

        lp = pc - offset
        new_pixels[pos[offset:]] = px[pos[:lp]]
        new_pixels[pos[:pc - lp]] = px[pos[lp:]]

        return new_pixels

    def decrypt(self):
        pos = _get_curve(self.width, self.height)
        offset = self.offset
        pc = self.pixel_count

        px = np.asarray(self.pixels, dtype=np.uint8).reshape(pc, -1)
        new_pixels = np.empty_like(px)

        lp = pc - offset
        new_pixels[pos[:lp]] = px[pos[offset:]]
        new_pixels[pos[lp:]] = px[pos[:pc - lp]]

        return new_pixels
