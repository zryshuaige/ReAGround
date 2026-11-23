import ctypes
import os
from dataclasses import dataclass

from PIL import ImageDraw, ImageGrab, Image


@dataclass(frozen=True)
class DesktopImage:
    image: Image.Image
    origin: tuple[int, int]


def enable_dpi_awareness():
    if os.name != "nt":
        return
    try:
        aware = ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        if not aware:
            ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def capture_desktop():
    if os.name == "nt":
        user32 = ctypes.windll.user32
        left, top = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
        width, height = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
        box = (left, top, left + width, top + height)
        return DesktopImage(ImageGrab.grab(bbox=box, all_screens=True).convert("RGB"), (left, top))
    return DesktopImage(ImageGrab.grab().convert("RGB"), (0, 0))


def draw_point(image, x, y):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    x, y = round(x), round(y)
    radius = max(12, min(24, min(image.size) // 80))
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 outline="white", width=7)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 outline="#ef4444", width=3)
    draw.line((x - radius - 8, y, x + radius + 8, y), fill="#ef4444", width=2)
    draw.line((x, y - radius - 8, x, y + radius + 8), fill="#ef4444", width=2)
    label = f"({x}, {y})"
    label_box = draw.textbbox((0, 0), label)
    w, h = label_box[2] - label_box[0], label_box[3] - label_box[1]
    lx = min(max(0, x + radius + 8), max(0, image.width - w - 12))
    ly = min(max(0, y - radius - h - 16), max(0, image.height - h - 12))
    draw.rectangle((lx, ly, lx + w + 12, ly + h + 12), fill="white", outline="#ef4444")
    draw.text((lx + 6, ly + 6), label, fill="#b91c1c")
    return marked


def click_once(point, origin=(0, 0)):
    if os.name != "nt":
        raise NotImplementedError("真实点击目前仅支持 Windows。")
    x = round(point[0] + origin[0])
    y = round(point[1] + origin[1])
    ctypes.windll.user32.SetCursorPos(x, y)
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
