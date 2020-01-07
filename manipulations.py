import numpy as np
import string
from PIL import ImageDraw, ImageFont


class RandomText:
    def __init__(self, p=0.5):
        self.ascii_chars = np.array(
            list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        )
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        font = ImageFont.truetype("fonts/FreeSerif.ttf", np.random.randint(16, 32))
        x = np.random.randint(0, img.width / 2)
        y = np.random.randint(0, img.height)
        strlen = np.random.randint(5, 15)
        text = "".join(np.random.choice(self.ascii_chars, strlen))
        ImageDraw.Draw(img).text(
            (x, y), text, fill=np.random.randint(0, 256), font=font
        )
        return img


class RandomRect:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        width, height = img.size
        x = np.random.randint(0, width / 2)
        y = np.random.randint(0, height / 2)
        rect_width = np.random.randint(width / 3, width / 2)
        rect_height = np.random.randint(height / 3, height / 2)
        ImageDraw.Draw(img).rectangle(
            [(x, y), (x + rect_width, y + rect_height)],
            fill=None,
            width=np.random.randint(1, 3),
            outline=np.random.randint(0, 256),
        )
        return img


class RandomErase:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        width, height = img.size
        r = np.random.randint(0.05 * width, 0.2 * width)
        x = np.random.randint(r, width - r)
        y = np.random.randint(r, height - r)
        ImageDraw.Draw(img).ellipse(
            [(x - r, y - r), (x + r, y + r)], outline=None, fill=(0,)
        )
        return img
