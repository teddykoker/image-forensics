import numpy as np
import string
from PIL import ImageDraw, ImageFont

ascii_chars = np.array(
    list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
)


class RandomText:
    def __init__(self, xmax, ymax, lenmax):
        self.xmax = xmax
        self.ymax = ymax
        self.lenmax = lenmax
        self.font = ImageFont.truetype("fonts/FreeSerif.ttf", 50)

    def __call__(self, img):
        x = np.random.randint(0, self.xmax)
        y = np.random.randint(0, self.ymax // 2)
        strlen = np.random.randint(0, self.lenmax)
        text = "".join(np.random.choice(ascii_chars, strlen))
        ImageDraw.Draw(img).text((x, y), text, (0,), font=self.font)
        return img
