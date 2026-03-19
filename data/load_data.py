from torch.utils.data import Dataset
from imutils import paths
import numpy as np
import random
import cv2
import os

# Indian license plate character set (A-Z + 0-9 + CTC blank)
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '-'  # CTC blank token (must be last)
]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len,
                 augment=False, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        self.augment = augment
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        if Image is None:
            raise IOError(f"Cannot read image: {filename}")
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)

        if self.augment:
            Image = self.random_affine(Image)

        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, _ = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = []
        for c in imgname:
            if c not in CHARS_DICT:
                raise ValueError(
                    f"Character '{c}' in filename '{basename}' "
                    f"not in CHARS vocabulary"
                )
            label.append(CHARS_DICT[c])

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    def random_affine(self, img):
        """Random rotation, scaling, and translation (paper Sec. 3.2)."""
        h, w = img.shape[:2]
        angle = random.uniform(-5, 5)
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-3, 3)
        ty = random.uniform(-3, 3)
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        return cv2.warpAffine(
            img, M, (w, h), borderMode=cv2.BORDER_REPLICATE
        )
