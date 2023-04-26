from torchvision import datasets, transforms
from PIL import ImageFilter, ImageOps, Image, ImageDraw
import torchvision.transforms.functional as F
import random
import torch
from models.pos_embed import get_2d_sincos_pos_embed, get_2d_local_sincos_pos_embed

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class PermutePatch(object):
    """
    Apply Patch permutation to the PIL image.
    """
    def __init__(self, psz):
        self.psz = psz

    def __call__(self, img):
        imgs = []
        imgwidth, imgheight = img.size
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                box = (j, i, j+self.psz, i+self.psz)
                imgs.append(img.crop(box))
        random.shuffle(imgs)
        new_img = Image.new('RGB', (imgwidth, imgheight))
        k = 0
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                new_img.paste(imgs[k], (j, i))
                k += 1
        return new_img

class HideAndSeek(object):
    """
    Apply Patch permutation to the PIL image.
    """
    def __init__(self, ratio, psz):
        self.ratio = ratio
        self.psz = psz

    def __call__(self, img):
        imgwidth, imgheight = img.size
        numw, numh = imgwidth // self.psz, imgheight // self.psz
        mask_num = int(numw * numh * self.ratio)
        mask_patch = np.random.choice(np.arange(numw * numh), mask_num, replace=False)
        mask_w, mask_h = mask_patch % numh, mask_patch // numh
        # img.save('test1.png')
        draw = ImageDraw.Draw(img)
        for mw, mh in zip(mask_w, mask_h):
            draw.rectangle((mw * self.psz,
                            mh * self.psz,
                            (mw + 1) * self.psz,
                            (mh + 1) * self.psz), fill="black")
        # img.save('test2.png')
        return img

class RandomResizedCropCoord(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=Image.BICUBIC):
        self.size = size
        self.ratio = ratio
        self.scale = scale
        self.interpolation = interpolation

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, (self.size, self.size), self.interpolation)
        return (i, j, h, w), img

    def __call__(self, img):
        return self.forward(img)

class DataAugmentation(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number,
                embed_dim=384, g_grid_size=14, l_grid_size=6, local_crops_size=64):
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.embed_dim = embed_dim
        self.g_grid_size = g_grid_size
        self.l_grid_size = local_crops_size // 16
        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        # we do not add the flip to global_transform1
        self.global_rcr = RandomResizedCropCoord(224, scale=global_crops_scale, interpolation=Image.BICUBIC)
        self.local_rcr = RandomResizedCropCoord(local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC)
        self.global_transfo1 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            normalize,
        ])

    def calculate_sin_cos(self, lpos, gpos):
        kg = gpos[3] / self.g_grid_size
        w_bias = (lpos[1] - gpos[1]) / kg
        kl = lpos[3] / self.l_grid_size
        w_scale = kl / kg
        kg = gpos[2] / self.g_grid_size
        h_bias = (lpos[0] - gpos[0]) / kg
        kl = lpos[2] / self.l_grid_size
        h_scale = kl / kg
        return get_2d_local_sincos_pos_embed(self.embed_dim, self.l_grid_size, w_bias, w_scale, h_bias, h_scale)


    def forward(self, img):
        return self.__call__(img)

    def __call__(self, image):
        crops = []
        local_pos1, local_pos2 = [], []
        gpos1, gimg1 = self.global_rcr(image)
        crops.append(self.global_transfo1(gimg1))
        gpos2, gimg2 = self.global_rcr(image)
        crops.append(self.global_transfo1(gimg2))
        for _ in range(self.local_crops_number):
            lpos, limg = self.local_rcr(image)
            local_pos1.append(torch.FloatTensor(self.calculate_sin_cos(lpos, gpos1)))
            local_pos2.append(torch.FloatTensor(self.calculate_sin_cos(lpos, gpos2)))
            crops.append(self.local_transfo(limg))
        return crops, local_pos1, local_pos2

# if __name__ == '__main__':
#     aug = DataAugmentation(global_crops_scale=(0.5, 1), local_crops_scale=(0, 0.5), global_crops_number=2, local_crops_number=10)
#     img = Image.open("./test.jpeg").convert('RGB')
#     print(aug.forward(img)[1])