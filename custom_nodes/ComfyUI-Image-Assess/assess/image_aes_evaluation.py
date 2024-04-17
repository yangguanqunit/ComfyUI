'''
File Name: image_aes_evaluation
Create File Time: 2024/4/17 16:24
File Create By Author: Yang Guanqun
Email: yangguanqunit@outlook.com
'''
import folder_paths
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF


from .utils_model import get_available_aes_model, get_aes_path
from .utils_model import AESInfo
# from comfy.model_patcher import ModelPatcher
DIMMAX = 8192


def aes_eval_sac(images: np.ndarray, width, height, clip_model, aes_model, max_n, device="cuda"):
    assert len(images.shape) == 4, "dim of images must be 4."
    max_n = min(len(images), max_n)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    score_dict = {}
    for img in images:
        img = Image.fromarray(img)
        img = TF.resize(img, width, transforms.InterpolationMode.LANCZOS)
        img = TF.center_crop(img, [width, height])
        img_tensor = TF.to_tensor(img).to(device)
        img_tensor = normalize(img_tensor)
        clip_image_embed = F.normalize(
            clip_model.encode_image(img_tensor[None, ...]).float(),
            dim=-1
        )
        score = aes_model(clip_image_embed)
        score_dict[img] = score

    sorted_score = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))  # 按美学得分从高到底排序
    return list(sorted_score.keys())[:max_n]


class AESLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "aes_name": (get_available_aes_model(),),
            },
        }

    RETURN_TYPES = ("AES_MODEL",)
    # TBD
    CATEGORY = ""
    FUNCTION = "load_aes_model"

    def load_aes_model(self, aes_name: str):
        # check if motion lora with name exists


        return (prev_motion_lora,)


class ImageAESEvaluation:

    @classmethod
    def INPUT_TUPES(s):
        # 好像输入的图片要么是路径，要么是concat的array或者tensor
        return {"required": {
                "images": ("IMAGE",),
                "width:": ("INT", {"default": 512, "min": 0, "max": DIMMAX}),
                "height": ("INT", {"default": 512, "min": 0, "max": DIMMAX}),
                "clip_model": ("MODEL",),
                "aes_model": ("MODEL",),
                "max_n": ("INT", {"default": 5, "min": 1})
                },
            }

    CATEGORY = "AES_EVAL"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)

    FUNCTION = "aes_eval"


    def aes_eval(self, kwargs):
        return aes_eval_sac(kwargs)

    @classmethod
    def IS_CHANGED(s):
        pass

    @classmethod
    def VALIDATE_INPUTS(s):
        pass