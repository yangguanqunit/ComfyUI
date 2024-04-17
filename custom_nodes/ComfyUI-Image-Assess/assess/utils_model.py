'''
File Name: utils_model
Create File Time: 2024/4/17 23:06
File Create By Author: Yang Guanqun
Email: yangguanqun01@corp.netease.com
Corp: Fuxi Tech, Netease
'''
import comfy.utils
import folder_paths

from comfy.model_patcher import ModelPatcher
from .models import AestheticMeanPredictionLinearModel as aesModel


# 存放模型的上一层目录
class Folders:
    AES_MODELS = "aes_models"
    CLIP_MODELS = "clip_models"


def get_available_aes_model():
    return folder_paths.get_filename_list(Folders.AES_MODELS)


def get_aes_path(aes_name: str):
    return folder_paths.get_full_path(Folders.AES_MODELS, aes_name)


def get_available_clip_model():
    return folder_paths.get_filename_list(Folders.CLIP_MODELS)


def get_clip_path(clip_name: str):
    return folder_paths.get_full_path(Folders.CLIP_MODELS, clip_name)


class AESInfo:
    def __init__(self, name: str, hash: str = ""):
        self.name = name
        self.hash = hash

    def set_hash(self, hash: str):
        self.hash = hash

    def clone(self):
        return AESInfo(self.name, self.hash)


class CLIPInfo:
    def __init__(self, name: str, hash: str = ""):
        self.name = name
        self.hash = hash

    def set_hash(self, hash: str):
        self.hash = hash

    def clone(self):
        return CLIPInfo(self.name, self.hash)


def load_aes_model(model_name: str):
    model_path = get_aes_path(model_name)
    aes_state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
    aes_model = aesModel(512).load_state_dict(aes_state_dict)




