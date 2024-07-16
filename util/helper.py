import os
import time
import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def naming_experiment_with_time():
    localtime = time.asctime(time.localtime(time.time()))
    date_list = localtime.split(' ')
    date_list.pop(-2)
    return '-'.join(date_list)


def get_checkpoint_path(root_dir='./checkpoints/mamba-330m_line/'):
    sub_dir = os.listdir(root_dir)[-1]
    return os.path.join(root_dir, sub_dir, 'pytorch_model.bin')

