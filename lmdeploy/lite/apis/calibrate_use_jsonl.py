# Copyright (c) OpenMMLab. All rights reserved.
from .llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

from pathlib import Path
from typing import Union

import json
import fire
import torch
from torch import nn
# from accelerate import (infer_auto_device_map, init_empty_weights,
#                         load_checkpoint_in_model)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lmdeploy.lite.quantization import CalibrationContext
from lmdeploy.lite.utils import collect_target_modules, get_calib_loaders, load_hf_from_pretrained

LAYER_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMDecoderLayer',
    'QWenLMHeadModel': 'QWenBlock',
    'BaiChuanForCausalLM': 'DecoderLayer',
    'LlamaForCausalLM': 'LlamaDecoderLayer',
}
NORM_TYPE_MAP = {
    'InternLMForCausalLM': 'InternLMRMSNorm',
    'QWenLMHeadModel': 'RMSNorm',
    'BaiChuanForCausalLM': 'RMSNorm',
    'LlamaForCausalLM': 'LlamaRMSNorm',
}

def _prepare_for_calibrate(model: nn.Module,
                           layer_type: Union[str, type],
                           head_name: str = 'lm_head',
                           device: str = 'cuda',
                           prefix: str = '') -> None:
    """Prepare the model for calibration by moving specific modules to CPU.

    This function goes through each child of a given model and checks whether
    it is an instance of a certain layer type or has the name equal to
    `head_name`.
    If yes, it moves the module to CPU, otherwise to the specified device
    (default is CUDA).

    If the child contains the target layer type in its sub-modules, the
    function performs the same operation recursively.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to prepare for calibration.
    layer_type : Union[str, Type]
        The type of the layer to be moved to CPU. Can be either a string of
        class name or the class type itself.
    head_name : str, optional
        The name of the module to be moved to CPU. Default is 'lm_head'.
    device : str, optional
        The device to which modules not matching the `layer_type` or
        `head_name` will be moved. Default is 'cuda'.
    prefix : str, optional
        The prefix used when printing the names of the moved modules.
        Default is ''.

    Raises
    ------
    TypeError
        If `layer_type` is neither a string nor a type.
    """

    for name, child in model.named_children():

        # Check if the child is an instance of the given layer type
        if isinstance(layer_type, str):
            is_layer = type(child).__name__ == layer_type
        elif isinstance(layer_type, type):
            is_layer = isinstance(child, layer_type)
        else:
            raise TypeError(
                'layer_type should be a string (class name) or a type')

        # Check if the child contains the target module type
        contain_layer = len(
            collect_target_modules(child, layer_type, [head_name]).keys()) > 0

        # Check if the child matches the head name
        is_head = name == head_name

        mod_name = f'{prefix}.{name}' if prefix else name

        # If the child is either an instance of the layer type or has the
        # head name, move it to CPU, otherwise move it to the specified device
        if is_layer or is_head:
            child.to('cpu')
            print(f'Move {mod_name} to CPU.')
        elif contain_layer:
            _prepare_for_calibrate(child, layer_type, head_name, device,
                                   mod_name)
        else:
            child.to(device)
            print(f'Move {mod_name} to GPU.')


def calibrate(model: str,
              calib_jsonl_path: str,
              calib_samples: int = 128,
              calib_seqlen: int = 2048,
              work_dir: str = './work_dir',
              device: str = 'cuda') -> None:
    """The main function for loading the model and performing calibration on a
    given dataset.

    Args:
        model (str): The model to be loaded.
        calib_dataset (str, optional): The calibration dataset name.
            Defaults to 'c4'.
        calib_samples (int, optional): The number of samples for calibration.
            Defaults to 128.
        calib_seqlen (int, optional): The sequence length for calibration.
            Defaults to 2048.
        work_dir (str): The working directory for outputs.
            Defaults to './work_dir'.
        device (str, optional): The device to be used for calculation.
            Defaults to 'cuda'.
    """

    # Load tokenizer and configuration
    tokenizer = AutoTokenizer.from_pretrained(model,
                                              use_fast=False,
                                              trust_remote_code=True)
    
    model = load_hf_from_pretrained(model,
                                    torch_dtype=torch.float16,
                                    trust_remote_code=True)
    
    model_type = type(model).__name__
    if model_type not in LAYER_TYPE_MAP or model_type not in NORM_TYPE_MAP:
        raise RuntimeError(
            f'Currently, quantification and calibration of {model_type} are '
            f'not supported. The supported model types are '
            f"{', '.join(LAYER_TYPE_MAP.keys())}.")

    if model_type == 'QWenLMHeadModel':
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise RuntimeError(
                'When using Qwen, you need to `pip install flash-attn` first, '
                'otherwise calibration and quantification will not work '
                'properly.')

    layer_type = LAYER_TYPE_MAP[type(model).__name__]
    norm_type = NORM_TYPE_MAP[type(model).__name__]

    _prepare_for_calibrate(model, layer_type, 'lm_head', device)

    # hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    # checkpoint = hf_config._name_or_path

    # with init_empty_weights():
    #     # Load model
    #     model = AutoModelForCausalLM.from_pretrained(model,
    #                                                  torch_dtype=torch.float16,
    #                                                  trust_remote_code=True)
    #     model.config.use_cache = False

    # layer_type = LAYER_TYPE_MAP[type(model).__name__]
    # norm_type = NORM_TYPE_MAP[type(model).__name__]

    # decoder_layers = collect_target_modules(model, layer_type)

    # # Infer device map
    # device_map = infer_auto_device_map(model,
    #                                    no_split_module_classes=[layer_type])
    # for name in device_map.keys():
    #     if name in decoder_layers or 'lm_head' in name:
    #         device_map[name] = 'cpu'
    #     else:
    #         device_map[name] = 0
    # load_checkpoint_in_model(model, checkpoint, device_map)

    print('Loading calibrate dataset ...')
    calib_loader = [[]]
    with open(calib_jsonl_path, "r", encoding="utf8") as f:
        for item in f:
            item = item.rstrip()
            if len(item) == 0:
                continue

            item = json.loads(item)['input_ids']
            calib_loader[-1].extend(item)
            if len(calib_loader[-1]) >= calib_seqlen:
                calib_loader[-1] = calib_loader[-1][:calib_seqlen]
                calib_loader.append([])
    
            if len(calib_loader) == calib_samples + 1:
                break

    if len(calib_loader[-1]) < calib_seqlen:
        calib_loader.pop(-1)

    # Initialize calibration context
    calib_ctx = CalibrationContext(model,
                                   tokenizer,
                                   layer_type=layer_type,
                                   norm_type=norm_type,
                                   device=device)

    with calib_ctx:
        all_data = torch.cat([
            torch.LongTensor([data])
            for data in calib_loader
        ]).to(device)
        print(all_data.shape)
        calib_ctx.calibrate(all_data)

    # Create work directory if not exists
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    calib_ctx.export(work_dir)


if __name__ == '__main__':
    fire.Fire(calibrate)
