import torch
import os.path as osp

def guess_load_checkpoint(pth_model):
    """
    Load checkpoint from various formats:
    - Single file: .safetensors, .pt, .pth, .bin
    - Directory with model.safetensors (merged weights from training)
    - Directory with DeepSpeed sharded checkpoint (legacy format)
    """
    if osp.isfile(pth_model):
        if pth_model.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(pth_model, device='cpu')
        else:
            state_dict = torch.load(pth_model, map_location='cpu', weights_only=False)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
    elif osp.isdir(pth_model):
        # Priority 1: merged safetensors checkpoint (GPU-count agnostic)
        safetensors_path = osp.join(pth_model, 'model.safetensors')
        if osp.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path, device='cpu')
        else:
            # Priority 2: DeepSpeed sharded checkpoint (legacy)
            try:
                from core.zero_to_any_dtype import \
                    get_state_dict_from_zero_checkpoint
            except ImportError:
                raise ImportError(
                    'The provided PTH model appears to be a DeepSpeed checkpoint. '
                    'However, DeepSpeed library is not detected in current '
                    'environment. This suggests that DeepSpeed may not be '
                    'installed or is incorrectly configured. Please verify your '
                    'setup.')
            state_dict = get_state_dict_from_zero_checkpoint(
                osp.dirname(pth_model), osp.basename(pth_model))
    else:
        raise FileNotFoundError(f'Cannot find {pth_model}')
    return state_dict
