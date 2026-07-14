import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_ti2v_5B import ti2v_5B

WAN_CONFIGS = {
    'ti2v-5B': ti2v_5B,
}
