from __future__ import annotations
import os
from typing import Optional, Tuple, Union, Dict, Sequence
from shutil import copyfile
import json
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import wget

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from flax.linen import partitioning as nn_partitioning
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from jax.sharding import PartitionSpec as PS
from scalax.sharding import MeshShardingHelper

from .utils import get_gradient_checkpoint_policy
from .splash import _tpu_splash_attention

OPENLLAMA_EASYLM_CACHE_DIR = os.path.abspath(str(Path.home() / '.cache/openllama/'))
OPENLLAMA_SPECIAL_TOKENS_MAP = {
    "bos_token": {
        "content": "<s>",
        "lstrip": False,
        "normalized": True,
        "rstrip": False,
        "single_word": False
    },
    "eos_token": {
        "content": "</s>",
        "lstrip": False,
        "normalized": True,
        "rstrip": False,
        "single_word": False
    },
    "unk_token": {
        "content": "<unk>",
        "lstrip": False,
        "normalized": True,
        "rstrip": False,
        "single_word": False
    },
}
OPENLLAMA_TOKENIZER_CONFIG = {
    "add_bos_token": True,
    "add_eos_token": False,
    "model_max_length": 2048,
    "pad_token": None,
    "sp_model_kwargs": {},
    "tokenizer_class": "LlamaTokenizer",
    "clean_up_tokenization_spaces": False,
    "bos_token": {
        "__type": "AddedToken",
        "content": "<s>",
        "lstrip": False,
        "normalized": True,
        "rstrip": False,
        "single_word": False
    },
    "eos_token": {
        "__type": "AddedToken",
        "content": "</s>",
        "lstrip": False,
        "normalized": True,
        "rstrip": False,
        "single_word": False
    },
    "unk_token": {
        "__type": "AddedToken",
        "content": "<unk>",
        "lstrip": False,
        "normalized": True,
        "rstrip": False,
        "single_word": False
    },
}

OPENLLAMA_EASYLM_URLS = {
    '3b_v1': {
        'model.msgpack': 'https://huggingface.co/openlm-research/open_llama_3b_easylm/resolve/main/open_llama_3b_easylm?download=true',
        'tokenizer.model': 'https://huggingface.co/openlm-research/open_llama_3b_easylm/resolve/main/tokenizer.model?download=true',
    },
    '7b_v1': {
        'model.msgpack': 'https://huggingface.co/openlm-research/open_llama_7b_easylm/resolve/main/open_llama_7b_easylm?download=true',
        'tokenizer.model': 'https://huggingface.co/openlm-research/open_llama_7b_easylm/resolve/main/tokenizer.model?download=true',
    },
    '13b_v1': {
        'model.msgpack': 'https://huggingface.co/openlm-research/open_llama_13b_easylm/resolve/main/open_llama_13b_easylm?download=true',
        'tokenizer.model': 'https://huggingface.co/openlm-research/open_llama_13b_easylm/resolve/main/tokenizer.model?download=true',
    },
    '3b_v2': {
        'model.msgpack': 'https://huggingface.co/openlm-research/open_llama_3b_v2_easylm/resolve/main/open_llama_3b_v2_easylm?download=true',
        'tokenizer.model': 'https://huggingface.co/openlm-research/open_llama_3b_v2_easylm/resolve/main/tokenizer.model?download=true',
    },
    '7b_v2': {
        'model.msgpack': 'https://huggingface.co/openlm-research/open_llama_7b_v2_easylm/resolve/main/open_llama_7b_v2_easylm?download=true',
        'tokenizer.model': 'https://huggingface.co/openlm-research/open_llama_7b_v2_easylm/resolve/main/tokenizer.model?download=true',
    },
}

def download_openllama_easylm(version: str, ignore_keys: Optional[Sequence[str]]=None) -> Dict[str]:
    assert version in OPENLLAMA_EASYLM_URLS, f'openllama {version} does not exist'
    # download files to cache
    os.makedirs(os.path.join(OPENLLAMA_EASYLM_CACHE_DIR, version), exist_ok=True)
    paths = dict()
    for filename, url in OPENLLAMA_EASYLM_URLS[version].items():
        if (ignore_keys is not None) and (filename in ignore_keys):
            continue
        download_path = os.path.join(OPENLLAMA_EASYLM_CACHE_DIR, version, filename)
        if not os.path.exists(download_path):
            wget.download(url, download_path)
        paths[filename] = download_path
    paths['params'] = paths.pop('model.msgpack')
    config_name = version.split('_')[0]
    config_path = os.path.join(OPENLLAMA_EASYLM_CACHE_DIR, version, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(LLAMA_STANDARD_CONFIGS[config_name], f)
    paths['config'] = config_path
    tokenizer_path = os.path.join(OPENLLAMA_EASYLM_CACHE_DIR, version, 'tokenizer')
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)
    copyfile(paths.pop('tokenizer.model'), os.path.join(tokenizer_path, 'tokenizer.model'))
    with open(os.path.join(tokenizer_path, 'tokenizer_config.json'), 'w') as f:
        json.dump(OPENLLAMA_TOKENIZER_CONFIG, f)
    with open(os.path.join(tokenizer_path, 'special_tokens_map.json'), 'w') as f:
        json.dump(OPENLLAMA_SPECIAL_TOKENS_MAP, f)
    paths['tokenizer'] = tokenizer_path
    return paths

LLAMA_STANDARD_CONFIGS = {
    '200m': {
        'vocab_size': 32000,
        'max_sequence_length': 4096,
        'hidden_size': 1024,
        'intermediate_size': 2752,  # ~2.7x the hidden size.
        'num_hidden_layers': 18,
        'num_attention_heads': 16,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '300m': {
        'vocab_size': 32000,
        'max_sequence_length': 4096,
        'hidden_size': 1024,
        'intermediate_size': 2752,  # ~2.7x the hidden size.
        'num_hidden_layers': 28,
        'num_attention_heads': 16,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '1b': {
        'vocab_size': 32000,
        'hidden_size': 2048,
        'intermediate_size': 5504,
        'num_hidden_layers': 22,
        'num_attention_heads': 16,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '3b': {
        'vocab_size': 32000,
        'hidden_size': 3200,
        'intermediate_size': 8640,
        'num_hidden_layers': 26,
        'num_attention_heads': 32,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '7b': {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '13b': {
        'vocab_size': 32000,
        'hidden_size': 5120,
        'intermediate_size': 13824,
        'num_hidden_layers': 40,
        'num_attention_heads': 40,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '30b': {
        'vocab_size': 32000,
        'hidden_size': 6656,
        'intermediate_size': 17920,
        'num_hidden_layers': 60,
        'num_attention_heads': 52,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '65b': {
        'vocab_size': 32000,
        'hidden_size': 8192,
        'intermediate_size': 22016,
        'num_hidden_layers': 80,
        'num_attention_heads': 64,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    'debug': { # A small model for debugging
        'vocab_size': 32000,
        'hidden_size': 128,
        'intermediate_size': 256,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'max_sequence_length': 2048,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '7b_v2': {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'max_sequence_length': 4096,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '13b_v2': {
        'vocab_size': 32000,
        'hidden_size': 5120,
        'intermediate_size': 13824,
        'num_hidden_layers': 40,
        'num_attention_heads': 40,
        'max_sequence_length': 4096,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '70b_v2': {
        'vocab_size': 32000,
        'hidden_size': 8192,
        'intermediate_size': 28672,
        'num_hidden_layers': 80,
        'num_attention_heads': 64,
        'max_sequence_length': 4096,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        "num_key_value_heads": 8,
    },
    '8b_v3': {
        'vocab_size': 128256,
        'hidden_size': 4096,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'num_key_value_heads': 8,
        'intermediate_size': 14336,
        'norm_eps': 1e-5,
        'rope_theta': 500000.0,
        'max_sequence_length': 8192,
        'initializer_range': 0.02,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    '70b_v3': {
        'vocab_size': 128256,
        'hidden_size': 8192,
        'num_hidden_layers': 64,
        'num_attention_heads': 80,
        'num_key_value_heads': 8,
        'intermediate_size': 28672,
        'norm_eps': 1e-5,
        'rope_theta': 500000.0,
        'max_sequence_length': 8192,
        'initializer_range': 0.02,
        'use_cache': True,
        'tie_word_embeddings': False,
    },
    'debug_v2': { # A small model for debugging
        'vocab_size': 32000,
        'hidden_size': 128,
        'intermediate_size': 256,
        'num_hidden_layers': 2,
        'num_attention_heads': 64,
        'max_sequence_length': 4096,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
        "num_key_value_heads": 8,
    },
    'debug_v3': { # A small model for debugging
        'vocab_size': 32000,
        'hidden_size': 1024,
        'intermediate_size': 2048,
        'num_hidden_layers': 2,
        'num_attention_heads': 8,
        'max_sequence_length': 4096,
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-6,
        'use_cache': True,
        'tie_word_embeddings': False,
        "num_key_value_heads": 4,
    },
}

class LLaMAConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~LLaMAModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LLaMAModel`] or [`~TFLLaMAModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_sequence_length (`int`, *optional*, defaults to 2048):
            Max sequence length for model (for RoPE computation)
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:
    ```python
    >>> from transformers import LLaMAModel, LLaMAConfig
    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LLaMAConfig()
    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LLaMAModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        max_sequence_length=2048,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        remat_block='',
        remat_attention='',
        remat_mlp='',
        emb_vocab_size=None,
        splash_attention=False,
        splash_block_size=None,
        rope_theta=10000.0,
        use_scaled_rope=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        if emb_vocab_size is None:
            self.emb_vocab_size = vocab_size
        else:
            self.emb_vocab_size = emb_vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_sequence_length = max_sequence_length
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.remat_block = remat_block
        self.remat_attention = remat_attention
        self.remat_mlp = remat_mlp
        self.splash_attention = splash_attention
        self.splash_block_size = splash_block_size
        self.rope_theta = rope_theta
        self.use_scaled_rope = use_scaled_rope
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    
    @staticmethod
    def get_partition_rules():
        """ Parition rules for GPTJ. Note that these rules are orderd, so that
            the beginning rules match first. It is important to use
            PartitionSpec() instead of None here because JAX does not treat
            None as a pytree leaf.
        """
        return (
            # embeddings
            ("transformer/wte/embedding", PS("mp", "fsdp")),
            # atention
            ("attention/(wq|wk|wv)/kernel", PS("fsdp", "mp")),
            ("attention/wo/kernel", PS("mp", "fsdp")),
            # mlp
            ("feed_forward/w1/kernel", PS("fsdp", "mp")),
            ("feed_forward/w2/kernel", PS("mp", "fsdp")),
            ("feed_forward/w3/kernel", PS("fsdp", "mp")),
            # layer norms
            ("attention_norm/kernel", PS(None)),
            ("ffn_norm/kernel", PS(None)),
            # output head
            ("transformer/ln_f/kernel", PS(None)),
            ("lm_head/kernel", PS("fsdp", "mp")),
            ('.*', PS(None)),
        )


remat = nn_partitioning.remat

logger = logging.get_logger(__name__)

class RMSNorm(nn.Module):
    dim: int
    eps: float=1e-6
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel', 
            nn.initializers.ones, 
            (self.dim,), 
            self.param_dtype, 
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        output = self._norm(x.astype(self.dtype)).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight

def apply_scaling(freqs: np.ndarray):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return np.asarray(new_freqs, dtype=freqs.dtype)

def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0, use_scaled_rope: bool=False, dtype: jnp.dtype=jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)  # type: ignore
    if use_scaled_rope:
        freqs = apply_scaling(freqs)
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)

def apply_rotary_emb(
    xq: jnp.ndarray, 
    xk: jnp.ndarray, 
    freqs_cis: jnp.ndarray, 
    dtype: jnp.dtype=jnp.float32, 
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))
    
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)

def repeat_kv(
    hidden_states: jnp.ndarray,
    n_rep: int,
) -> jnp.ndarray:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :]
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=3)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class FlaxLLaMAAttention(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.wq = nn.Dense(
            config.num_attention_heads*self.head_dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range), 
            precision=self.precision, 
        )
        self.wk = nn.Dense(
            config.num_key_value_heads*self.head_dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range), 
            precision=self.precision, 
        )
        self.wv = nn.Dense(
            config.num_key_value_heads*self.head_dim, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range), 
            precision=self.precision, 
        )
        self.wo = nn.Dense(
            config.hidden_size, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range), 
            precision=self.precision, 
        )

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool")

        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            # config.max_sequence_length * 2,
            config.max_sequence_length,
            theta=config.rope_theta,
            use_scaled_rope=config.use_scaled_rope,
            dtype=self.dtype,
        )
    
    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        og_attention_mask = attention_mask
        xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

        xq = MeshShardingHelper.with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp"))
        xk = MeshShardingHelper.with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp"))
        xv = MeshShardingHelper.with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp"))

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        # xq = MeshShardingHelper.with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp", None))
        # xk = MeshShardingHelper.with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp", None))
        # xv = MeshShardingHelper.with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp", None))

        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype)

        query_length, key_length = xq.shape[1], xk.shape[1]
        
        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")
        
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)
        
        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        xk = repeat_kv(xk, self.num_key_value_groups)
        xv = repeat_kv(xv, self.num_key_value_groups)

        if not self.config.splash_attention:
            # usual dot product attention
            attn_weights = dot_product_attention_weights(
                xq, 
                xk, 
                bias=attention_bias, 
                dropout_rng=dropout_rng, 
                dropout_rate=self.config.attn_pdrop, 
                deterministic=deterministic, 
                dtype=self.dtype, 
                precision=self.precision, 
            )
            attn_weights = MeshShardingHelper.with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), "mp", None, None))

            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision)
        else:
            attn_output = _tpu_splash_attention(
                xq,
                xk,
                xv,
                attention_mask=og_attention_mask.astype(jnp.int32),
                dropout=self.config.attn_pdrop,
                attention_dtype=self.dtype,
                block_size=self.config.splash_block_size,
            )
            attn_output = MeshShardingHelper.with_sharding_constraint(attn_output, PS(("dp", "fsdp"), None, "mp", None))

        attn_output = self._merge_heads(attn_output)
        attn_output = MeshShardingHelper.with_sharding_constraint(attn_output, PS(("dp", "fsdp"), None, "mp"))
        attn_output = self.wo(attn_output)
        attn_output = MeshShardingHelper.with_sharding_constraint(attn_output, PS(("dp", "fsdp"), None, "mp"))
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

class FlaxLLaMAMLP(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        config = self.config

        self.w1 = nn.Dense(
            config.intermediate_size, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range), 
            precision=self.precision, 
        )
        self.w2 = nn.Dense(
            config.hidden_size, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range), 
            precision=self.precision, 
        )
        self.w3 = nn.Dense(
            config.intermediate_size, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range), 
            precision=self.precision, 
        )
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x, deterministic=deterministic)
        return x

class FlaxLLaMABlock(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        attention_module = FlaxLLaMAAttention
        mlp_module = FlaxLLaMAMLP
        if self.config.remat_attention != '':
            attention_module = remat(
                attention_module, static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.remat_attention),
                prevent_cse=True,
            )
        if self.config.remat_mlp != '':
            mlp_module = remat(
                mlp_module, static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(self.config.remat_mlp),
                prevent_cse=True,
            )

        self.attention = attention_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.feed_forward = mlp_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.attention_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.ffn_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
    
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        attn_outputs = self.attention(
            self.attention_norm(hidden_states), 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            deterministic=deterministic, 
            init_cache=init_cache, 
            output_attentions=output_attentions, 
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        feed_forward_hidden_states = self.feed_forward(
            self.ffn_norm(hidden_states), 
            deterministic=deterministic, 
        )
        hidden_states = hidden_states + feed_forward_hidden_states

        return (hidden_states,) + attn_outputs[1:]

class FlaxLLaMAPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LLaMAConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: LLaMAConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    @add_start_docstrings_to_model_forward("")
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTJAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs

class FlaxLLaMABlockCollection(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        block = FlaxLLaMABlock
        if self.config.remat_block != '':
            block = remat(
                FlaxLLaMABlock, static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.remat_block)
            )
        self.blocks = [
            block(
                self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            ) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states, 
                attention_mask, 
                position_ids, 
                deterministic, 
                init_cache, 
                output_attentions, 
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxGPTJModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs

class FlaxLLaMAModule(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.wte = nn.Embed(
            self.config.emb_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxLLaMABlockCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.ln_f = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.wte(input_ids.astype("i4"))

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )

@add_start_docstrings("", "")
class FlaxLLaMAModel(FlaxLLaMAPreTrainedModel):
    module_class = FlaxLLaMAModule

# append_call_sample_docstring(
#     FlaxLLaMAModel,
#     _TOKENIZER_FOR_DOC,
#     _CHECKPOINT_FOR_DOC,
#     FlaxCausalLMOutput,
#     _CONFIG_FOR_DOC,
# )

class FlaxLLaMAForCausalLMModule(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.transformer = FlaxLLaMAModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype, 
            use_bias=False, 
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range), 
            precision=self.precision, 
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@add_start_docstrings("", "")
class FlaxLLaMAForCausalLM(FlaxLLaMAPreTrainedModel):
    module_class = FlaxLLaMAForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.ndarray] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since GPTJ uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs

# append_call_sample_docstring(
#     FlaxGPTJForCausalLM,
#     _TOKENIZER_FOR_DOC,
#     _CHECKPOINT_FOR_DOC,
#     FlaxCausalLMOutput,
#     _CONFIG_FOR_DOC,
# )
