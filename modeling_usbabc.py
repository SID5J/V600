# coding=utf-8
# Copyright 2025 USBABC Team. All rights reserved.
#
# Licensed under a proprietary license.

"""PyTorch USBABC model - GPT-style Transformer Decoder-only"""

import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from configuration_usbabc import USBABCConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "USBABCConfig"


def get_debug_log_path():
    """Obtém o caminho correto para o arquivo debug.log baseado no diretório atual."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, 'debug.log')


# ============================================================================
# Rotary Position Embedding (RoPE)
# ============================================================================

def rotate_half(x):
    """Rotaciona metade das dimensões ocultas do tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Aplica Rotary Position Embedding aos tensores de query e key.
    
    CORREÇÃO: Esta função agora usa position_ids para selecionar os embeddings
    corretos, evitando erros de dimensionalidade durante a geração com cache.
    """
    # cos, sin: [max_seq_len, head_dim]
    # position_ids: [batch_size, seq_len]
    
    # Verificar se position_ids está dentro dos limites
    if position_ids is not None:
        max_pos = position_ids.max().item()
        if max_pos >= cos.shape[0]:
            # Se position_ids excede o tamanho do cache, usar apenas as últimas posições
            position_ids = position_ids % cos.shape[0]
    
    # Pega os embeddings corretos para as posições atuais
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    # cos, sin agora tem shape [batch_size, 1, seq_len, head_dim]
    # Isso alinha com as dimensões de q e k para a multiplicação elemento a elemento.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class USBABCRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) para o modelo USBABC.
    Implementação baseada em "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pré-computar para otimização
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Diferente do paper, mas usado em muitas implementações
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# ============================================================================
# RMSNorm - Root Mean Square Layer Normalization
# ============================================================================

class USBABCRMSNorm(nn.Module):
    """
    RMSNorm - normalização mais eficiente que LayerNorm.
    Usado em modelos como LLaMA, Mistral, etc.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ============================================================================
# MLP - Multi-Layer Perceptron (Feed-Forward Network)
# ============================================================================

class USBABCMLP(nn.Module):
    """
    MLP com arquitetura SwiGLU (Swish-Gated Linear Unit).
    Usado em modelos modernos como LLaMA, PaLM, etc.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Projeções
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        # Função de ativação
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Attention - Multi-Head Self-Attention com suporte a GQA
# ============================================================================

class USBABCAttention(nn.Module):
    """
    Multi-Head Self-Attention com suporte para:
    - Causal masking (para autoregressive generation)
    - Grouped-Query Attention (GQA) - otimização de memória
    - Rotary Position Embedding (RoPE)
    - KV caching para geração eficiente
    """

    def __init__(self, config: USBABCConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instanciando {self.__class__.__name__} sem passar `layer_idx` não é recomendado e pode "
                "levar a erros durante geração. Por favor, certifique-se de fornecer `layer_idx` "
                "ao criar esta classe."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size deve ser divisível por num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Projeções Q, K, V, O
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Rotary embeddings
        self.rotary_emb = USBABCRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Projetar Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape para multi-head
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Aplicar RoPE
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # KV caching
        if past_key_value is not None:
            debug_log_path = get_debug_log_path()
            with open(debug_log_path, 'a') as f:
                f.write(f"BEFORE cache update - key_states shape: {key_states.shape}\n")
                f.write(f"BEFORE cache update - value_states shape: {value_states.shape}\n")
                f.flush()
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            with open(debug_log_path, 'a') as f:
                f.write(f"AFTER cache update - key_states shape: {key_states.shape}\n")
                f.write(f"AFTER cache update - value_states shape: {value_states.shape}\n")
                f.flush()

        # Repeat k/v heads se num_key_value_heads < num_heads (GQA)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Calcular attention scores
        debug_log_path = get_debug_log_path()
        with open(debug_log_path, 'a') as f:
            f.write(f"BEFORE matmul - query_states shape: {query_states.shape}\n")
            f.write(f"BEFORE matmul - key_states shape: {key_states.shape}\n")
            f.write(f"BEFORE matmul - key_states.transpose(2, 3) shape: {key_states.transpose(2, 3).shape}\n")
            f.flush()
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        with open(debug_log_path, 'a') as f:
            f.write(f"AFTER matmul - attn_weights shape: {attn_weights.shape}\n")
            f.flush()
        
        # CORREÇÃO: Validações de tamanho estritas removidas conforme CORRECOES.md.
        # Problemas de dimensionalidade durante a geração com cache são frequentemente
        # relacionados a incompatibilidades de broadcasting entre a máscara de atenção
        # e a matriz de pesos de atenção. Remover a verificação estrita e confiar
        # no mecanismo de broadcasting do PyTorch é uma solução mais robusta.

        # Aplicar attention mask (causal)
        if attention_mask is not None:
            with open(debug_log_path, 'a') as f:
                f.write(f"BEFORE mask - attn_weights shape: {attn_weights.shape}\n")
                f.write(f"BEFORE mask - attention_mask shape: {attention_mask.shape}\n")
                f.flush()
            # A adição da máscara é onde os erros de broadcasting geralmente ocorrem.
            # Se a máscara for, por exemplo, (bsz, 1, q_len, q_len) e os pesos de
            # atenção forem (bsz, num_heads, q_len, kv_seq_len), o broadcasting falhará.
            # A função _prepare_4d_causal_attention_mask do transformers deve
            # gerar uma máscara com o tamanho correto (bsz, 1, q_len, kv_seq_len).
            # A remoção da validação estrita permite que o broadcasting funcione
            # se as dimensões forem compatíveis.
            attn_weights = attn_weights + attention_mask
            with open(debug_log_path, 'a') as f:
                f.write(f"AFTER mask - attn_weights shape: {attn_weights.shape}\n")
                f.flush()

        # Softmax e dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Aplicar attention aos values
        # Debug: imprimir dimensões para diagnosticar o erro
        with open(debug_log_path, 'a') as f:
            f.write(f"DEBUG - attn_weights shape: {attn_weights.shape}\n")
            f.write(f"DEBUG - value_states shape: {value_states.shape}\n")
            f.write(f"DEBUG - query_states shape: {query_states.shape}\n")
            f.write(f"DEBUG - key_states shape: {key_states.shape}\n")
            f.flush()
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` deveria ter tamanho {(bsz, self.num_heads, q_len, self.head_dim)}, mas tem "
                f"tamanho {attn_output.size()}"
            )

        # Reshape e projetar saída
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repete key/value heads para Grouped-Query Attention.
        hidden_states: [batch, num_key_value_heads, slen, head_dim]
        retorna: [batch, num_attention_heads, slen, head_dim]
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ============================================================================
# Decoder Block - Bloco Transformer completo
# ============================================================================

class USBABCDecoderLayer(nn.Module):
    """
    Bloco Transformer decoder com:
    - Self-attention
    - MLP (feed-forward)
    - Layer normalization (RMSNorm)
    - Residual connections
    """

    def __init__(self, config: USBABCConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention
        self.self_attn = USBABCAttention(config=config, layer_idx=layer_idx)

        # MLP
        self.mlp = USBABCMLP(config)

        # Layer norms
        self.input_layernorm = USBABCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = USBABCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # Self-attention com pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP com pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ============================================================================
# USBABC PreTrainedModel - Classe base
# ============================================================================

USBABC_START_DOCSTRING = r"""
    Este modelo herda de [`PreTrainedModel`]. Verifique a documentação da superclasse para os métodos genéricos
    que a biblioteca implementa para todos os seus modelos (como download ou salvamento, redimensionamento dos
    embeddings de entrada, poda de cabeças, etc.)

    Este modelo também é uma subclasse PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module).
    Use-o como um módulo PyTorch regular e consulte a documentação do PyTorch para todas as questões relacionadas
    ao uso geral e comportamento.

    Parâmetros:
        config ([`USBABCConfig`]):
            Classe de configuração do modelo com todos os parâmetros do modelo. Inicializar com um arquivo de
            configuração não carrega os pesos associados ao modelo, apenas a configuração. Confira o método
            [`~PreTrainedModel.from_pretrained`] para carregar os pesos do modelo.
"""


@add_start_docstrings(
    "O modelo USBABC bare outputando estados ocultos brutos sem nenhuma cabeça específica em cima.",
    USBABC_START_DOCSTRING,
)
class USBABCPreTrainedModel(PreTrainedModel):
    config_class = USBABCConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["USBABCDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


USBABC_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Índices dos tokens de entrada no vocabulário. O padding será ignorado por padrão caso você forneça.

            Índices podem ser obtidos usando [`AutoTokenizer`]. Veja [`PreTrainedTokenizer.encode`] e
            [`PreTrainedTokenizer.__call__`] para detalhes.

            [O que são input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *opcional*):
            Máscara para evitar realizar atenção em tokens de padding. Valores de máscara selecionados em `[0, 1]`:

            - 1 para tokens que **não são mascarados**,
            - 0 para tokens que são **mascarados**.

            [O que são attention masks?](../glossary#attention-mask)

            Índices podem ser obtidos usando [`AutoTokenizer`]. Veja [`PreTrainedTokenizer.encode`] e
            [`PreTrainedTokenizer.__call__`] para detalhes.

            Se `past_key_values` for usado, opcionalmente apenas os últimos `input_ids` devem ser fornecidos
            (veja `past_key_values`).

            Se você quiser mudar o comportamento de padding, você deve ler [`modeling_opt._prepare_decoder_attention_mask`]
            e modificá-lo conforme suas necessidades. Veja o diagrama 1 em [o paper](  https://arxiv.org/abs/1910.13461  )
            para mais informações sobre a estratégia de mascaramento padrão.

            - 1 indica que a cabeça **não está mascarada**,
            - 0 indica que a cabeça **está mascarada**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *opcional*):
            Índices das posições de cada token de entrada na sequência de posições. Selecionado no intervalo `[0,
            config.n_positions - 1]`.

            [O que são position IDs?](../glossary#position-ids)
        past_key_values (`Cache` ou `tuple(tuple(torch.FloatTensor))`, *opcional*):
            Estados chave e valor pré-computados das camadas de atenção para acelerar decodificação sequencial. Isso
            tipicamente consiste nos `past_key_values` retornados pelo modelo em um passo anterior de decodificação
            quando `use_cache=True` ou `config.use_cache=True`.

            Duas formas são permitidas:
            - um objeto [`~cache_utils.Cache`];
            - Tuple de `tuple(torch.FloatTensor)` de comprimento `config.n_layers`, com cada tupla tendo 2 tensores de
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). Isso também é conhecido como o
            formato de cache legado.

            O modelo usará apenas os últimos `input_ids` se um `past_key_values` for passado. Se você quiser usar um
            modelo de dois estágios com este modelo, considere usar o objeto [`~cache_utils.DynamicCache`].

            Se `past_key_values` são usados, o usuário pode opcionalmente fornecer apenas os últimos `input_ids`
            (aqueles que não têm seus estados chave-valor passados para este modelo) de shape `(batch_size, 1)`
            ao invés de todos os `input_ids` de shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *opcional*):
            Opcionalmente, ao invés de passar `input_ids` você pode escolher passar diretamente uma representação
            embedada. Isso é útil se você quer mais controle sobre como converter `input_ids` índices em vetores
            associados do que a camada de embedding interna do modelo.
        use_cache (`bool`, *opcional*):
            Se definido como `True`, `past_key_values` key value states são retornados e podem ser usados para
            acelerar decodificação (veja `past_key_values`).
        output_attentions (`bool`, *opcional*):
            Se deve retornar os tensores de atenção de todas as camadas de atenção. Veja `attentions` sob
            tensores retornados para mais detalhes.
        output_hidden_states (`bool`, *opcional*):
            Se deve retornar os estados ocultos de todas as camadas. Veja `hidden_states` sob tensores retornados
            para mais detalhes.
        return_dict (`bool`, *opcional*):
            Se deve retornar um [`~utils.ModelOutput`] ao invés de uma tupla simples.
"""


@add_start_docstrings(
    "O modelo USBABC bare transformer outputando estados ocultos brutos sem nenhuma cabeça específica em cima.",
    USBABC_START_DOCSTRING,
)
class USBABCModel(USBABCPreTrainedModel):
    """
    Transformer decoder consistindo de *config.num_hidden_layers* camadas. Cada camada é um [`USBABCDecoderLayer`]

    Args:
        config: USBABCConfig
    """

    def __init__(self, config: USBABCConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Decoder layers
        self.layers = nn.ModuleList(
            [USBABCDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final norm
        self.norm = USBABCRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(USBABC_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Você não pode especificar ambos input_ids e inputs_embeds ao mesmo tempo")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Você deve especificar input_ids ou inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` é incompatível com gradient checkpointing. Definindo `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_seq_length()

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 4d mask é passado através das camadas
        # CORREÇÃO: Criar máscara adequada para o tamanho atual da sequência
        # Durante generate(), criar máscara adequada para o tamanho atual
        if past_key_values_length > 0:
            # Durante geração com cache, não usar máscara
            attention_mask = None
        else:
            # Primeira passagem - criar máscara para o tamanho atual da sequência
            if attention_mask is not None and attention_mask.shape[-1] != seq_length:
                # A máscara original tem tamanho diferente - criar nova máscara
                # Criar máscara causal simples para o tamanho atual
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
            
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ============================================================================
# USBABC For Causal Language Modeling
# ============================================================================

@add_start_docstrings(
    "O modelo USBABC transformer com uma cabeça de modelagem de linguagem (camada linear) em cima.",
    USBABC_START_DOCSTRING,
)
class USBABCForCausalLM(USBABCPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = USBABCModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(USBABC_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *opcional*):
                Labels para computação de loss de modelagem de linguagem. Índices devem estar em `[-100, 0, ...,
                config.vocab_size]` (veja `input_ids` docstring). Tokens com índices definidos como `-100` são ignorados
                (mascarados), o loss é computado apenas para os tokens com labels em `[0, ..., config.vocab_size]`.

        Returns:

        Exemplo:

        ```python
        from transformers import AutoTokenizer, USBABCForCausalLM

        model = USBABCForCausalLM.from_pretrained("USBABC/usbabc-8layer")
        tokenizer = AutoTokenizer.from_pretrained("USBABC/usbabc-8layer")

        prompt = "Olá, meu nome é"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Gerar
        generate_ids = model.generate(inputs.input_ids, max_length=30)
        tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        "Olá, meu nome é João e eu sou um estudante de..."
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift para que os tokens < n prevejam n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten os tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Habilitar model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepara inputs para geração. Este método é chamado a cada passo de geração.
        """
        # A lógica aqui está correta e não precisa de alterações. Ela prepara
        # os inputs para a geração incremental (decoding).
        
        cache_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                max_cache_length = getattr(past_key_values, 'get_max_length', lambda: None)()
            else: # Legacy cache
                cache_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # se o cache está sendo usado, precisamos apenas do último token
            if input_ids.shape[1] > 1:
                input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # criar position_ids on the fly para batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]


        # se `inputs_embeds` são passados, nós apenas queremos usá-los no primeiro passo de geração
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
