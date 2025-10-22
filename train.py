#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Treinamento OTIMIZADO - COM DETECÇÃO AUTOMÁTICA DE CAMPOS
Versão melhorada que detecta automaticamente a estrutura do JSON
"""

import os
import json
import logging
import subprocess
import shutil
import gc
import warnings

# Filtros de avisos
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*pad_token.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*token_type_ids.*')
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import zipfile
try:
    from flask_socketio import SocketIO
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False
# Importações condicionais para evitar erros
try:
    import yaml
except ImportError:
    yaml = None

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        TrainerCallback
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Registrar modelo USBABC
try:
    from modeling_usbabc import USBABCConfig, USBABCForCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM
    
    # Registrar no sistema transformers
    AutoConfig.register("usbabc", USBABCConfig)
    AutoModelForCausalLM.register(USBABCConfig, USBABCForCausalLM)
    print("✓ Modelo USBABC registrado no sistema de treinamento")
except ImportError as e:
    print(f"⚠️ Aviso: Modelo USBABC não disponível: {e}")

# Importações para suporte universal
try:
    from safetensors import safe_open
    from safetensors.torch import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import gguf
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Reduzir verbosidade de bibliotecas externas
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('peft').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class AITrainingSystem:
    """Sistema otimizado com detecção automática de estrutura JSON"""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.original_gguf_path = None
        self.temp_hf_dir = "temp_modelo_hf"
        self.adapter_dir = "temp_lora_adapter"
        
        Path("modelos").mkdir(exist_ok=True)
        Path("dados").mkdir(exist_ok=True)
        Path(self.adapter_dir).mkdir(exist_ok=True)
        
        logger.info("✓ Sistema inicializado (modo otimizado)")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configurações otimizadas"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Usando config padrão otimizado: {e}")
        
        return {
            'training': {
                'batch_size': 1,
                'epochs': 1,
                'learning_rate': 2e-4,
                'max_length': 384,
                'gradient_accumulation_steps': 8,
                'logging_steps': 5,
                'save_steps': 99999,
                'warmup_steps': 5,
                'output_model_path': 'modelos/modelo_treinado.gguf'
            },
            'lora': {
                'r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.05,
                'target_modules': ["q_proj", "v_proj"]
            }
        }
    
    def _detect_model_format(self, model_path: str) -> str:
        """Detecta formato do modelo"""
        path = Path(model_path)
        suffix = path.suffix.lower()
        
        if suffix == '.zip':
            return 'zip'
        elif suffix == '.gguf':
            return 'gguf'
        elif suffix == '.safetensors':
            return 'safetensors'
        elif suffix in ['.bin', '.pt', '.pth']:
            return 'pytorch'
        elif path.is_dir():
            # Verificar se é diretório HuggingFace
            if (path / 'config.json').exists():
                return 'huggingface'
        
        return 'unknown'
    
    def _is_usbabc_model(self, model_path: str) -> bool:
        """Verifica se é um modelo USBABC analisando arquivos"""
        try:
            # Verificar se é um diretório HuggingFace
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        arch = config.get('architectures', [])
                        return any('usbabc' in str(a).lower() for a in arch)
            
            # Verificar se é um arquivo SafeTensors ou PyTorch
            elif model_path.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                # Tentar carregar metadados
                if model_path.endswith('.safetensors') and HAS_SAFETENSORS:
                    from safetensors import safe_open
                    with safe_open(model_path, framework="pt", device="cuda") as f:
                        # Verificar se há chaves específicas do USBABC
                        keys = list(f.keys())
                        return any('usbabc' in key.lower() for key in keys[:10])
                
                elif model_path.endswith(('.bin', '.pt', '.pth')):
                    # Carregar apenas metadados do PyTorch
                    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
                    if isinstance(checkpoint, dict):
                        # Verificar chaves do state_dict
                        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
                        if isinstance(state_dict, dict):
                            keys = list(state_dict.keys())
                            return any('usbabc' in key.lower() for key in keys[:10])
            
            return False
        except Exception as e:
            logger.warning(f"Erro ao verificar modelo USBABC: {e}")
            return False
    
    def _detect_model_type(self, model_path: str) -> str:
        """Detecta tipo do modelo pelo nome ou metadados"""
        name = os.path.basename(model_path).lower()
        
        # PRIORIDADE 1: Detectar modelos USBABC
        if 'usbabc' in name or self._is_usbabc_model(model_path):
            logger.info("🎯 Modelo USBABC detectado!")
            return "USBABC_CUSTOM"
        
        # Tentar detectar por metadados primeiro
        try:
            format_type = self._detect_model_format(model_path)
            
            if format_type == 'gguf' and HAS_GGUF:
                reader = gguf.GGUFReader(model_path)
                # Tentar extrair informações do GGUF
                for key, value in reader.fields.items():
                    if 'name' in key.lower():
                        model_name = str(value).lower()
                        if 'usbabc' in model_name:
                            return "USBABC_CUSTOM"
                        elif 'qwen' in model_name:
                            return "Qwen/Qwen2.5-1.5B-Instruct"
                        elif 'llama' in model_name:
                            return "gemma-portuguese-luana-2b.Q2_K.gguf"
                        elif 'mistral' in model_name:
                            return "mistralai/Mistral-7B-Instruct-v0.3"
            
            elif format_type == 'huggingface':
                config_path = Path(model_path) / 'config.json'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        arch = config.get('architectures', [''])[0].lower()
                        if 'usbabc' in arch:
                            return "USBABC_CUSTOM"
                        elif 'qwen' in arch:
                            return "Qwen/Qwen2.5-1.5B-Instruct"
                        elif 'llama' in arch:
                            return "gemma-portuguese-luana-2b.Q2_K.gguf"
                        elif 'mistral' in arch:
                            return "mistralai/Mistral-7B-Instruct-v0.3"
        
        except Exception as e:
            logger.warning(f"Erro ao detectar metadados: {e}")
        
        # Fallback para detecção por nome
        if "usbabc" in name:
            return "USBABC_CUSTOM"
        elif "qwen" in name:
            return "Qwen/Qwen2.5-1.5B-Instruct"
        elif "llama-3" in name or "llama3" in name:
            return "meta-llama/Llama-3.2-1B-Instruct"
        elif "llama" in name:
            return "gemma-portuguese-luana-2b.Q2_K.gguf"
        elif "mistral" in name:
            return "mistralai/Mistral-7B-Instruct-v0.3"
        elif "phi" in name:
            return "microsoft/phi-2"
        else:
            return "gemma-portuguese-luana-2b.Q2_K.gguf"
    
    def _convert_gguf_to_hf(self, gguf_path: str) -> str:
        """Converte GGUF para HF de forma leve com verificação de integridade"""
        logger.info("=" * 60)
        logger.info("PREPARANDO MODELO PARA TREINAMENTO")
        logger.info("=" * 60)
        
        self.original_gguf_path = os.path.abspath(gguf_path)
        base_model = self._detect_model_type(gguf_path)
        
        logger.info(f"🔍 Detectado: {base_model}")
        logger.info(f"📦 GGUF original: {gguf_path}")
        
        # FORÇAR USO APENAS DE MODELOS USBABC LOCAIS
        logger.info("🎯 Forçando uso de modelo USBABC local...")
        self._load_usbabc_model(gguf_path)
        return "USBABC_LOCAL"
        
        # Se todos os fallbacks falharam
        raise Exception("❌ Todos os modelos fallback falharam. Verifique sua conexão com a internet.")
    
    def _verify_cached_model(self, model_name: str, cache_dir: str) -> bool:
        """Verifica se o modelo em cache está íntegro"""
        try:
            import glob
            model_cache_pattern = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
            cache_dirs = glob.glob(model_cache_pattern + "*")
            
            if not cache_dirs:
                return False
            
            # Verificar se há arquivos SafeTensors no cache
            for cache_path in cache_dirs:
                safetensors_files = glob.glob(os.path.join(cache_path, "**", "*.safetensors"), recursive=True)
                for st_file in safetensors_files:
                    try:
                        from safetensors import safe_open
                        with safe_open(st_file, framework="pt") as f:
                            if len(f.keys()) == 0:
                                return False
                    except Exception:
                        return False
            
            return True
        except Exception:
            return False
    
    def _clear_corrupted_cache(self, model_name: str):
        """Remove cache corrompido do modelo"""
        try:
            import shutil
            import glob
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_pattern = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
            cache_dirs = glob.glob(model_cache_pattern + "*")
            
            for cache_path in cache_dirs:
                if os.path.exists(cache_path):
                    shutil.rmtree(cache_path)
                    logger.info(f"🗑️ Cache corrompido removido: {cache_path}")
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível limpar cache: {e}")
    
    def load_model(self, model_path: str = None):
        """Carrega modelo universal com LoRA otimizado"""
        try:
            if not model_path:
                raise ValueError("Caminho do modelo é obrigatório")
            
            format_type = self._detect_model_format(model_path)
            logger.info(f"📦 Formato detectado: {format_type}")
            
            if format_type == 'zip':
                logger.info(f"📦 ZIP detectado: {model_path}")
                self._load_zip_model(model_path)
                
            elif format_type == 'gguf':
                logger.info(f"⚠️ GGUF detectado: {model_path}")
                self._convert_gguf_to_hf(model_path)
                
            elif format_type == 'safetensors':
                logger.info(f"🔒 SafeTensors detectado: {model_path}")
                self._load_safetensors_model(model_path)
                
            elif format_type == 'pytorch':
                logger.info(f"🔥 PyTorch detectado: {model_path}")
                self._load_pytorch_model(model_path)
                
            elif format_type == 'huggingface':
                logger.info(f"🤗 HuggingFace detectado: {model_path}")
                self._load_huggingface_model(model_path)
                
            else:
                # Tentar como HuggingFace por padrão
                logger.warning(f"⚠️ Formato desconhecido, tentando como HuggingFace: {model_path}")
                self._load_huggingface_model(model_path)
            
            # Detectar módulos alvo automaticamente baseado na arquitetura
            target_modules = self._get_target_modules()
            
            lora_cfg = self.config.get('lora', {})
            lora_config = LoraConfig(
                r=lora_cfg.get('r', 8),
                lora_alpha=lora_cfg.get('lora_alpha', 16),
                target_modules=target_modules,
                lora_dropout=lora_cfg.get('lora_dropout', 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("✓ Modelo pronto (LoRA aplicado)")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar: {e}")
            raise
    
    def _get_target_modules(self):
        """Detecta automaticamente os módulos alvo para LoRA baseado na arquitetura do modelo"""
        try:
            # Mapear nomes de módulos comuns por arquitetura
            module_patterns = {
                'gpt2': ['c_attn', 'c_proj'],
                'llama': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                'mistral': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                'qwen': ['c_attn', 'c_proj'],
                'phi': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                'gemma': ['q_proj', 'v_proj', 'k_proj', 'o_proj']
            }
            
            # Obter todos os nomes de módulos do modelo
            all_modules = []
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                    all_modules.append(name.split('.')[-1])
            
            # Tentar encontrar padrões conhecidos
            for arch, patterns in module_patterns.items():
                found_modules = [m for m in patterns if m in all_modules]
                if found_modules:
                    logger.info(f"🎯 Detectada arquitetura {arch}, usando módulos: {found_modules}")
                    return found_modules
            
            # Fallback: procurar por padrões comuns
            common_patterns = ['attn', 'attention', 'proj', 'linear', 'dense']
            fallback_modules = []
            
            for pattern in common_patterns:
                matches = [m for m in all_modules if pattern in m.lower()]
                fallback_modules.extend(matches[:2])  # Limitar a 2 por padrão
            
            if fallback_modules:
                logger.info(f"🎯 Usando módulos detectados automaticamente: {fallback_modules[:4]}")
                return fallback_modules[:4]  # Máximo 4 módulos
            
            # Último recurso: usar módulos lineares genéricos
            linear_modules = [m for m in all_modules if 'linear' in m.lower() or 'dense' in m.lower()]
            if linear_modules:
                logger.info(f"🎯 Usando módulos lineares: {linear_modules[:2]}")
                return linear_modules[:2]
            
            # Se nada funcionar, usar padrão básico
            logger.warning("⚠️ Não foi possível detectar módulos automaticamente, usando padrão básico")
            return ["c_attn"]  # Padrão mais comum
            
        except Exception as e:
            logger.warning(f"⚠️ Erro na detecção automática de módulos: {e}")
            return ["c_attn"]
    
    def _load_zip_model(self, model_path: str):
        """Carrega modelo de arquivo ZIP"""
        try:
            import tempfile
            
            # Criar diretório temporário
            temp_dir = tempfile.mkdtemp(prefix="zip_model_")
            
            # Extrair ZIP
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Procurar por arquivos de modelo
            extracted_files = list(Path(temp_dir).rglob("*"))
            
            # Procurar por GGUF primeiro
            gguf_files = [f for f in extracted_files if f.suffix.lower() == '.gguf']
            if gguf_files:
                logger.info(f"📄 Encontrado GGUF no ZIP: {gguf_files[0]}")
                logger.warning("⚠️ GGUF detectado mas conversão desabilitada - usando modelo base")
                self._load_huggingface_model("gemma-portuguese-luana-2b.Q2_K.gguf")
                return
            
            # Procurar por SafeTensors
            safetensors_files = [f for f in extracted_files if f.suffix.lower() == '.safetensors']
            if safetensors_files:
                logger.info(f"🔒 Encontrado SafeTensors no ZIP: {safetensors_files[0]}")
                self._load_safetensors_model(str(safetensors_files[0]))
                return
            
            # Procurar por PyTorch
            pytorch_files = [f for f in extracted_files if f.suffix.lower() in ['.bin', '.pt', '.pth']]
            if pytorch_files:
                logger.info(f"🔥 Encontrado PyTorch no ZIP: {pytorch_files[0]}")
                self._load_pytorch_model(str(pytorch_files[0]))
                return
            
            # Procurar por diretório HuggingFace
            config_files = [f for f in extracted_files if f.name == 'config.json']
            if config_files:
                hf_dir = config_files[0].parent
                logger.info(f"🤗 Encontrado HuggingFace no ZIP: {hf_dir}")
                self._load_huggingface_model(str(hf_dir))
                return
            
            raise Exception("Nenhum formato de modelo reconhecido encontrado no ZIP")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar ZIP: {e}")
            raise
        finally:
            # Limpar diretório temporário
            if 'temp_dir' in locals() and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _load_huggingface_model(self, model_path: str):
        """Carrega modelo HuggingFace"""
        if not HAS_TORCH:
            raise ImportError("PyTorch não disponível")
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                use_fast=False
            )
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar tokenizer de {model_path}: {e}")
            logger.info("🔄 Usando tokenizer fallback (TinyLlama)...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gemma-portuguese-luana-2b.Q2_K.gguf",
                trust_remote_code=True,
                use_fast=False
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Sincronizar com model config para evitar avisos
            if hasattr(self.model, 'config'):
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    def _load_safetensors_model(self, model_path: str):
        """Carrega modelo SafeTensors"""
        # Lógica de carregamento de SafeTensors (mantida)
        base_model_name = self._detect_model_type(model_path)
        logger.info(f"Base model: {base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True,
            use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Sincronizar com model config para evitar avisos
            if hasattr(self.model, 'config'):
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if HAS_SAFETENSORS:
            from safetensors import safe_open
            state_dict = {}
            with safe_open(model_path, framework="pt", device="cuda") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"⚠️ Chaves faltando: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"⚠️ Chaves inesperadas: {len(unexpected_keys)}")
        
        logger.info("✓ SafeTensors carregado")
    
    def _load_pytorch_model(self, model_path: str):
        """Carrega modelo PyTorch"""
        # Lógica de carregamento de PyTorch (mantida)
        base_model_name = self._detect_model_type(model_path)
        logger.info(f"Base model: {base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True,
            use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Sincronizar com model config para evitar avisos
            if hasattr(self.model, 'config'):
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"⚠️ Chaves faltando: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"⚠️ Chaves inesperadas: {len(unexpected_keys)}")
        
        logger.info("✓ PyTorch carregado")
    
    def _load_usbabc_model(self, model_path: str):
        """Carrega modelo USBABC usando a classe customizada"""
        logger.info("🎯 Carregando modelo USBABC customizado...")
        
        try:
            # Importar a classe USBABC
            from modeling_usbabc import USBABCForCausalLM, USBABCConfig
            from transformers import AutoTokenizer
            
            # Verificar se é um diretório HuggingFace
            if os.path.isdir(model_path):
                logger.info("📁 Carregando modelo USBABC de diretório HuggingFace...")
                
                # Carregar tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=False
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Carregar modelo
                self.model = USBABCForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
            elif model_path.endswith('.gguf'):
                # Arquivo GGUF - usar llama-cpp-python para carregamento
                logger.info("📄 Carregando modelo USBABC GGUF...")
                
                # Criar tokenizer local simples
                logger.info("🔧 Criando tokenizer local...")
                self.tokenizer = self._create_local_tokenizer()
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Para GGUF, vamos usar um modelo base e depois substituir os pesos
                # Criar configuração USBABC padrão
                config = USBABCConfig(
                    vocab_size=12000,
                    hidden_size=256,
                    intermediate_size=1376,
                    num_hidden_layers=8,
                    num_attention_heads=8,
                    num_key_value_heads=8,
                    max_position_embeddings=2048,
                    rms_norm_eps=1e-5,
                    rope_theta=10000.0,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                # Criar modelo vazio
                self.model = USBABCForCausalLM(config)
                
                # Para GGUF, vamos inicializar com pesos aleatórios por enquanto
                # TODO: Implementar conversão GGUF->PyTorch para USBABC
                logger.warning("⚠️ GGUF USBABC: Usando inicialização aleatória (TODO: implementar conversão)")
                
                # Mover para dispositivo apropriado
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(device)
                
            else:
                # Arquivo único (SafeTensors ou PyTorch)
                logger.info("📄 Carregando modelo USBABC de arquivo único...")
                
                # Criar tokenizer local simples
                logger.info("🔧 Criando tokenizer local...")
                self.tokenizer = self._create_local_tokenizer()
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Criar configuração USBABC padrão
                config = USBABCConfig(
                    vocab_size=12000,
                    hidden_size=256,
                    intermediate_size=1376,
                    num_hidden_layers=8,
                    num_attention_heads=8,
                    num_key_value_heads=8,
                    max_position_embeddings=2048,
                    rms_norm_eps=1e-5,
                    rope_theta=10000.0,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                # Criar modelo vazio
                self.model = USBABCForCausalLM(config)
                
                # Carregar pesos
                if model_path.endswith('.safetensors') and HAS_SAFETENSORS:
                    logger.info("📥 Carregando pesos do SafeTensors...")
                    from safetensors import safe_open
                    with safe_open(model_path, framework="pt", device="cuda") as f:
                        state_dict = {}
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                else:
                    logger.info("📥 Carregando pesos do PyTorch...")
                    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
                    if isinstance(checkpoint, dict):
                        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
                    else:
                        state_dict = checkpoint
                
                # Aplicar pesos
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logger.warning(f"⚠️ Chaves faltando: {len(missing_keys)}")
                if unexpected_keys:
                    logger.warning(f"⚠️ Chaves inesperadas: {len(unexpected_keys)}")
                
                # Mover para dispositivo apropriado
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(device)
            
            logger.info("✅ Modelo USBABC carregado com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo USBABC: {e}")
            raise
    
    def _create_local_tokenizer(self):
        """Cria um tokenizer local simples sem downloads"""
        try:
            from transformers import PreTrainedTokenizerFast
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
            
            # Criar tokenizer BPE simples
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.decoder = decoders.BPEDecoder()
            
            # Carregar dados de treinamento para criar vocabulário português
            vocab_text = self._load_portuguese_vocab()
            
            trainer = trainers.BpeTrainer(
                vocab_size=12000,
                min_frequency=1,
                special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
            )
            
            tokenizer.train_from_iterator(vocab_text, trainer)
            
            # Converter para PreTrainedTokenizerFast
            fast_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                pad_token="<pad>",
                unk_token="<unk>",
                bos_token="<s>",
                eos_token="</s>"
            )
            
            logger.info("✅ Tokenizer local criado com sucesso!")
            return fast_tokenizer
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar tokenizer local: {e}")
            # Fallback: criar tokenizer muito simples
            return self._create_simple_tokenizer()
    
    def _create_simple_tokenizer(self):
        """Cria um tokenizer português simples como fallback"""
        class PortugueseTokenizer:
            def __init__(self):
                # Vocabulário português básico com caracteres especiais
                self.vocab = {
                    "<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3,
                    " ": 4, "\n": 5, "\t": 6, "!": 7, "?": 8, ".": 9, ",": 10,
                    "a": 11, "e": 12, "i": 13, "o": 14, "u": 15,
                    "á": 16, "é": 17, "í": 18, "ó": 19, "ú": 20,
                    "à": 21, "è": 22, "ì": 23, "ò": 24, "ù": 25,
                    "â": 26, "ê": 27, "î": 28, "ô": 29, "û": 30,
                    "ã": 31, "õ": 32, "ç": 33,
                    "b": 34, "c": 35, "d": 36, "f": 37, "g": 38,
                    "h": 39, "j": 40, "k": 41, "l": 42, "m": 43,
                    "n": 44, "p": 45, "q": 46, "r": 47, "s": 48,
                    "t": 49, "v": 50, "w": 51, "x": 52, "y": 53, "z": 54,
                    "0": 55, "1": 56, "2": 57, "3": 58, "4": 59,
                    "5": 60, "6": 61, "7": 62, "8": 63, "9": 64
                }
                
                # Criar vocabulário reverso
                self.id_to_token = {v: k for k, v in self.vocab.items()}
                
                # Tokens especiais
                self.pad_token = "<pad>"
                self.eos_token = "</s>"
                self.bos_token = "<s>"
                self.unk_token = "<unk>"
                self.pad_token_id = 0
                self.eos_token_id = 3
                self.bos_token_id = 2
                self.unk_token_id = 1
                
            def encode(self, text, **kwargs):
                """Codifica texto em IDs de tokens"""
                if isinstance(text, str):
                    text = text.lower()  # Normalizar para minúsculas
                    tokens = []
                    for char in text[:512]:  # Limitar tamanho máximo
                        token_id = self.vocab.get(char, self.unk_token_id)
                        tokens.append(token_id)
                    return tokens
                return []
                
            def decode(self, tokens, **kwargs):
                """Decodifica IDs de tokens em texto"""
                if isinstance(tokens, list):
                    text = ""
                    for token_id in tokens:
                        if isinstance(token_id, (int, float)):
                            token_id = int(token_id)
                            char = self.id_to_token.get(token_id, self.unk_token)
                            if char not in ["<pad>", "<s>", "</s>", "<unk>"]:
                                text += char
                    return text
                return ""
                
            def __call__(self, text, **kwargs):
                """Interface compatível com transformers"""
                if isinstance(text, str):
                    input_ids = self.encode(text)
                    return {"input_ids": [input_ids]}
                elif isinstance(text, list):
                    batch_ids = []
                    for t in text:
                        batch_ids.append(self.encode(t))
                    return {"input_ids": batch_ids}
                return {"input_ids": [[]]}
                
            def batch_decode(self, sequences, **kwargs):
                """Decodifica batch de sequências"""
                results = []
                for seq in sequences:
                    results.append(self.decode(seq, **kwargs))
                return results
        
        logger.info("✅ Tokenizer português simples criado como fallback!")
        return PortugueseTokenizer()
    
    def _load_portuguese_vocab(self):
        """Carrega vocabulário português dos dados de treinamento"""
        try:
            # Tentar carregar do arquivo de dados
            data_path = Path('dados/train.json')
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extrair textos para vocabulário
                vocab_texts = []
                for item in data[:1000]:  # Usar primeiros 1000 exemplos
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, str) and len(value) > 10:
                                vocab_texts.append(value)
                    elif isinstance(item, str):
                        vocab_texts.append(item)
                
                if vocab_texts:
                    logger.info(f"✅ Carregados {len(vocab_texts)} textos para vocabulário português")
                    return vocab_texts
            
            # Fallback: vocabulário português básico
            return self._get_basic_portuguese_vocab()
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar vocabulário: {e}")
            return self._get_basic_portuguese_vocab()
    
    def _get_basic_portuguese_vocab(self):
        """Vocabulário português básico para fallback"""
        return [
            "olá mundo como você está hoje",
            "muito bem obrigado e você como está",
            "inteligência artificial aprendizado de máquina",
            "processamento de linguagem natural português",
            "dados de treinamento modelo neural transformers",
            "atenção camadas parâmetros otimização gradiente",
            "backpropagation redes neurais deep learning",
            "python programação desenvolvimento software",
            "brasil brasileiro português linguagem natural",
            "tecnologia inovação pesquisa desenvolvimento",
            "universidade educação ensino aprendizagem",
            "ciência computação algoritmos estruturas dados",
            "sistema operacional linux windows aplicações",
            "internet web desenvolvimento frontend backend",
            "banco dados sql nosql mongodb postgresql",
            "framework biblioteca api rest json xml",
            "segurança criptografia autenticação autorização",
            "nuvem cloud computing aws azure google",
            "devops docker kubernetes containers microserviços",
            "análise dados estatística matemática probabilidade",
            "visualização gráficos dashboards relatórios métricas",
            "negócios empresa mercado cliente produto serviço",
            "gestão projeto metodologia agile scrum kanban",
            "qualidade teste unitário integração automação",
            "performance otimização escalabilidade disponibilidade",
            "monitoramento logs métricas alertas observabilidade",
            "cultura sociedade história geografia política",
            "economia finanças investimento mercado ações",
            "saúde medicina tratamento diagnóstico prevenção",
            "esporte futebol basquete tênis natação corrida",
            "arte música literatura cinema teatro dança",
            "culinária receita ingrediente preparo cozinha",
            "viagem turismo destino hotel passagem avião",
            "família amigos relacionamento amor carinho",
            "trabalho carreira profissão emprego salário",
            "casa moradia apartamento construção decoração",
            "transporte carro ônibus metrô bicicleta caminhada",
            "tempo clima temperatura chuva sol vento",
            "natureza meio ambiente sustentabilidade ecologia",
            "animal cachorro gato pássaro peixe cavalo",
            "planta árvore flor jardim agricultura fazenda",
            "cidade estado país continente mundo planeta",
            "governo política democracia eleição voto cidadão",
            "direito lei justiça tribunal advogado juiz",
            "educação escola universidade professor aluno",
            "comunicação telefone email mensagem conversa",
            "compra venda produto preço desconto promoção",
            "problema solução resposta pergunta dúvida",
            "início meio fim começo término conclusão"
        ]
    
    def _analyze_json_structure(self, data: List[dict]) -> Dict:
        """Analisa estrutura do JSON para detectar campos"""
        logger.info("=" * 60)
        logger.info("🔍 ANALISANDO ESTRUTURA DO JSON")
        logger.info("=" * 60)
        
        all_keys = set()
        for item in data[:10]:  # Analisa primeiros 10 exemplos
            all_keys.update(item.keys())
        
        logger.info(f"📋 Campos encontrados: {sorted(all_keys)}")
        
        # Detectar campos de entrada/saída
        input_candidates = []
        output_candidates = []
        
        for key in all_keys:
            key_lower = key.lower()
            
            # Possíveis campos de ENTRADA
            if any(word in key_lower for word in ['input', 'pergunta', 'question', 'prompt', 'texto', 'text', 'consulta', 'query', 'user', 'usuario']):
                input_candidates.append(key)
            
            # Possíveis campos de SAÍDA
            if any(word in key_lower for word in ['output', 'resposta', 'answer', 'completion', 'response', 'resultado', 'result', 'assistant']):
                output_candidates.append(key)
        
        logger.info(f"📥 Campos de ENTRADA detectados: {input_candidates}")
        logger.info(f"📤 Campos de SAÍDA detectados: {output_candidates}")
        
        # Mostrar exemplo
        if data:
            logger.info("=" * 60)
            logger.info("📄 EXEMPLO DO PRIMEIRO ITEM:")
            logger.info("=" * 60)
            example = data[0]
            for key, value in example.items():
                value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                logger.info(f"  • {key}: {value_str}")
            logger.info("=" * 60)
        
        return {
            'all_keys': all_keys,
            'input_candidates': input_candidates,
            'output_candidates': output_candidates
        }
    
    def _format_item(self, item: dict) -> str:
        """Formata item - VERSÃO MELHORADA com mais opções"""
        
        # OPÇÃO 1: Formatos padrão
        if 'instruction' in item and 'output' in item:
            inp = item.get('input', '')
            full_input = f"{item['instruction']}\n{inp}".strip()
            return f"<|user|>\n{full_input}</s>\n<|assistant|>\n{item['output']}</s>"
        
        if 'prompt' in item and 'completion' in item:
            return f"<|user|>\n{item['prompt']}</s>\n<|assistant|>\n{item['completion']}</s>"
        
        if 'input' in item and 'output' in item:
            return f"<|user|>\n{item['input']}</s>\n<|assistant|>\n{item['output']}</s>"
        
        if 'question' in item and 'answer' in item:
            return f"<|user|>\n{item['question']}</s>\n<|assistant|>\n{item['answer']}</s>"
        
        # OPÇÃO 2: Formato em português
        if 'pergunta' in item and 'resposta' in item:
            return f"<|user|>\n{item['pergunta']}</s>\n<|assistant|>\n{item['resposta']}</s>"
        
        if 'consulta' in item and 'resultado' in item:
            return f"<|user|>\n{item['consulta']}</s>\n<|assistant|>\n{item['resultado']}</s>"
        
        if 'texto' in item and 'resposta' in item:
            return f"<|user|>\n{item['texto']}</s>\n<|assistant|>\n{item['resposta']}</s>"
        
        # OPÇÃO 3: Formato user/assistant
        if 'user' in item and 'assistant' in item:
            return f"<|user|>\n{item['user']}</s>\n<|assistant|>\n{item['assistant']}</s>"
        
        # OPÇÃO 4: Campos genéricos (primeiro que tiver conteúdo razoável)
        if 'text' in item and isinstance(item['text'], str) and len(item['text']) > 20:
            return item['text']
        
        # OPÇÃO 5: Tentar detectar automaticamente
        keys = list(item.keys())
        if len(keys) >= 2:
            # Pegar primeiros dois campos com conteúdo significativo
            potential_input = None
            potential_output = None
            
            for key in keys:
                value = str(item[key])
                if len(value) > 10:  # Mínimo de 10 caracteres
                    if potential_input is None:
                        potential_input = value
                    elif potential_output is None:
                        potential_output = value
                        break
            
            if potential_input and potential_output:
                logger.info(f"⚠️ Usando campos genéricos: '{keys[0]}' -> '{keys[1]}'")
                return f"<|user|>\n{potential_input}</s>\n<|assistant|>\n{potential_output}</s>"
        
        return None
    
    def prepare_dataset(self, data_path: str):
        """Prepara dataset com análise automática"""
        try:
            if not data_path:
                raise ValueError("Caminho do arquivo de dados é obrigatório")

            logger.info(f"📂 Carregando: {data_path}")

            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.json'):
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                else:
                    data = [json.loads(line) for line in f if line.strip()]
            
            MAX_SAMPLES = 5000
            if len(data) > MAX_SAMPLES:
                logger.warning(f"⚠️ Limitando de {len(data)} para {MAX_SAMPLES} exemplos")
                data = data[:MAX_SAMPLES]
            
            logger.info(f"📊 Total: {len(data)} exemplos")
            
            # ANÁLISE DA ESTRUTURA
            self._analyze_json_structure(data)
            
            # Formatar
            formatted = []
            rejected = 0
            
            for i, item in enumerate(data):
                text = self._format_item(item)
                if text and len(text.strip()) > 20:
                    formatted.append({'text': text})
                else:
                    rejected += 1
                    if rejected <= 3:  # Mostrar primeiros 3 rejeitados
                        logger.warning(f"⚠️ Item {i+1} rejeitado (muito curto ou formato inválido)")
            
            logger.info("=" * 60)
            logger.info(f"✓ Aceitos: {len(formatted)} exemplos")
            logger.info(f"✗ Rejeitados: {rejected} exemplos")
            logger.info("=" * 60)
            
            if not formatted:
                logger.error("=" * 60)
                logger.error("❌ NENHUM DADO VÁLIDO ENCONTRADO!")
                logger.error("=" * 60)
                logger.error("🔧 SOLUÇÕES:")
                logger.error("1. Verifique se o JSON tem um dos formatos:")
                logger.error("   • {'instruction': '...', 'output': '...'}")
                logger.error("   • {'prompt': '...', 'completion': '...'}")
                logger.error("   • {'input': '...', 'output': '...'}")
                logger.error("   • {'question': '...', 'answer': '...'}")
                logger.error("   • {'pergunta': '...', 'resposta': '...'}")
                logger.error("   • {'text': '...'}")
                logger.error("2. Cada campo deve ter pelo menos 20 caracteres")
                logger.error("3. Verifique o exemplo acima para ver a estrutura real")
                logger.error("=" * 60)
                raise ValueError("Nenhum dado válido - verifique estrutura do JSON")
            
            dataset = Dataset.from_list(formatted)
            
            # Tokenizar
            def tokenize(examples):
                max_len = self.config.get('training', {}).get('max_length', 384)
                # Adicionando `return_token_type_ids=False` para evitar o erro `token_type_ids`
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=max_len,
                    padding='max_length',
                    return_tensors=None,
                    return_token_type_ids=False # CORREÇÃO APLICADA AQUI
                )
            
            self.dataset = dataset.map(
                tokenize,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizando"
            )
            
            logger.info(f"✓ Dataset pronto: {len(self.dataset)} exemplos")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Erro no dataset: {e}")
            raise
    
    def train(self) -> Dict:
        """Treina de forma otimizada"""
        try:
            logger.info("=" * 60)
            logger.info("INICIANDO TREINAMENTO OTIMIZADO")
            logger.info("=" * 60)
            
            training_cfg = self.config.get('training', {})
            
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            training_args = TrainingArguments(
                output_dir=self.adapter_dir,
                num_train_epochs=training_cfg.get('epochs', 1),
                per_device_train_batch_size=training_cfg.get('batch_size', 1),
                gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 8),
                learning_rate=training_cfg.get('learning_rate', 2e-4),
                logging_steps=training_cfg.get('logging_steps', 5),
                save_steps=99999,
                save_total_limit=1,
                save_only_model=True,
                optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
                fp16=False,
                bf16=False,
                max_grad_norm=0.2,
                warmup_ratio=training_cfg.get('warmup_steps', 5) / len(self.dataset) if len(self.dataset) > 0 else 0.03,
                group_by_length=True,
                lr_scheduler_type="cosine",
                report_to="none",
                disable_tqdm=False,
                remove_unused_columns=False
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                processing_class=self.tokenizer,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            )
            
            logger.info("🚀 Começando o treinamento...")
            trainer.train()
            logger.info("✅ Treinamento concluído.")
            
            # Salvar adaptador LoRA
            trainer.model.save_pretrained(self.adapter_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(self.adapter_dir, safe_serialization=True)
            logger.info(f"✓ Adaptador LoRA salvo em: {self.adapter_dir}")
            
            # ⭐ ÚNICA CHAMADA DE save_model() - AQUI ⭐
            output_path = self.config.get('training', {}).get('output_model_path', 'modelos/modelo_treinado')
            logger.info(f"💾 Salvando modelo final em: {output_path}")
            final_model_path = self.save_model(output_path=output_path)
            
            logger.info("=" * 60)
            logger.info("✅ TREINAMENTO FINALIZADO COM SUCESSO")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'log_history': trainer.state.log_history if hasattr(trainer.state, 'log_history') else [],
                'output_path': final_model_path,
                'message': 'Treinamento concluído com sucesso'
            }
            
        except Exception as e:
            logger.error(f"❌ Erro durante o treinamento: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'message': f'Erro durante o treinamento: {e}'
            }
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def save_model(self, output_path: str = None):
        """Salva modelo com LoRA fundido + tokenizer completo"""
        temp_merged_hf_dir = Path("temp_merged_final")
        try:
            logger.info("=" * 60)
            logger.info("SALVANDO MODELO TREINADO (COM FUSÃO LORA)")
            logger.info("=" * 60)
            
            # PASSO 1: FUNDIR LORA NO MODELO BASE (ENGORDAMENTO)
            logger.info("🔥 Passo 1/5: Fundindo pesos LoRA no modelo base...")            
            if isinstance(self.model, PeftModel):
                logger.info("   ℹ️ Modelo é PeftModel, aplicando merge_and_unload...")
                try:
                    # MERGE + UNLOAD = modelo "engordado" com conhecimento do treinamento
                    self.model = self.model.merge_and_unload()
                    logger.info("   ✅ LoRA fundido no modelo base!")
                except Exception as e:
                    logger.error(f"   ❌ Erro na fusão: {e}")
                    raise
            else:
                logger.warning("   ⚠️ Modelo não é PeftModel - pode não ter treinamento LoRA")
            
            # PASSO 2: PREPARAR DIRETÓRIO DE SAÍDA
            logger.info("📁 Passo 2/5: Preparando diretório de saída...")
            
            output_path = output_path or self.config.get('training', {}).get('output_model_path', 'modelos/modelo_treinado')
            output_path = Path(output_path)
            
            if output_path.suffix == '.gguf':
                output_path = output_path.with_suffix('')
                logger.info(f"   ⚠️ Extensão .gguf removida: {output_path}")
            
            temp_merged_hf_dir.mkdir(exist_ok=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"   ✓ Output: {output_path}")
            
            # PASSO 3: SALVAR MODELO FUNDIDO
            logger.info("💾 Passo 3/5: Salvando modelo fundido...")
            
            try:
                self.model.save_pretrained(
                    temp_merged_hf_dir, 
                    safe_serialization=True,
                    max_shard_size="500MB"
                )
                logger.info("   ✅ Modelo salvo em SafeTensors (shards de 500MB)")
            except Exception as e:
                logger.error(f"   ❌ Erro ao salvar modelo: {e}")
                logger.info("   🔄 Tentando salvar sem sharding...")
                try:
                    self.model.save_pretrained(
                        temp_merged_hf_dir, 
                        safe_serialization=True
                    )
                    logger.info("   ✅ Modelo salvo sem sharding")
                except Exception as e2:
                    logger.error(f"   ❌ Erro final: {e2}")
                    raise
            
            # PASSO 4: SALVAR TOKENIZER (CRÍTICO!)
            logger.info("📤 Passo 4/5: Salvando tokenizer...")
            
            try:
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(temp_merged_hf_dir)
                    logger.info("   ✅ Tokenizer salvo")
                    
                    # Verificar arquivos críticos
                    critical_files = [
                        'tokenizer.json',
                        'tokenizer_config.json',
                        'special_tokens_map.json'
                    ]
                    
                    missing = []
                    for file in critical_files:
                        if not (temp_merged_hf_dir / file).exists():
                            missing.append(file)
                    
                    if missing:
                        logger.warning(f"   ⚠️ Arquivos faltando: {missing}")
                        
                        # Criar tokenizer_config.json mínimo
                        tokenizer_config = {
                            "tokenizer_class": "LlamaTokenizer",
                            "model_max_length": 2048,
                            "padding_side": "right",
                            "pad_token": str(self.tokenizer.pad_token) if hasattr(self.tokenizer, 'pad_token') else "<pad>",
                            "eos_token": str(self.tokenizer.eos_token) if hasattr(self.tokenizer, 'eos_token') else "</s>",
                            "bos_token": str(self.tokenizer.bos_token) if hasattr(self.tokenizer, 'bos_token') else "<s>"
                        }
                        
                        with open(temp_merged_hf_dir / 'tokenizer_config.json', 'w', encoding='utf-8') as f:
                            import json
                            json.dump(tokenizer_config, f, indent=2)
                        
                        logger.info("   ✓ tokenizer_config.json criado")
                else:
                    logger.error("   ❌ Tokenizer é None!")
                    raise ValueError("Tokenizer não disponível para salvar")
                    
            except Exception as e:
                logger.error(f"   ❌ Erro ao salvar tokenizer: {e}")
                raise
            
            # PASSO 5: MOVER PARA DESTINO FINAL
            logger.info("📦 Passo 5/5: Movendo para destino final...")
            
            if output_path.exists():
                logger.info(f"   🗑️ Removendo {output_path} antigo...")
                shutil.rmtree(output_path)
            
            shutil.copytree(temp_merged_hf_dir, output_path)
            logger.info(f"   ✅ Modelo movido para: {output_path}")
            
            # Verificar integridade final
            logger.info("🔍 Verificando integridade do modelo salvo...")
            
            required_files = ['config.json', 'tokenizer_config.json']
            model_files = list(output_path.glob('*.safetensors')) + list(output_path.glob('model.safetensors.index.json'))
            
            if not model_files:
                logger.error("   ❌ ERRO: Nenhum arquivo de modelo encontrado!")
                raise ValueError("Modelo não foi salvo corretamente")
            
            for req_file in required_files:
                if not (output_path / req_file).exists():
                    logger.warning(f"   ⚠️ Arquivo faltando: {req_file}")
            
            # Calcular tamanho
            total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024**2)
            
            logger.info("=" * 60)
            logger.info("✅ MODELO SALVO COM SUCESSO!")
            logger.info("=" * 60)
            logger.info(f"📂 Localização: {output_path}")
            logger.info(f"💾 Tamanho: {size_mb:.1f} MB")
            logger.info(f"📊 Arquivos de modelo: {len(model_files)}")
            logger.info(f"📤 Tokenizer: {'✓' if (output_path / 'tokenizer_config.json').exists() else '✗'}")
            logger.info(f"🔥 LoRA fundido: {'✓' if not isinstance(self.model, PeftModel) else '✗'}")
            logger.info("=" * 60)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO ao salvar modelo: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Limpeza agressiva de arquivos temporários
            logger.info("🧹 Limpando arquivos temporários...")
            
            # Limpar diretórios temporários
            temp_dirs = [temp_merged_hf_dir, Path(self.adapter_dir), Path("temp_merged"), Path("temp_model")]
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        logger.info(f"   ✓ Removido: {temp_dir}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Erro ao remover {temp_dir}: {e}")
            
            # Limpar cache do PyTorch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Forçar garbage collection
            gc.collect()
            
            logger.info("✓ Limpeza concluída - espaço em disco liberado")


    def create_model_from_scratch(self, model_name="USBABC", base_model="gemma-portuguese-luana-2b.Q2_K.gguf"):
        """Cria um modelo do zero com train.json incorporado"""
        try:
            if not HAS_TORCH:
                self.logger.error("❌ PyTorch não disponível. Instale as dependências necessárias.")
                return None
                
            self.logger.info(f"🔨 Criando modelo do zero: {model_name}")
            
            # Carregar modelo base
            self.logger.info(f"📥 Carregando modelo base: {base_model}")
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Incorporar train.json automaticamente
            train_path = "dados/train.json"
            if os.path.exists(train_path):
                self.logger.info(f"📂 Incorporando {train_path} no modelo...")
                self.prepare_dataset(train_path)
                
                # Aplicar LoRA para treinamento rápido
                target_modules = self._get_target_modules(model)
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=target_modules
                )
                
                model = get_peft_model(model, peft_config)
                
                # Treinamento rápido com train.json
                training_args = TrainingArguments(
                    output_dir="./temp_training",
                    num_train_epochs=1,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=8,
                    warmup_steps=10,
                    logging_steps=10,
                    save_steps=50,
                    evaluation_strategy="no",
                    save_strategy="epoch",
                    load_best_model_at_end=False,
                    report_to=None,
                    remove_unused_columns=False
                )
                
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=self.dataset,
                    tokenizer=tokenizer,
                    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                )
                
                self.logger.info("🔥 Treinando com train.json...")
                trainer.train()
                
                # Mesclar LoRA com modelo base
                self.logger.info("🔗 Mesclando LoRA com modelo base...")
                model = model.merge_and_unload()
            
            # Salvar modelo temporário
            temp_dir = f"temp_{model_name}"
            os.makedirs(temp_dir, exist_ok=True)
            
            model.save_pretrained(temp_dir, safe_serialization=True)
            tokenizer.save_pretrained(temp_dir, safe_serialization=True)
            
            # Salvar modelo final em SafeTensors (formato original)
            final_model_dir = f"modelos/{model_name.lower()}"
            os.makedirs("modelos", exist_ok=True)
            
            self.logger.info(f"💾 Salvando modelo em SafeTensors: {final_model_dir}")
            
            # Mover modelo do temp para diretório final
            if os.path.exists(final_model_dir):
                shutil.rmtree(final_model_dir)
            shutil.move(temp_dir, final_model_dir)
            
            # Adicionar metadados
            metadata = {
                "name": model_name,
                "base_model": base_model,
                "format": "safetensors",
                "trained_with": "train.json" if os.path.exists(train_path) else None,
                "created": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(final_model_dir, "model_info.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Modelo criado em SafeTensors: {final_model_dir}")
            return final_model_dir
                
        except Exception as e:
            self.logger.error(f"❌ Erro na criação: {e}")
            return None

# O restante do script (função main) é mantido inalterado
def main():
    """Função principal de execução"""
    try:
        # Tenta carregar o arquivo de dados do upload
        data_path = Path("/home/ubuntu/upload/pasted_content.txt")
        if not data_path.exists():
            # Tenta um nome de arquivo de dados padrão
            data_path = Path("dados/data.json")
            if not data_path.exists():
                logger.error("❌ Arquivo de dados não encontrado. Certifique-se de que 'pasted_content.txt' ou 'dados/data.json' existe.")
                return

        # Instanciar o sistema
        system = AITrainingSystem()
        
        # O script original não passa o model_path para load_model na main, 
        # mas a função load_model exige. Vamos usar um nome de modelo padrão
        # para que o script possa ser executado, e o usuário deve ajustar.
        
        model_to_load = system.config.get('training', {}).get('base_model_path', 'gemma-portuguese-luana-2b.Q2_K.gguf')
        
        # 1. Carregar modelo
        system.load_model(model_to_load)
        
        # 2. Preparar dataset
        system.prepare_dataset(str(data_path))
        
        # 3. Treinar
        # A chamada a save_model foi movida para dentro de train() para garantir que ocorra após o treinamento.
        system.train()
        
        logger.info("🎉 Processo concluído com sucesso!")

    except Exception as e:
        logger.error(f"❌ Falha crítica na execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

