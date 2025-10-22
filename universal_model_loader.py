#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISSÃO 2: Sistema Universal de Carregamento de Modelos - CORRIGIDO
Suporta: GGUF, SafeTensors, BIN, PT, PTH, ZIP
Com validação robusta e tratamento de erros aprimorado
"""

import os
import json
import torch
import zipfile
import logging
import tempfile
import shutil
import struct
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM
)
import safetensors
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)

class UniversalModelLoader:
    """Carregador universal para todos os tipos de modelo"""
    
    def __init__(self):
        self.supported_formats = {
            '.gguf': self._load_gguf,
            '.safetensors': self._load_safetensors,
            '.bin': self._load_bin,
            '.pt': self._load_pytorch,
            '.pth': self._load_pytorch,
            '.zip': self._load_zip
        }
        self.temp_dirs = []
        
    def __del__(self):
        """Limpar diretórios temporários"""
        self._cleanup_temp_dirs()
    
    def _cleanup_temp_dirs(self):
        """Limpar diretórios temporários criados"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Erro ao limpar diretório temporário {temp_dir}: {e}")
        self.temp_dirs.clear()
    
    def _validate_gguf_file(self, file_path: Path) -> Tuple[bool, str]:
        """Valida arquivo GGUF de forma robusta"""
        try:
            if not file_path.exists():
                return False, "Arquivo não encontrado"
            
            file_size = file_path.stat().st_size
            if file_size < 100:  # Arquivo muito pequeno
                return False, f"Arquivo muito pequeno ({file_size} bytes)"
            
            with open(file_path, 'rb') as f:
                # Ler magic number
                magic = f.read(4)
                if magic != b'GGUF':
                    return False, f"Magic number inválido: {magic}"
                
                # Ler versão
                version_bytes = f.read(4)
                if len(version_bytes) < 4:
                    return False, "Arquivo truncado (versão)"
                
                version = struct.unpack('<I', version_bytes)[0]
                if version < 1 or version > 3:
                    return False, f"Versão GGUF não suportada: {version}"
                
                # Ler contadores
                tensor_count_bytes = f.read(8)
                metadata_count_bytes = f.read(8)
                
                if len(tensor_count_bytes) < 8 or len(metadata_count_bytes) < 8:
                    return False, "Arquivo truncado (contadores)"
                
                tensor_count = struct.unpack('<Q', tensor_count_bytes)[0]
                metadata_count = struct.unpack('<Q', metadata_count_bytes)[0]
                
                if tensor_count == 0:
                    return False, "Arquivo sem tensores"
                
                logger.info(f"✅ GGUF válido: v{version}, {tensor_count} tensores, {metadata_count} metadados")
                return True, "OK"
                
        except Exception as e:
            return False, f"Erro na validação: {str(e)}"
    
    def load_model(self, model_path: Union[str, Path], **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Carrega qualquer tipo de modelo suportado
        
        Args:
            model_path: Caminho para o modelo
            **kwargs: Argumentos adicionais para carregamento
            
        Returns:
            Tuple[model, tokenizer, metadata]
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        logger.info(f"🔍 Carregando modelo: {model_path}")
        
        # Determinar tipo do arquivo
        if model_path.is_dir():
            return self._load_directory(model_path, **kwargs)
        
        file_ext = model_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Formato não suportado: {file_ext}")
        
        # Carregar usando o método apropriado
        loader_func = self.supported_formats[file_ext]
        return loader_func(model_path, **kwargs)
    
    def _load_directory(self, model_dir: Path, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Carrega modelo de um diretório (formato HuggingFace)"""
        try:
            logger.info(f"📂 Carregando modelo HuggingFace: {model_dir}")
            
            # Verificar se é um modelo válido
            config_path = model_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"config.json não encontrado em {model_dir}")
            
            # Carregar configuração
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Carregar tokenizer
            tokenizer = self._load_tokenizer_safe(model_dir)
            
            # Carregar modelo
            model = self._load_model_safe(model_dir, **kwargs)
            
            metadata = {
                'format': 'huggingface',
                'path': str(model_dir),
                'config': config_data,
                'model_type': config_data.get('model_type', 'unknown'),
                'vocab_size': config_data.get('vocab_size', 0),
                'hidden_size': config_data.get('hidden_size', 0)
            }
            
            logger.info(f"✅ Modelo HuggingFace carregado: {config_data.get('model_type', 'unknown')}")
            return model, tokenizer, metadata
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar diretório {model_dir}: {e}")
            raise
    
    def _load_gguf(self, model_path: Path, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Carrega modelo GGUF com validação robusta"""
        try:
            logger.info(f"🔧 Carregando modelo GGUF: {model_path}")
            
            # VALIDAÇÃO CRÍTICA: Verificar se arquivo é válido
            is_valid, validation_msg = self._validate_gguf_file(model_path)
            if not is_valid:
                logger.error(f"❌ Arquivo GGUF inválido: {validation_msg}")
                raise ValueError(f"Arquivo GGUF inválido: {validation_msg}")
            
            # Verificar se existe um diretório com o mesmo nome
            model_dir = model_path.parent / model_path.stem
            if model_dir.exists() and model_dir.is_dir():
                logger.info(f"📂 Encontrado diretório associado: {model_dir}")
                return self._load_directory(model_dir, **kwargs)
            
            # Tentar carregar com llama-cpp-python
            try:
                from llama_cpp import Llama
                logger.info("🔧 Carregando GGUF com llama-cpp-python...")
                
                # Configurações mais conservadoras para evitar erros
                load_config = {
                    'model_path': str(model_path),
                    'n_ctx': 2048,
                    'n_threads': min(os.cpu_count() or 4, 8),
                    'verbose': False,
                    'use_mmap': True,
                    'use_mlock': False
                }
                
                # Adicionar GPU se disponível
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_gb = gpu_memory / (1024**3)
                    
                    if gpu_memory_gb > 8:
                        load_config['n_gpu_layers'] = -1  # Todas camadas
                    elif gpu_memory_gb > 4:
                        load_config['n_gpu_layers'] = 20
                    else:
                        load_config['n_gpu_layers'] = 10
                    
                    load_config['n_batch'] = 512
                else:
                    load_config['n_batch'] = 256
                
                logger.info(f"📋 Configuração de carregamento: {load_config}")
                
                # Tentar carregar com timeout implícito
                try:
                    llm = Llama(**load_config)
                except Exception as e1:
                    logger.warning(f"⚠️ Primeira tentativa falhou: {e1}")
                    logger.info("🔄 Tentando sem GPU...")
                    
                    # Fallback sem GPU
                    load_config['n_gpu_layers'] = 0
                    load_config['n_batch'] = 128
                    load_config['n_ctx'] = 1024
                    
                    llm = Llama(**load_config)
                
                metadata = {
                    'format': 'gguf',
                    'path': str(model_path),
                    'size': model_path.stat().st_size,
                    'model_type': 'gguf',
                    'quantized': True,
                    'loaded_with': 'llama-cpp-python',
                    'valid': True
                }
                
                logger.info(f"✅ GGUF carregado com llama-cpp-python: {model_path}")
                return llm, None, metadata
                
            except ImportError:
                logger.error("❌ llama-cpp-python não disponível")
                logger.error("💡 Instale com: pip install llama-cpp-python")
                raise ImportError("llama-cpp-python não instalado")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar GGUF {model_path}: {e}")
            raise
    
    def _load_safetensors(self, model_path: Path, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Carrega modelo SafeTensors"""
        try:
            logger.info(f"🔒 Carregando modelo SafeTensors: {model_path}")
            
            # Verificar se existe um diretório com arquivos de configuração
            model_dir = model_path.parent
            config_path = model_dir / "config.json"
            
            if config_path.exists():
                # Verificar se existe model.safetensors no diretório
                expected_model_path = model_dir / "model.safetensors"
                if not expected_model_path.exists() and model_path.name != "model.safetensors":
                    # Criar link simbólico ou copiar arquivo com nome esperado
                    import shutil
                    shutil.copy2(model_path, expected_model_path)
                    logger.info(f"📋 Arquivo copiado para nome padrão: {expected_model_path}")
                
                # Carregar como modelo HuggingFace com SafeTensors
                return self._load_directory(model_dir, **kwargs)
            
            # Carregar SafeTensors diretamente
            state_dict = load_safetensors(model_path)
            
            metadata = {
                'format': 'safetensors',
                'path': str(model_path),
                'size': model_path.stat().st_size,
                'tensors': list(state_dict.keys()),
                'tensor_count': len(state_dict)
            }
            
            logger.info(f"✅ SafeTensors carregado: {len(state_dict)} tensores")
            
            # Retornar state_dict como modelo
            return state_dict, None, metadata
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar SafeTensors {model_path}: {e}")
            raise
    
    def _load_bin(self, model_path: Path, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Carrega modelo BIN (PyTorch) com detecção automática de state_dict"""
        try:
            logger.info(f"📦 Carregando modelo BIN: {model_path}")
            
            # Verificar se existe um diretório com arquivos de configuração
            model_dir = model_path.parent
            config_path = model_dir / "config.json"
            
            if config_path.exists():
                # Carregar como modelo HuggingFace
                logger.info("🤗 Detectado modelo HuggingFace com BIN")
                return self._load_directory(model_dir, **kwargs)
            
            # Carregar BIN diretamente com diferentes estratégias
            logger.info("🔧 Carregando BIN diretamente...")
            
            # Tentar primeiro com weights_only=True (mais seguro)
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                logger.info("✅ BIN carregado com weights_only=True")
            except Exception as e1:
                logger.warning(f"⚠️ weights_only=True falhou: {e1}")
                try:
                    # Fallback sem weights_only
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    logger.info("✅ BIN carregado com weights_only=False")
                except Exception as e2:
                    logger.error(f"❌ Ambas as tentativas falharam: {e2}")
                    raise
            
            # Detectar e extrair state_dict automaticamente
            state_dict, extra_info = self._extract_state_dict(checkpoint, 'BIN')
            
            metadata = {
                'format': 'pytorch_bin',
                'path': str(model_path),
                'size': model_path.stat().st_size,
                'tensors': list(state_dict.keys()) if isinstance(state_dict, dict) else [],
                'tensor_count': len(state_dict) if isinstance(state_dict, dict) else 0,
                'extra_info': extra_info
            }
            
            logger.info(f"✅ BIN carregado: {metadata['tensor_count']} tensores")
            return state_dict, None, metadata
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar BIN {model_path}: {e}")
            raise
    
    def _load_pytorch(self, model_path: Path, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Carrega modelo PyTorch (.pt/.pth) com detecção automática de state_dict"""
        try:
            logger.info(f"🔥 Carregando modelo PyTorch: {model_path}")
            
            # Tentar carregar com diferentes estratégias
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                logger.info("✅ PyTorch carregado com weights_only=True")
            except Exception as e1:
                logger.warning(f"⚠️ weights_only=True falhou: {e1}")
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    logger.info("✅ PyTorch carregado com weights_only=False")
                except Exception as e2:
                    logger.error(f"❌ Ambas as tentativas falharam: {e2}")
                    raise
            
            # Detectar e extrair state_dict automaticamente
            state_dict, extra_info = self._extract_state_dict(checkpoint, 'PyTorch')
            
            metadata = {
                'format': 'pytorch',
                'path': str(model_path),
                'size': model_path.stat().st_size,
                'tensors': list(state_dict.keys()) if isinstance(state_dict, dict) else [],
                'tensor_count': len(state_dict) if isinstance(state_dict, dict) else 0,
                'extra_info': extra_info
            }
            
            logger.info(f"✅ PyTorch carregado: {metadata['tensor_count']} tensores")
            return state_dict, None, metadata
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar PyTorch {model_path}: {e}")
            raise
    
    def _extract_state_dict(self, checkpoint: Any, format_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extrai state_dict de um checkpoint com detecção automática"""
        extra_info = {
            'original_type': type(checkpoint).__name__,
            'checkpoint_keys': [],
            'extraction_method': 'direct'
        }
        
        if not isinstance(checkpoint, dict):
            logger.info(f"📄 {format_name}: Checkpoint não é dict, usando diretamente")
            return checkpoint, extra_info
        
        # Registrar chaves do checkpoint
        extra_info['checkpoint_keys'] = list(checkpoint.keys())
        logger.info(f"🔑 {format_name}: Chaves encontradas: {extra_info['checkpoint_keys']}")
        
        # Estratégias de extração em ordem de prioridade
        extraction_strategies = [
            ('model_state_dict', 'model_state_dict'),
            ('state_dict', 'state_dict'),
            ('model', 'model'),
            ('net', 'net'),
            ('network', 'network'),
            ('weights', 'weights'),
            ('parameters', 'parameters')
        ]
        
        # Tentar cada estratégia
        for key, method in extraction_strategies:
            if key in checkpoint:
                state_dict = checkpoint[key]
                if isinstance(state_dict, dict) and len(state_dict) > 0:
                    # Verificar se parece com um state_dict válido
                    if self._is_valid_state_dict(state_dict):
                        extra_info['extraction_method'] = method
                        logger.info(f"✅ {format_name}: State_dict extraído usando '{method}' ({len(state_dict)} tensores)")
                        return state_dict, extra_info
        
        # Se nenhuma estratégia funcionou, verificar se o próprio checkpoint é um state_dict
        if self._is_valid_state_dict(checkpoint):
            extra_info['extraction_method'] = 'direct_checkpoint'
            logger.info(f"✅ {format_name}: Checkpoint é um state_dict válido ({len(checkpoint)} tensores)")
            return checkpoint, extra_info
        
        # Última tentativa: procurar por qualquer dict que pareça um state_dict
        for key, value in checkpoint.items():
            if isinstance(value, dict) and self._is_valid_state_dict(value):
                extra_info['extraction_method'] = f'found_in_{key}'
                logger.info(f"✅ {format_name}: State_dict encontrado em '{key}' ({len(value)} tensores)")
                return value, extra_info
        
        # Se chegou aqui, não conseguiu extrair um state_dict válido
        logger.warning(f"⚠️ {format_name}: Não foi possível extrair state_dict válido, retornando checkpoint original")
        extra_info['extraction_method'] = 'fallback_original'
        return checkpoint, extra_info
    
    def _is_valid_state_dict(self, obj: Any) -> bool:
        """Verifica se um objeto parece ser um state_dict válido"""
        if not isinstance(obj, dict):
            return False
        
        if len(obj) == 0:
            return False
        
        # Verificar se pelo menos algumas chaves parecem nomes de parâmetros de modelo
        tensor_count = 0
        valid_keys = 0
        
        for key, value in obj.items():
            if isinstance(key, str):
                # Verificar se a chave parece nome de parâmetro
                if any(pattern in key.lower() for pattern in [
                    'weight', 'bias', 'embedding', 'linear', 'conv', 'norm', 
                    'attention', 'layer', 'block', 'head', 'mlp', 'ffn'
                ]):
                    valid_keys += 1
                
                # Verificar se o valor é um tensor
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    tensor_count += 1
        
        # Considerar válido se pelo menos 50% das chaves parecem válidas
        # e pelo menos 50% dos valores são tensores
        key_ratio = valid_keys / len(obj) if len(obj) > 0 else 0
        tensor_ratio = tensor_count / len(obj) if len(obj) > 0 else 0
        
        is_valid = key_ratio >= 0.3 and tensor_ratio >= 0.5
        
        if is_valid:
            logger.debug(f"✅ State_dict válido: {valid_keys}/{len(obj)} chaves válidas, {tensor_count}/{len(obj)} tensores")
        else:
            logger.debug(f"❌ State_dict inválido: {valid_keys}/{len(obj)} chaves válidas, {tensor_count}/{len(obj)} tensores")
        
        return is_valid
    
    def _load_zip(self, model_path: Path, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Carrega modelo de arquivo ZIP com detecção inteligente de estrutura"""
        try:
            logger.info(f"📦 Extraindo modelo ZIP: {model_path}")
            
            # Verificar se o ZIP é válido
            if not zipfile.is_zipfile(model_path):
                raise ValueError(f"Arquivo não é um ZIP válido: {model_path}")
            
            # Criar diretório temporário
            temp_dir = tempfile.mkdtemp(prefix="model_zip_")
            self.temp_dirs.append(temp_dir)
            
            # Analisar conteúdo do ZIP antes de extrair
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                logger.info(f"📋 Arquivos no ZIP: {len(file_list)} itens")
                
                # Detectar estrutura do ZIP
                model_files = []
                config_files = []
                tokenizer_files = []
                
                for file_name in file_list:
                    file_lower = file_name.lower()
                    if file_lower.endswith(('.safetensors', '.bin', '.pt', '.pth', '.gguf')):
                        model_files.append(file_name)
                    elif file_lower.endswith('config.json'):
                        config_files.append(file_name)
                    elif 'tokenizer' in file_lower and file_lower.endswith('.json'):
                        tokenizer_files.append(file_name)
                
                logger.info(f"🔍 Detectados: {len(model_files)} modelos, {len(config_files)} configs, {len(tokenizer_files)} tokenizers")
                
                # Extrair com segurança (evitar zip bombs)
                total_size = 0
                for file_info in zip_ref.infolist():
                    total_size += file_info.file_size
                    if total_size > 50 * 1024 * 1024 * 1024:  # 50GB limite
                        raise ValueError("ZIP muito grande (possível zip bomb)")
                
                # Extrair arquivos
                zip_ref.extractall(temp_dir)
            
            logger.info(f"📂 ZIP extraído para: {temp_dir}")
            
            # Estratégias de detecção de modelo
            strategies = [
                self._find_huggingface_model,
                self._find_single_model_file,
                self._find_nested_model,
                self._find_any_model_file
            ]
            
            for strategy in strategies:
                try:
                    result = strategy(Path(temp_dir), **kwargs)
                    if result:
                        return result
                except Exception as e:
                    logger.warning(f"⚠️ Estratégia {strategy.__name__} falhou: {e}")
                    continue
            
            raise ValueError(f"Nenhum modelo válido encontrado no ZIP: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar ZIP {model_path}: {e}")
            raise
    
    def _find_huggingface_model(self, temp_dir: Path, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Procura por modelo HuggingFace completo"""
        # Procurar por diretório com config.json
        for item in temp_dir.rglob('config.json'):
            model_dir = item.parent
            logger.info(f"🤗 Encontrado modelo HuggingFace: {model_dir}")
            return self._load_directory(model_dir, **kwargs)
        return None
    
    def _find_single_model_file(self, temp_dir: Path, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Procura por arquivo único de modelo"""
        model_files = []
        for ext in ['.safetensors', '.bin', '.pt', '.pth', '.gguf']:
            model_files.extend(list(temp_dir.rglob(f'*{ext}')))
        
        if len(model_files) == 1:
            logger.info(f"📄 Encontrado arquivo único de modelo: {model_files[0]}")
            return self.load_model(model_files[0], **kwargs)
        return None
    
    def _find_nested_model(self, temp_dir: Path, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Procura por modelo em subdiretórios"""
        for subdir in temp_dir.iterdir():
            if subdir.is_dir():
                try:
                    return self._load_directory(subdir, **kwargs)
                except Exception:
                    continue
        return None
    
    def _find_any_model_file(self, temp_dir: Path, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Procura por qualquer arquivo de modelo válido"""
        for ext in ['.safetensors', '.bin', '.pt', '.pth', '.gguf']:
            for model_file in temp_dir.rglob(f'*{ext}'):
                logger.info(f"🔍 Tentando carregar: {model_file}")
                try:
                    return self.load_model(model_file, **kwargs)
                except Exception as e:
                    logger.warning(f"⚠️ Falha ao carregar {model_file}: {e}")
                    continue
        return None
    
    def _load_tokenizer_safe(self, model_path: Union[str, Path]) -> Any:
        """Carrega tokenizer com fallbacks seguros"""
        try:
            # Tentar AutoTokenizer primeiro
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                use_fast=False
            )
            logger.info("✅ AutoTokenizer carregado")
            return tokenizer
            
        except Exception as e1:
            logger.warning(f"⚠️ AutoTokenizer falhou: {e1}")
            
            try:
                # Tentar LlamaTokenizer
                tokenizer = LlamaTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=True
                )
                logger.info("✅ LlamaTokenizer carregado")
                return tokenizer
                
            except Exception as e2:
                logger.warning(f"⚠️ LlamaTokenizer falhou: {e2}")
                
                # Retornar None se não conseguir carregar
                logger.warning("⚠️ Nenhum tokenizer carregado")
                return None
    
    def _load_model_safe(self, model_path: Union[str, Path], **kwargs) -> Any:
        """Carrega modelo com fallbacks seguros e otimização CUDA"""
        try:
            # Detectar se CUDA está disponível
            device_available = torch.cuda.is_available()
            logger.info(f"🎮 CUDA disponível: {device_available}")
            
            # Configurações otimizadas baseadas no hardware disponível
            if device_available:
                load_kwargs = {
                    'trust_remote_code': True,
                    'torch_dtype': torch.float16,
                    'device_map': 'auto',
                    'low_cpu_mem_usage': True,
                    'max_memory': {0: "6GB"} if torch.cuda.device_count() > 0 else None,
                    **kwargs
                }
            else:
                load_kwargs = {
                    'trust_remote_code': True,
                    'torch_dtype': torch.float32,
                    'device_map': 'cpu',
                    'low_cpu_mem_usage': True,
                    **kwargs
                }
            
            logger.info(f"🔧 Configurações de carregamento: {load_kwargs}")
            
            # Tentar AutoModelForCausalLM primeiro
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    **load_kwargs
                )
                logger.info("✅ AutoModelForCausalLM carregado")
                
                # Mover para GPU se disponível e não foi feito automaticamente
                if device_available and hasattr(model, 'to'):
                    try:
                        model = model.to('cuda')
                        logger.info("🎮 Modelo movido para GPU")
                    except Exception as e:
                        logger.warning(f"⚠️ Não foi possível mover para GPU: {e}")
                
                return model
                
            except Exception as e1:
                logger.warning(f"⚠️ AutoModelForCausalLM falhou: {e1}")
                
                # Tentar LlamaForCausalLM
                try:
                    model = LlamaForCausalLM.from_pretrained(
                        str(model_path),
                        **load_kwargs
                    )
                    logger.info("✅ LlamaForCausalLM carregado")
                    
                    # Mover para GPU se disponível
                    if device_available and hasattr(model, 'to'):
                        try:
                            model = model.to('cuda')
                            logger.info("🎮 Modelo movido para GPU")
                        except Exception as e:
                            logger.warning(f"⚠️ Não foi possível mover para GPU: {e}")
                    
                    return model
                    
                except Exception as e2:
                    logger.warning(f"⚠️ LlamaForCausalLM falhou: {e2}")
                    
                    # Tentar sem device_map (fallback para CPU)
                    try:
                        fallback_kwargs = load_kwargs.copy()
                        fallback_kwargs.pop('device_map', None)
                        fallback_kwargs.pop('max_memory', None)
                        fallback_kwargs['torch_dtype'] = torch.float32
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            str(model_path),
                            **fallback_kwargs
                        )
                        logger.info("✅ AutoModelForCausalLM carregado (fallback CPU)")
                        return model
                        
                    except Exception as e3:
                        logger.error(f"❌ Todos os métodos de carregamento falharam: {e3}")
                        raise
                        
        except Exception as e:
            logger.error(f"❌ Erro fatal no carregamento do modelo: {e}")
            raise
    
    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Obtém informações sobre um modelo sem carregá-lo completamente"""
        model_path = Path(model_path)
        
        info = {
            'path': str(model_path),
            'exists': model_path.exists(),
            'size': model_path.stat().st_size if model_path.exists() else 0,
            'format': 'unknown',
            'supported': False,
            'valid': False,
            'validation_message': ''
        }
        
        if not model_path.exists():
            info['validation_message'] = 'Arquivo não encontrado'
            return info
        
        if model_path.is_dir():
            info['format'] = 'directory'
            info['supported'] = True
            
            # Verificar arquivos no diretório
            config_path = model_path / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    info['config'] = config
                    info['model_type'] = config.get('model_type', 'unknown')
                    info['valid'] = True
                    info['validation_message'] = 'OK'
                except Exception as e:
                    info['validation_message'] = f'Erro ao ler config: {str(e)}'
        else:
            file_ext = model_path.suffix.lower()
            info['format'] = file_ext
            info['supported'] = file_ext in self.supported_formats
            
            # Validação específica para GGUF
            if file_ext == '.gguf':
                is_valid, msg = self._validate_gguf_file(model_path)
                info['valid'] = is_valid
                info['validation_message'] = msg
        
        return info


# Instância global do carregador
universal_loader = UniversalModelLoader()


def load_any_model(model_path: Union[str, Path], **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Função conveniente para carregar qualquer modelo
    
    Args:
        model_path: Caminho para o modelo
        **kwargs: Argumentos adicionais
        
    Returns:
        Tuple[model, tokenizer, metadata]
    """
    return universal_loader.load_model(model_path, **kwargs)


def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Função conveniente para obter informações do modelo
    
    Args:
        model_path: Caminho para o modelo
        
    Returns:
        Dict com informações do modelo
    """
    return universal_loader.get_model_info(model_path)


if __name__ == "__main__":
    # Teste do carregador
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"🔍 Analisando modelo: {model_path}")
        
        info = get_model_info(model_path)
        print(f"📊 Informações: {info}")
        
        if info['supported']:
            if info.get('valid', True):  # Se não tem validação ou é válido
                try:
                    model, tokenizer, metadata = load_any_model(model_path)
                    print(f"✅ Modelo carregado com sucesso!")
                    print(f"📋 Metadata: {metadata}")
                except Exception as e:
                    print(f"❌ Erro ao carregar: {e}")
            else:
                print(f"❌ Modelo inválido: {info['validation_message']}")
        else:
            print(f"❌ Formato não suportado: {info['format']}")
    else:
        print("Uso: python universal_model_loader.py <caminho_do_modelo>")