#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerenciador Universal de Arquivos - VERSÃO APRIMORADA
Sistema unificado para carregar, salvar e converter modelos de IA
Suporta: GGUF, SafeTensors, PyTorch, HuggingFace, USBABC
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from datetime import datetime
import shutil
import hashlib
from contextlib import contextmanager

# Imports condicionais
try:
    import gguf
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    logging.warning("GGUF não disponível - instale: pip install gguf")

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as safetensors_save
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logging.warning("SafeTensors não disponível - instale: pip install safetensors")

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers não disponível - instale: pip install transformers")

try:
    from modeling_usbabc import USBABCForCausalLM
    from configuration_usbabc import USBABCConfig
    USBABC_AVAILABLE = True
except ImportError:
    USBABC_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelLoadError(Exception):
    """Exceção customizada para erros de carregamento"""
    pass


class UniversalFileManager:
    """Gerenciador universal para todos os tipos de arquivos de modelo"""
    
    SUPPORTED_FORMATS = {
        'gguf': ['.gguf'],
        'safetensors': ['.safetensors'],
        'pytorch': ['.bin', '.pt', '.pth'],
        'huggingface': ['config.json', 'pytorch_model.bin', 'model.safetensors'],
        'usbabc': ['config.json', 'pytorch_model.bin', 'model_info.json']
    }
    
    # Cache de modelos carregados
    _model_cache = {}
    _max_cache_size = 3
    
    def __init__(self, base_dir: str = "modelos"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        # Diretórios organizados
        self.dirs = {
            'models': self.base_dir / 'models',
            'checkpoints': self.base_dir / 'checkpoints', 
            'exports': self.base_dir / 'exports',
            'backups': self.base_dir / 'backups',
            'temp': self.base_dir / 'temp',
            'cache': self.base_dir / 'cache'  # NOVO: cache de modelos
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Carregar cache de metadados
        self.metadata_cache_path = self.dirs['cache'] / 'metadata_cache.json'
        self.metadata_cache = self._load_metadata_cache()
    
    def _load_metadata_cache(self) -> Dict:
        """Carrega cache de metadados para acelerar listagens"""
        if self.metadata_cache_path.exists():
            try:
                with open(self.metadata_cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erro ao carregar cache de metadados: {e}")
        return {}
    
    def _save_metadata_cache(self):
        """Salva cache de metadados"""
        try:
            with open(self.metadata_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache de metadados: {e}")
    
    def _get_file_hash(self, path: Path, quick: bool = True) -> str:
        """Gera hash do arquivo para verificação de integridade"""
        hasher = hashlib.md5()
        
        if quick and path.is_file():
            # Hash rápido: apenas primeiros e últimos bytes + tamanho
            size = path.stat().st_size
            with open(path, 'rb') as f:
                hasher.update(f.read(8192))  # Primeiros 8KB
                if size > 16384:
                    f.seek(-8192, 2)  # Últimos 8KB
                    hasher.update(f.read(8192))
                hasher.update(str(size).encode())
        
        return hasher.hexdigest()
    
    def detect_format(self, path: Union[str, Path]) -> str:
        """Detecta o formato do modelo - VERSÃO MELHORADA"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Caminho não encontrado: {path}")
        
        if path.is_file():
            suffix = path.suffix.lower()
            
            if suffix == '.gguf':
                return 'gguf'
            elif suffix == '.safetensors':
                return 'safetensors'
            elif suffix in ['.bin', '.pt', '.pth']:
                # Verificar se é parte de um modelo HuggingFace
                parent_dir = path.parent
                if (parent_dir / 'config.json').exists():
                    return 'huggingface'
                return 'pytorch'
        
        elif path.is_dir():
            # Verificar arquivos no diretório
            files = list(path.glob('*'))
            file_names = [f.name for f in files]
            
            # Verificar USBABC (prioridade)
            if 'model_info.json' in file_names and 'config.json' in file_names:
                return 'usbabc'
            
            # Verificar HuggingFace
            if 'config.json' in file_names:
                model_files = ['pytorch_model.bin', 'model.safetensors', 
                              'pytorch_model.bin.index.json', 'model.safetensors.index.json']
                if any(f in file_names for f in model_files):
                    return 'huggingface'
            
            # Verificar se contém apenas arquivos PyTorch
            pytorch_files = [f for f in files if f.suffix in ['.bin', '.pt', '.pth']]
            if pytorch_files:
                return 'pytorch'
        
        return 'unknown'
    
    def validate_model(self, path: Union[str, Path]) -> Tuple[bool, str]:
        """Valida se o modelo pode ser carregado - NOVO"""
        path = Path(path)
        format_type = self.detect_format(path)
        
        try:
            if format_type == 'gguf':
                if not GGUF_AVAILABLE:
                    return False, "Biblioteca GGUF não instalada"
                if not path.is_file():
                    return False, "Arquivo GGUF não encontrado"
                # Tentar abrir o arquivo
                _ = gguf.GGUFReader(str(path))
                return True, "OK"
            
            elif format_type == 'safetensors':
                if not SAFETENSORS_AVAILABLE:
                    return False, "Biblioteca SafeTensors não instalada"
                if not path.is_file():
                    return False, "Arquivo SafeTensors não encontrado"
                # Tentar abrir
                with safe_open(path, framework="pt", device="cpu") as f:
                    _ = list(f.keys())
                return True, "OK"
            
            elif format_type == 'pytorch':
                if not path.exists():
                    return False, "Arquivo PyTorch não encontrado"
                # Tentar carregar
                _ = torch.load(path, map_location='cpu', weights_only=False)
                return True, "OK"
            
            elif format_type == 'huggingface':
                if not TRANSFORMERS_AVAILABLE:
                    return False, "Biblioteca Transformers não instalada"
                config_path = path / 'config.json' if path.is_dir() else path.parent / 'config.json'
                if not config_path.exists():
                    return False, "config.json não encontrado"
                # Verificar arquivos do modelo
                model_files = list(path.glob('*.bin')) + list(path.glob('*.safetensors'))
                if not model_files:
                    return False, "Arquivos do modelo não encontrados"
                return True, "OK"
            
            elif format_type == 'usbabc':
                required_files = ['config.json', 'model_info.json']
                for req_file in required_files:
                    if not (path / req_file).exists():
                        return False, f"{req_file} não encontrado"
                return True, "OK"
            
            else:
                return False, f"Formato não suportado: {format_type}"
        
        except Exception as e:
            return False, f"Erro na validação: {str(e)}"
    
    def load_model(self, path: Union[str, Path], 
                   use_cache: bool = True,
                   device: str = 'cpu',
                   **kwargs) -> Any:
        """Carrega modelo de forma inteligente - NOVO MÉTODO"""
        path = Path(path)
        cache_key = str(path.absolute())
        
        # Verificar cache
        if use_cache and cache_key in self._model_cache:
            logger.info(f"✅ Modelo carregado do cache: {path.name}")
            return self._model_cache[cache_key]
        
        # Validar antes de carregar
        is_valid, message = self.validate_model(path)
        if not is_valid:
            raise ModelLoadError(f"Validação falhou: {message}")
        
        format_type = self.detect_format(path)
        logger.info(f"📥 Carregando modelo {format_type}: {path.name}")
        
        try:
            model = None
            
            if format_type == 'pytorch':
                model = torch.load(path, map_location=device, weights_only=False)
            
            elif format_type == 'safetensors':
                state_dict = {}
                with safe_open(path, framework="pt", device=device) as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                model = state_dict
            
            elif format_type == 'huggingface':
                model = AutoModel.from_pretrained(path, **kwargs)
                if device != 'cpu':
                    model = model.to(device)
            
            elif format_type == 'usbabc':
                if USBABC_AVAILABLE:
                    config = USBABCConfig.from_pretrained(path)
                    model = USBABCForCausalLM.from_pretrained(path, config=config)
                    if device != 'cpu':
                        model = model.to(device)
                else:
                    raise ModelLoadError("USBABC não disponível")
            
            elif format_type == 'gguf':
                # GGUF é normalmente carregado por outros frameworks
                model = {'reader': gguf.GGUFReader(str(path)), 'path': path}
            
            # Adicionar ao cache
            if use_cache and model is not None:
                self._manage_cache(cache_key, model)
            
            logger.info(f"✅ Modelo carregado com sucesso!")
            return model
        
        except Exception as e:
            raise ModelLoadError(f"Erro ao carregar modelo: {str(e)}")
    
    def _manage_cache(self, key: str, model: Any):
        """Gerencia cache de modelos (LRU simples)"""
        if len(self._model_cache) >= self._max_cache_size:
            # Remove o item mais antigo
            oldest_key = next(iter(self._model_cache))
            del self._model_cache[oldest_key]
            logger.info(f"🗑️ Modelo removido do cache: {oldest_key}")
        
        self._model_cache[key] = model
    
    def clear_cache(self):
        """Limpa cache de modelos - NOVO"""
        self._model_cache.clear()
        logger.info("🧹 Cache de modelos limpo")
    
    def get_model_info(self, path: Union[str, Path], use_cache: bool = True) -> Dict[str, Any]:
        """Obtém informações detalhadas do modelo - VERSÃO MELHORADA"""
        path = Path(path)
        cache_key = str(path.absolute())
        
        # Verificar cache de metadados
        if use_cache and cache_key in self.metadata_cache:
            cached = self.metadata_cache[cache_key]
            # Verificar se arquivo foi modificado
            if path.exists():
                current_mtime = path.stat().st_mtime
                if cached.get('_mtime') == current_mtime:
                    return cached
        
        format_type = self.detect_format(path)
        
        info = {
            'path': str(path),
            'name': path.name if path.is_file() else path.name,
            'format': format_type,
            'size_mb': 0,
            'parameters': 0,
            'architecture': 'unknown',
            'created_at': None,
            'modified_at': None,
            'is_valid': False,
            'validation_message': ''
        }
        
        try:
            # Validar modelo
            is_valid, validation_msg = self.validate_model(path)
            info['is_valid'] = is_valid
            info['validation_message'] = validation_msg
            
            if path.is_file():
                stat = path.stat()
                info['size_mb'] = round(stat.st_size / (1024 * 1024), 2)
                info['created_at'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
                info['modified_at'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                info['_mtime'] = stat.st_mtime
            
            elif path.is_dir():
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                info['size_mb'] = round(total_size / (1024 * 1024), 2)
                stat = path.stat()
                info['created_at'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
                info['modified_at'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                info['_mtime'] = stat.st_mtime
            
            # Informações específicas por formato (com tratamento de erro individual)
            try:
                if format_type == 'gguf' and GGUF_AVAILABLE and is_valid:
                    info.update(self._get_gguf_info(path))
            except Exception as e:
                logger.warning(f"Erro ao obter info GGUF: {e}")
            
            try:
                if format_type == 'huggingface' and TRANSFORMERS_AVAILABLE and is_valid:
                    info.update(self._get_huggingface_info(path))
            except Exception as e:
                logger.warning(f"Erro ao obter info HuggingFace: {e}")
            
            try:
                if format_type == 'usbabc' and is_valid:
                    info.update(self._get_usbabc_info(path))
            except Exception as e:
                logger.warning(f"Erro ao obter info USBABC: {e}")
            
            try:
                if format_type == 'pytorch' and is_valid:
                    info.update(self._get_pytorch_info(path))
            except Exception as e:
                logger.warning(f"Erro ao obter info PyTorch: {e}")
            
            try:
                if format_type == 'safetensors' and SAFETENSORS_AVAILABLE and is_valid:
                    info.update(self._get_safetensors_info(path))
            except Exception as e:
                logger.warning(f"Erro ao obter info SafeTensors: {e}")
            
            # Salvar no cache
            if use_cache:
                self.metadata_cache[cache_key] = info
                self._save_metadata_cache()
        
        except Exception as e:
            logger.error(f"Erro ao obter info de {path}: {e}")
            info['error'] = str(e)
        
        return info
    
    def _get_gguf_info(self, path: Path) -> Dict[str, Any]:
        """Informações específicas de GGUF - MELHORADO"""
        try:
            reader = gguf.GGUFReader(str(path))
            
            info = {
                'architecture': 'gguf',
                'quantization': 'unknown',
                'tensor_count': 0
            }
            
            # Extrair metadados
            if hasattr(reader, 'fields'):
                for field_name, field in reader.fields.items():
                    field_name_lower = field_name.lower()
                    
                    if 'general.architecture' in field_name_lower:
                        info['architecture'] = str(field.parts[0]) if field.parts else 'unknown'
                    elif 'general.parameter_count' in field_name_lower:
                        info['parameters'] = int(field.parts[0]) if field.parts else 0
                    elif 'general.quantization_version' in field_name_lower:
                        info['quantization'] = str(field.parts[0]) if field.parts else 'unknown'
            
            # Contar tensores
            if hasattr(reader, 'tensors'):
                info['tensor_count'] = len(reader.tensors)
            
            return info
        
        except Exception as e:
            return {'error': f"Erro GGUF: {e}"}
    
    def _get_huggingface_info(self, path: Path) -> Dict[str, Any]:
        """Informações específicas de HuggingFace - MELHORADO"""
        try:
            config_path = path / 'config.json'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                info = {
                    'architecture': config.get('model_type', config.get('architectures', ['unknown'])[0] if 'architectures' in config else 'unknown'),
                    'vocab_size': config.get('vocab_size', 0),
                    'hidden_size': config.get('hidden_size', 0),
                    'num_layers': config.get('num_hidden_layers', config.get('n_layers', 0)),
                    'num_heads': config.get('num_attention_heads', config.get('n_heads', 0))
                }
                
                # Calcular parâmetros aproximados se não estiver disponível
                if 'num_parameters' in config:
                    info['parameters'] = config['num_parameters']
                elif info['hidden_size'] and info['num_layers']:
                    # Estimativa aproximada
                    info['parameters'] = info['hidden_size'] * info['num_layers'] * 12
                
                return info
        
        except Exception as e:
            return {'error': f"Erro HuggingFace: {e}"}
        
        return {}
    
    def _get_usbabc_info(self, path: Path) -> Dict[str, Any]:
        """Informações específicas de USBABC"""
        try:
            info_path = path / 'model_info.json'
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Adicionar prefixo para evitar conflitos
                    return {f'usbabc_{k}': v for k, v in data.items()}
        
        except Exception as e:
            return {'error': f"Erro USBABC: {e}"}
        
        return {}
    
    def _get_pytorch_info(self, path: Path) -> Dict[str, Any]:
        """Informações específicas de PyTorch - MELHORADO"""
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            info = {'architecture': 'pytorch'}
            
            if isinstance(checkpoint, dict):
                # Detectar estrutura do checkpoint
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    if 'optimizer_state_dict' in checkpoint:
                        info['has_optimizer'] = True
                    if 'epoch' in checkpoint:
                        info['epoch'] = checkpoint['epoch']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Contar parâmetros
                total_params = 0
                trainable_params = 0
                
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        params = value.numel()
                        total_params += params
                        if value.requires_grad:
                            trainable_params += params
                
                info['parameters'] = total_params
                info['trainable_parameters'] = trainable_params
                
                # Detectar arquitetura pelos nomes das camadas
                keys = list(state_dict.keys())
                if any('transformer' in k.lower() for k in keys):
                    info['architecture'] = 'transformer'
                elif any('bert' in k.lower() for k in keys):
                    info['architecture'] = 'bert'
                elif any('gpt' in k.lower() for k in keys):
                    info['architecture'] = 'gpt'
                elif any('llama' in k.lower() for k in keys):
                    info['architecture'] = 'llama'
            
            return info
        
        except Exception as e:
            return {'error': f"Erro PyTorch: {e}"}
    
    def _get_safetensors_info(self, path: Path) -> Dict[str, Any]:
        """Informações específicas de SafeTensors - MELHORADO"""
        try:
            info = {'architecture': 'safetensors', 'tensor_count': 0}
            
            with safe_open(path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                info['tensor_count'] = len(keys)
                
                total_params = 0
                for key in keys:
                    tensor = f.get_tensor(key)
                    total_params += tensor.numel()
                
                info['parameters'] = total_params
                
                # Detectar arquitetura
                if any('transformer' in k.lower() for k in keys):
                    info['architecture'] = 'transformer'
                elif any('bert' in k.lower() for k in keys):
                    info['architecture'] = 'bert'
                elif any('gpt' in k.lower() for k in keys):
                    info['architecture'] = 'gpt'
            
            return info
        
        except Exception as e:
            return {'error': f"Erro SafeTensors: {e}"}
    
    def list_models(self, format_filter: Optional[str] = None, 
                   valid_only: bool = False) -> List[Dict[str, Any]]:
        """Lista todos os modelos disponíveis - MELHORADO"""
        models = []
        seen_paths = set()
        
        # Buscar em models
        for model_path in self.dirs['models'].rglob('*'):
            if model_path in seen_paths:
                continue
            
            # Evitar duplicatas e diretórios vazios
            if model_path.is_file():
                if model_path.suffix in ['.json', '.txt', '.md']:
                    continue
                seen_paths.add(model_path)
            elif model_path.is_dir():
                if not any(model_path.glob('*')):
                    continue
                seen_paths.add(model_path)
            else:
                continue
            
            try:
                format_type = self.detect_format(model_path)
                
                if format_filter and format_type != format_filter:
                    continue
                
                info = self.get_model_info(model_path)
                
                if valid_only and not info.get('is_valid', False):
                    continue
                
                models.append(info)
            
            except Exception as e:
                logger.warning(f"Erro ao processar {model_path}: {e}")
        
        return sorted(models, key=lambda x: x.get('modified_at', ''), reverse=True)
    
    def organize_model(self, source_path: Union[str, Path], 
                      name: Optional[str] = None,
                      create_backup: bool = True) -> Dict[str, Any]:
        """Organiza um modelo no sistema de arquivos - MELHORADO"""
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {source_path}")
        
        # Validar antes de organizar
        is_valid, message = self.validate_model(source_path)
        if not is_valid:
            logger.warning(f"⚠️ Modelo pode ter problemas: {message}")
        
        format_type = self.detect_format(source_path)
        
        if not name:
            name = source_path.stem
        
        # Criar diretório organizado
        target_dir = self.dirs['models'] / name
        target_dir.mkdir(exist_ok=True, parents=True)
        
        # Backup se já existir
        if create_backup and target_dir.exists() and any(target_dir.glob('*')):
            logger.info(f"📦 Criando backup do modelo existente...")
            self.backup_model(target_dir)
        
        if source_path.is_file():
            # Copiar arquivo
            target_path = target_dir / source_path.name
            shutil.copy2(source_path, target_path)
        else:
            # Copiar diretório
            target_path = target_dir
            if target_path.exists() and any(target_path.glob('*')):
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        
        # Criar metadados
        info = self.get_model_info(target_path)
        info['organized_at'] = datetime.now().isoformat()
        info['original_path'] = str(source_path)
        
        metadata_path = target_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Modelo organizado: {target_path}")
        return info
    
    def backup_model(self, model_path: Union[str, Path]) -> Path:
        """Cria backup de um modelo"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{model_path.stem}_{timestamp}"
        backup_path = self.dirs['backups'] / backup_name
        
        if model_path.is_file():
            backup_path.mkdir(exist_ok=True, parents=True)
            shutil.copy2(model_path, backup_path / model_path.name)
        else:
            shutil.copytree(model_path, backup_path)
        
        # Criar info do backup
        backup_info = {
            'original_path': str(model_path),
            'backup_path': str(backup_path),
            'created_at': datetime.now().isoformat(),
            'format': self.detect_format(model_path)
        }
        
        with open(backup_path / 'backup_info.json', 'w', encoding='utf-8') as f:
            json.dump(backup_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Backup criado: {backup_path}")
        return backup_path
    
    def cleanup_temp(self):
        """Limpa arquivos temporários"""
        if self.dirs['temp'].exists():
            shutil.rmtree(self.dirs['temp'])
            self.dirs['temp'].mkdir(parents=True)
        logger.info("🧹 Arquivos temporários limpos")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Estatísticas de armazenamento"""
        stats = {}
        
        for name, dir_path in self.dirs.items():
            if dir_path.exists():
                total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                file_count = len([f for f in dir_path.rglob('*') if f.is_file()])
                
                stats[name] = {
                    'size_mb': round(total_size / (1024 * 1024), 2),
                    'size_gb': round(total_size / (1024 * 1024 * 1024), 2),
                    'file_count': file_count,
                    'path': str(dir_path)
                }
        
        # Total geral
        total_size_mb = sum(s['size_mb'] for s in stats.values())
        stats['total'] = {
            'size_mb': round(total_size_mb, 2),
            'size_gb': round(total_size_mb / 1024, 2),
            'file_count': sum(s['file_count'] for s in stats.values())
        }
        
        return stats
    
    @contextmanager
    def temp_model_path(self, model_name: str):
        """Context manager para paths temporários"""
        temp_path = self.dirs['temp'] / model_name
        temp_path.mkdir(exist_ok=True, parents=True)
        try:
            yield temp_path
        finally:
            if temp_path.exists():
                shutil.rmtree(temp_path)
    
    def search_models(self, query: str, 
                     format_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Busca modelos por nome ou metadados - NOVO"""
        query_lower = query.lower()
        models = self.list_models(format_filter=format_filter)
        
        results = []
        for model in models:
            # Buscar em nome, path e arquitetura
            searchable = f"{model.get('name', '')} {model.get('path', '')} {model.get('architecture', '')}".lower()
            if query_lower in searchable:
                results.append(model)
        
        return results
    
    def get_model_health(self) -> Dict[str, Any]:
        """Verifica saúde geral dos modelos - NOVO"""
        models = self.list_models()
        
        health = {
            'total_models': len(models),
            'valid_models': sum(1 for m in models if m.get('is_valid', False)),
            'invalid_models': sum(1 for m in models if not m.get('is_valid', False)),
            'formats': {},
            'total_size_gb': sum(m.get('size_mb', 0) for m in models) / 1024,
            'issues': []
        }
        
        # Contar por formato
        for model in models:
            fmt = model.get('format', 'unknown')
            if fmt not in health['formats']:
                health['formats'][fmt] = 0
            health['formats'][fmt] += 1
        
        # Identificar problemas
        for model in models:
            if not model.get('is_valid', False):
                health['issues'].append({
                    'model': model.get('name'),
                    'path': model.get('path'),
                    'issue': model.get('validation_message', 'Unknown error')
                })
        
        return health
    
    def export_model_list(self, output_path: Optional[str] = None) -> str:
        """Exporta lista de modelos para JSON - NOVO"""
        models = self.list_models()
        
        if not output_path:
            output_path = self.dirs['exports'] / f'model_list_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'total_models': len(models),
            'models': models
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Lista exportada: {output_path}")
        return str(output_path)


# Instância global
file_manager = UniversalFileManager()


def get_file_manager() -> UniversalFileManager:
    """Obtém a instância do gerenciador de arquivos"""
    return file_manager


# Funções de conveniência para importação direta
def load_model(path: str, **kwargs):
    """Carrega um modelo (função de conveniência)"""
    return file_manager.load_model(path, **kwargs)


def list_models(format_filter: Optional[str] = None, valid_only: bool = False):
    """Lista modelos (função de conveniência)"""
    return file_manager.list_models(format_filter=format_filter, valid_only=valid_only)


def validate_model(path: str) -> Tuple[bool, str]:
    """Valida um modelo (função de conveniência)"""
    return file_manager.validate_model(path)


if __name__ == "__main__":
    # Teste do sistema
    import sys
    
    fm = UniversalFileManager()
    
    print("🗂️  Sistema Universal de Arquivos - VERSÃO APRIMORADA")
    print("=" * 60)
    
    # Estatísticas
    print("\n📊 ESTATÍSTICAS DE ARMAZENAMENTO")
    stats = fm.get_storage_stats()
    for name, info in stats.items():
        if name != 'total':
            print(f"  📁 {name:.<20} {info['size_mb']:>8.2f} MB ({info['file_count']} arquivos)")
    
    print(f"  {'─' * 58}")
    print(f"  📦 TOTAL{' ' * 14} {stats['total']['size_mb']:>8.2f} MB ({stats['total']['file_count']} arquivos)")
    
    # Listar modelos
    print(f"\n🔍 MODELOS ENCONTRADOS")
    models = fm.list_models()
    print(f"  Total: {len(models)} modelos")
    
    if models:
        print(f"\n  {'Nome':<30} {'Formato':<15} {'Tamanho':<12} {'Status'}")
        print(f"  {'-' * 75}")
        
        for model in models[:10]:  # Mostrar apenas os 10 primeiros
            name = Path(model['path']).name[:28]
            fmt = model['format']
            size = f"{model['size_mb']} MB"
            status = "✅" if model.get('is_valid', False) else "❌"
            
            print(f"  {name:<30} {fmt:<15} {size:<12} {status}")
        
        if len(models) > 10:
            print(f"\n  ... e mais {len(models) - 10} modelos")
    
    # Saúde dos modelos
    print(f"\n🏥 SAÚDE DOS MODELOS")
    health = fm.get_model_health()
    print(f"  ✅ Válidos: {health['valid_models']}")
    print(f"  ❌ Inválidos: {health['invalid_models']}")
    print(f"  📊 Total: {health['total_models']}")
    
    if health['formats']:
        print(f"\n  📋 Por formato:")
        for fmt, count in health['formats'].items():
            print(f"    • {fmt}: {count}")
    
    if health['issues']:
        print(f"\n  ⚠️  Problemas encontrados:")
        for issue in health['issues'][:5]:
            print(f"    • {issue['model']}: {issue['issue']}")
    
    print("\n" + "=" * 60)
    print("✅ Sistema pronto para uso!")
    print("\nComandos disponíveis:")
    print("  fm.load_model(path)           - Carregar modelo")
    print("  fm.list_models()              - Listar modelos")
    print("  fm.validate_model(path)       - Validar modelo")
    print("  fm.search_models(query)       - Buscar modelos")
    print("  fm.get_model_health()         - Verificar saúde")
    print("  fm.organize_model(path, name) - Organizar modelo")
    print("  fm.backup_model(path)         - Criar backup")