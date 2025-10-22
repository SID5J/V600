#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Rotação de APIs - ARQV30 Enhanced v3.0
Gerenciador robusto com fallback automático e monitoramento
"""

import os
import time
import logging
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class APIStatus(Enum):
    """Status de uma API"""
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


@dataclass
class APIKey:
    """Representação de uma chave de API"""
    key: str
    provider: str
    name: str
    status: APIStatus = APIStatus.UNKNOWN
    last_used: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0
    rate_limit_reset: Optional[datetime] = None
    daily_usage: int = 0
    monthly_usage: int = 0
    
    def __post_init__(self):
        # Mascarar a chave para logs
        self.masked_key = self._mask_key(self.key)
    
    def _mask_key(self, key: str) -> str:
        """Mascara a chave para logs seguros"""
        if len(key) <= 8:
            return "*" * len(key)
        return key[:4] + "*" * (len(key) - 8) + key[-4:]
    
    def is_available(self) -> bool:
        """Verifica se a API está disponível para uso"""
        if self.status == APIStatus.DISABLED:
            return False
        
        if self.status == APIStatus.RATE_LIMITED:
            if self.rate_limit_reset and datetime.now() < self.rate_limit_reset:
                return False
        
        # Limite de erros consecutivos
        if self.error_count >= 5:
            return False
        
        return True
    
    def record_success(self):
        """Registra uso bem-sucedido"""
        self.last_used = datetime.now()
        self.success_count += 1
        self.daily_usage += 1
        self.monthly_usage += 1
        self.error_count = 0  # Reset contador de erros
        self.status = APIStatus.ACTIVE
    
    def record_error(self, error_type: str = "general"):
        """Registra erro de uso"""
        self.error_count += 1
        self.last_used = datetime.now()
        
        if error_type == "rate_limit":
            self.status = APIStatus.RATE_LIMITED
            # Rate limit por 1 hora por padrão
            self.rate_limit_reset = datetime.now() + timedelta(hours=1)
        else:
            self.status = APIStatus.ERROR


class APIRotationManager:
    """Gerenciador de rotação de APIs com fallback automático"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa o gerenciador de rotação
        
        Args:
            config_file: Arquivo de configuração opcional
        """
        self.apis: Dict[str, List[APIKey]] = {}
        self.current_indices: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.stats_file = "logs/api_stats.json"
        self.config = self._load_config()
        
        # Configurações
        self.rotation_enabled = os.getenv('API_ROTATION_ENABLED', 'true').lower() == 'true'
        self.max_retries = int(os.getenv('MAX_API_RETRIES', '3'))
        self.request_timeout = int(os.getenv('API_REQUEST_TIMEOUT', '30'))
        self.rotation_interval = int(os.getenv('API_ROTATION_INTERVAL', '0'))
        
        # Carregar APIs do ambiente
        self._load_apis_from_env()
        
        # Validar chaves se configurado
        if os.getenv('VALIDATE_KEYS_ON_STARTUP', 'true').lower() == 'true':
            self._validate_all_keys()
        
        # Carregar estatísticas
        self._load_stats()
        
        logger.info(f"✅ APIRotationManager inicializado")
        logger.info(f"   - Rotação ativa: {self.rotation_enabled}")
        logger.info(f"   - Provedores: {list(self.apis.keys())}")
        logger.info(f"   - Total de chaves: {sum(len(keys) for keys in self.apis.values())}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega configurações"""
        return {
            'llm_providers': ['gemini', 'openrouter', 'openai'],
            'search_providers': ['serper', 'jina', 'google_search', 'bing'],
            'fallback_order': {
                'llm': ['gemini', 'openrouter', 'openai'],
                'search': ['serper', 'jina', 'google_search', 'bing']
            }
        }
    
    def _load_apis_from_env(self):
        """Carrega chaves de API das variáveis de ambiente"""
        # Mapeamento de provedores
        providers = {
            'gemini': 'GEMINI_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY', 
            'openai': 'OPENAI_API_KEY',
            'serper': 'SERPER_API_KEY',
            'jina': 'JINA_API_KEY',
            'google_search': 'GOOGLE_SEARCH_API_KEY',
            'bing': 'BING_SEARCH_API_KEY'
        }
        
        for provider, base_key in providers.items():
            keys = []
            
            # Chave principal
            main_key = os.getenv(base_key)
            if main_key and main_key != f"your_{provider}_api_key_here":
                keys.append(APIKey(
                    key=main_key,
                    provider=provider,
                    name=f"{provider}_main"
                ))
            
            # Chaves backup (2, 3, etc.)
            for i in range(2, 6):  # Até 5 chaves por provedor
                backup_key = os.getenv(f"{base_key}_{i}")
                if backup_key and backup_key != f"your_backup_{provider}_key_here":
                    keys.append(APIKey(
                        key=backup_key,
                        provider=provider,
                        name=f"{provider}_backup_{i}"
                    ))
            
            if keys:
                self.apis[provider] = keys
                self.current_indices[provider] = 0
                logger.info(f"📋 {provider}: {len(keys)} chave(s) carregada(s)")
    
    def _validate_all_keys(self):
        """Valida todas as chaves carregadas"""
        logger.info("🔍 Validando chaves de API...")
        
        for provider, keys in self.apis.items():
            for api_key in keys:
                try:
                    # Validação básica (formato)
                    if self._validate_key_format(api_key):
                        api_key.status = APIStatus.ACTIVE
                        logger.info(f"✅ {api_key.name}: Formato válido")
                    else:
                        api_key.status = APIStatus.ERROR
                        logger.warning(f"⚠️ {api_key.name}: Formato inválido")
                        
                except Exception as e:
                    api_key.status = APIStatus.ERROR
                    logger.error(f"❌ {api_key.name}: Erro na validação - {e}")
    
    def _validate_key_format(self, api_key: APIKey) -> bool:
        """Valida formato básico da chave"""
        key = api_key.key
        provider = api_key.provider
        
        # Validações básicas por provedor
        validations = {
            'gemini': lambda k: k.startswith('AIza') and len(k) > 30,
            'openai': lambda k: k.startswith('sk-') and len(k) > 40,
            'openrouter': lambda k: k.startswith('sk-or-') and len(k) > 50,
            'serper': lambda k: len(k) == 40 and k.isalnum(),
            'jina': lambda k: k.startswith('jina_') and len(k) > 40,
            'google_search': lambda k: k.startswith('AIza') and len(k) > 30,
            'bing': lambda k: len(k) == 32 and k.isalnum()
        }
        
        validator = validations.get(provider)
        if validator:
            return validator(key)
        
        # Validação genérica
        return len(key) > 10 and key.strip() != ""
    
    def get_api_key(self, provider: str, force_rotation: bool = False) -> Optional[APIKey]:
        """
        Obtém uma chave de API disponível para o provedor
        
        Args:
            provider: Nome do provedor
            force_rotation: Forçar rotação para próxima chave
            
        Returns:
            APIKey ou None se nenhuma disponível
        """
        with self.lock:
            if provider not in self.apis:
                logger.warning(f"⚠️ Provedor não encontrado: {provider}")
                return None
            
            keys = self.apis[provider]
            if not keys:
                logger.warning(f"⚠️ Nenhuma chave disponível para: {provider}")
                return None
            
            # Rotação forçada ou automática
            if force_rotation or (self.rotation_enabled and self.rotation_interval == 0):
                self.current_indices[provider] = (self.current_indices[provider] + 1) % len(keys)
            
            # Tentar encontrar chave disponível
            start_index = self.current_indices[provider]
            attempts = 0
            
            while attempts < len(keys):
                current_key = keys[self.current_indices[provider]]
                
                if current_key.is_available():
                    logger.debug(f"🔑 Usando {current_key.name} ({current_key.masked_key})")
                    return current_key
                
                # Próxima chave
                self.current_indices[provider] = (self.current_indices[provider] + 1) % len(keys)
                attempts += 1
            
            logger.error(f"❌ Nenhuma chave disponível para {provider}")
            return None
    
    def record_api_usage(self, api_key: APIKey, success: bool, error_type: str = None, response_time: float = None):
        """
        Registra uso de uma API
        
        Args:
            api_key: Chave utilizada
            success: Se a requisição foi bem-sucedida
            error_type: Tipo de erro se houver
            response_time: Tempo de resposta em segundos
        """
        with self.lock:
            if success:
                api_key.record_success()
                logger.debug(f"✅ {api_key.name}: Sucesso (tempo: {response_time:.2f}s)")
            else:
                api_key.record_error(error_type)
                logger.warning(f"❌ {api_key.name}: Erro ({error_type})")
            
            # Salvar estatísticas
            self._save_stats()
    
    def get_fallback_providers(self, service_type: str) -> List[str]:
        """
        Obtém lista de provedores em ordem de fallback
        
        Args:
            service_type: 'llm' ou 'search'
            
        Returns:
            Lista de provedores ordenada
        """
        fallback_order = self.config.get('fallback_order', {})
        return fallback_order.get(service_type, [])
    
    def execute_with_fallback(
        self,
        service_type: str,
        operation: Callable[[APIKey], Any],
        max_attempts: Optional[int] = None
    ) -> Tuple[Any, Optional[APIKey]]:
        """
        Executa operação com fallback automático entre provedores
        
        Args:
            service_type: Tipo de serviço ('llm' ou 'search')
            operation: Função que recebe APIKey e retorna resultado
            max_attempts: Máximo de tentativas (padrão: todos os provedores)
            
        Returns:
            Tuple (resultado, api_key_usada) ou (None, None) se falhar
        """
        providers = self.get_fallback_providers(service_type)
        max_attempts = max_attempts or len(providers) * self.max_retries
        
        attempt = 0
        for provider in providers:
            if attempt >= max_attempts:
                break
                
            # Tentar com todas as chaves do provedor
            for retry in range(self.max_retries):
                if attempt >= max_attempts:
                    break
                    
                api_key = self.get_api_key(provider, force_rotation=(retry > 0))
                if not api_key:
                    break
                
                try:
                    logger.info(f"🔄 Tentativa {attempt + 1}: {api_key.name}")
                    start_time = time.time()
                    
                    result = operation(api_key)
                    
                    response_time = time.time() - start_time
                    self.record_api_usage(api_key, True, response_time=response_time)
                    
                    logger.info(f"✅ Sucesso com {api_key.name}")
                    return result, api_key
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    error_type = self._classify_error(str(e))
                    
                    self.record_api_usage(api_key, False, error_type, response_time)
                    
                    logger.warning(f"❌ Erro com {api_key.name}: {e}")
                    
                    # Se for rate limit, tentar próximo provedor imediatamente
                    if error_type == "rate_limit":
                        break
                    
                    attempt += 1
                    
                    # Pequena pausa entre tentativas
                    if retry < self.max_retries - 1:
                        time.sleep(1)
        
        logger.error(f"❌ Todas as tentativas falharam para {service_type}")
        return None, None
    
    def _classify_error(self, error_message: str) -> str:
        """Classifica tipo de erro baseado na mensagem"""
        error_lower = error_message.lower()
        
        if any(term in error_lower for term in ['rate limit', 'quota', 'too many requests', '429']):
            return "rate_limit"
        elif any(term in error_lower for term in ['unauthorized', '401', 'invalid key', 'api key']):
            return "auth_error"
        elif any(term in error_lower for term in ['timeout', 'connection', 'network']):
            return "network_error"
        else:
            return "general_error"
    
    def get_provider_stats(self, provider: str) -> Dict[str, Any]:
        """Obtém estatísticas de um provedor"""
        if provider not in self.apis:
            return {}
        
        keys = self.apis[provider]
        total_success = sum(key.success_count for key in keys)
        total_errors = sum(key.error_count for key in keys)
        active_keys = sum(1 for key in keys if key.is_available())
        
        return {
            'provider': provider,
            'total_keys': len(keys),
            'active_keys': active_keys,
            'total_success': total_success,
            'total_errors': total_errors,
            'success_rate': total_success / (total_success + total_errors) if (total_success + total_errors) > 0 else 0,
            'keys': [
                {
                    'name': key.name,
                    'masked_key': key.masked_key,
                    'status': key.status.value,
                    'success_count': key.success_count,
                    'error_count': key.error_count,
                    'last_used': key.last_used.isoformat() if key.last_used else None,
                    'is_available': key.is_available()
                }
                for key in keys
            ]
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas completas"""
        stats = {
            'rotation_enabled': self.rotation_enabled,
            'total_providers': len(self.apis),
            'total_keys': sum(len(keys) for keys in self.apis.values()),
            'providers': {}
        }
        
        for provider in self.apis.keys():
            stats['providers'][provider] = self.get_provider_stats(provider)
        
        return stats
    
    def _save_stats(self):
        """Salva estatísticas em arquivo"""
        try:
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            
            stats = self.get_all_stats()
            stats['last_updated'] = datetime.now().isoformat()
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erro ao salvar estatísticas: {e}")
    
    def _load_stats(self):
        """Carrega estatísticas salvas"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                # Restaurar contadores se possível
                for provider_name, provider_stats in stats.get('providers', {}).items():
                    if provider_name in self.apis:
                        for key_stat in provider_stats.get('keys', []):
                            # Encontrar chave correspondente
                            for api_key in self.apis[provider_name]:
                                if api_key.name == key_stat['name']:
                                    api_key.success_count = key_stat.get('success_count', 0)
                                    api_key.error_count = key_stat.get('error_count', 0)
                                    break
                
                logger.info("📊 Estatísticas carregadas")
                
        except Exception as e:
            logger.warning(f"Aviso ao carregar estatísticas: {e}")
    
    def reset_provider_errors(self, provider: str):
        """Reseta contadores de erro de um provedor"""
        if provider in self.apis:
            for api_key in self.apis[provider]:
                api_key.error_count = 0
                if api_key.status == APIStatus.ERROR:
                    api_key.status = APIStatus.UNKNOWN
            
            logger.info(f"🔄 Erros resetados para {provider}")
    
    def disable_provider(self, provider: str):
        """Desabilita um provedor"""
        if provider in self.apis:
            for api_key in self.apis[provider]:
                api_key.status = APIStatus.DISABLED
            
            logger.info(f"🚫 Provedor desabilitado: {provider}")
    
    def enable_provider(self, provider: str):
        """Habilita um provedor"""
        if provider in self.apis:
            for api_key in self.apis[provider]:
                if api_key.status == APIStatus.DISABLED:
                    api_key.status = APIStatus.UNKNOWN
            
            logger.info(f"✅ Provedor habilitado: {provider}")


# Instância global do gerenciador
_api_manager = None


def get_api_manager() -> APIRotationManager:
    """Obtém instância global do gerenciador de APIs"""
    global _api_manager
    if _api_manager is None:
        _api_manager = APIRotationManager()
    return _api_manager


def get_llm_api_key(provider: str = None) -> Optional[APIKey]:
    """Obtém chave de API para LLM"""
    manager = get_api_manager()
    
    if provider:
        return manager.get_api_key(provider)
    
    # Usar ordem de fallback
    for prov in manager.get_fallback_providers('llm'):
        key = manager.get_api_key(prov)
        if key:
            return key
    
    return None


def get_search_api_key(provider: str = None) -> Optional[APIKey]:
    """Obtém chave de API para busca"""
    manager = get_api_manager()
    
    if provider:
        return manager.get_api_key(provider)
    
    # Usar ordem de fallback
    for prov in manager.get_fallback_providers('search'):
        key = manager.get_api_key(prov)
        if key:
            return key
    
    return None


if __name__ == "__main__":
    # Teste do sistema
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testando APIRotationManager...")
    
    manager = APIRotationManager()
    
    # Mostrar estatísticas
    stats = manager.get_all_stats()
    print(f"\n📊 Estatísticas:")
    print(f"   - Provedores: {stats['total_providers']}")
    print(f"   - Chaves totais: {stats['total_keys']}")
    print(f"   - Rotação ativa: {stats['rotation_enabled']}")
    
    # Testar obtenção de chaves
    for service_type in ['llm', 'search']:
        print(f"\n🔑 Testando {service_type}:")
        providers = manager.get_fallback_providers(service_type)
        for provider in providers:
            key = manager.get_api_key(provider)
            if key:
                print(f"   ✅ {provider}: {key.masked_key}")
            else:
                print(f"   ❌ {provider}: Não disponível")
    
    print("\n✅ Teste concluído!")