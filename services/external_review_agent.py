#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v3.0 - External Review Agent
Agente principal de revisão externa - ponto de entrada do módulo
"""

import logging
import os
import yaml
import json
import glob
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import asyncio

# Imports integrados ao sistema principal
from .sentiment_analyzer import ExternalSentimentAnalyzer
from .bias_disinformation_detector import ExternalBiasDisinformationDetector
from .llm_reasoning_service import ExternalLLMReasoningService
from .rule_engine import ExternalRuleEngine
from .contextual_analyzer import ExternalContextualAnalyzer
from .confidence_thresholds import ExternalConfidenceThresholds

logger = logging.getLogger(__name__)

class ExternalReviewAgent:
    """Agente de revisão externa - orquestrador principal do módulo"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o agente de revisão externa

        Args:
            config_path (Optional[str]): Caminho para arquivo de configuração
        """
        # Inicializar logger da instância
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.config = self._load_config(config_path)

        # Initialize all analysis services
        self.sentiment_analyzer = ExternalSentimentAnalyzer(self.config)
        self.bias_detector = ExternalBiasDisinformationDetector(self.config)
        self.llm_service = ExternalLLMReasoningService(self.config)
        self.rule_engine = ExternalRuleEngine(self.config)
        self.contextual_analyzer = ExternalContextualAnalyzer(self.config)
        self.confidence_thresholds = ExternalConfidenceThresholds(self.config)

        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'approved': 0,
            'rejected': 0,
            'start_time': datetime.now(),
            'processing_times': []
        }

        logger.info(f"✅ External Review Agent inicializado com sucesso")
        logger.info(f"🔧 Configurações carregadas: {len(self.config)} seções")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Carrega configuração do módulo"""
        try:
            # Default config path
            if config_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(current_dir, '..', 'config', 'default_config.yaml')

            # Load configuration file
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                logger.info(f"✅ Configuração carregada: {config_path}")
                return config
            else:
                logger.warning(f"⚠️ Arquivo de configuração não encontrado: {config_path}")
                return self._get_default_config()

        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Retorna configuração padrão"""
        return {
            'thresholds': {
                'approval': 0.75,
                'rejection': 0.35,
                'high_confidence': 0.85,
                'low_confidence': 0.5,
                'bias_high_risk': 0.7
            },
            'sentiment_analysis': {'enabled': True},
            'bias_detection': {'enabled': True},
            'llm_reasoning': {'enabled': True},
            'contextual_analysis': {'enabled': True},
            'rules': []
        }

    def process_item(self, item_data: Dict[str, Any], massive_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processa um item individual através de todas as análises com validação aprimorada

        Args:
            item_data (Dict[str, Any]): Dados do item para análise
            massive_data (Optional[Dict[str, Any]]): Contexto adicional

        Returns:
            Dict[str, Any]: Resultado completo da análise
        """
        start_time = datetime.now()

        try:
            item_id = item_data.get('id', f'item_{self.stats["total_processed"]}')
            logger.info(f"🔍 Iniciando análise do item: {item_id}")

            # Validação prévia do item
            validation_result = self._validate_item_data(item_data)
            if not validation_result['valid']:
                logger.warning(f"⚠️ Item inválido: {validation_result['reason']}")
                return self._create_validation_error_result(item_data, validation_result['reason'])

            # Extract text content for analysis
            text_content = self._extract_text_content(item_data)

            if not text_content or len(text_content.strip()) < 5:
                logger.warning("⚠️ Item com conteúdo textual insuficiente")
                return self._create_insufficient_content_result(item_data)

            # Initialize analysis results
            analysis_result = {
                'item_id': item_data.get('id', f'item_{self.stats["total_processed"]}'),
                'original_item': item_data,
                'processing_timestamp': start_time.isoformat(),
                'text_analyzed': text_content[:500],  # First 500 chars for reference
            }

            # Step 1: Sentiment Analysis
            logger.debug("Executando análise de sentimento...")
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(text_content)
            analysis_result['sentiment_analysis'] = sentiment_result

            # Step 2: Bias & Disinformation Detection
            logger.debug("Executando detecção de viés/desinformação...")
            bias_result = self.bias_detector.detect_bias_disinformation(text_content)
            analysis_result['bias_disinformation_analysis'] = bias_result

            # Step 3: LLM Reasoning (for ambiguous cases)
            should_use_llm = self._should_use_llm_analysis(sentiment_result, bias_result)
            if should_use_llm:
                logger.debug("Executando análise LLM...")
                context = self._create_llm_context(analysis_result, massive_data)
                llm_result = self.llm_service.analyze_with_llm(text_content, context)
                analysis_result['llm_reasoning_analysis'] = llm_result
            else:
                analysis_result['llm_reasoning_analysis'] = {
                    'llm_confidence': 0.5,
                    'llm_recommendation': 'NÃO_EXECUTADO',
                    'analysis_reasoning': 'LLM não necessário para este item'
                }

            # Step 4: Contextual Analysis
            logger.debug("Executando análise contextual...")
            contextual_result = self.contextual_analyzer.analyze_context(item_data, massive_data)
            analysis_result['contextual_analysis'] = contextual_result

            # Step 5: Rule Engine Application
            logger.debug("Aplicando regras de negócio...")
            rule_result = self.rule_engine.apply_rules(analysis_result)
            analysis_result['rule_decision'] = rule_result

            # Step 6: Final Decision
            final_decision = self._make_final_decision(analysis_result)
            analysis_result['ai_review'] = final_decision

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(final_decision['status'], processing_time)

            analysis_result['processing_time_seconds'] = processing_time

            logger.info(f"✅ Item processado: {final_decision['status']} (confiança: {final_decision['final_confidence']:.3f})")

            return analysis_result

        except Exception as e:
            logger.error(f"Erro no processamento do item: {e}")
            error_result = self._create_error_result(item_data, str(e))
            self._update_stats('error', (datetime.now() - start_time).total_seconds())
            return error_result

    def _validate_item_data(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Valida dados do item antes do processamento"""
        if not isinstance(item_data, dict):
            return {'valid': False, 'reason': 'Item deve ser um dicionário'}
        
        if not item_data:
            return {'valid': False, 'reason': 'Item vazio'}
        
        # Verificar se tem pelo menos um campo de conteúdo
        content_fields = ['content', 'text', 'title', 'description', 'summary', 'body']
        has_content = any(field in item_data and item_data[field] for field in content_fields)
        
        if not has_content:
            return {'valid': False, 'reason': 'Item não possui conteúdo textual válido'}
        
        return {'valid': True, 'reason': 'Item válido'}

    def _extract_text_content(self, item_data: Dict[str, Any]) -> str:
        """Extrai conteúdo textual do item com priorização"""
        # Campos priorizados por importância
        priority_fields = ['content', 'text', 'description', 'summary', 'title', 'body']

        text_parts = []
        for field in priority_fields:
            if field in item_data and item_data[field]:
                content = str(item_data[field]).strip()
                if content:
                    text_parts.append(content)

        # Adicionar campos extras se existirem
        extra_fields = ['subtitle', 'excerpt', 'abstract', 'caption']
        for field in extra_fields:
            if field in item_data and item_data[field]:
                content = str(item_data[field]).strip()
                if content:
                    text_parts.append(content)

        return ' '.join(text_parts).strip()

    def _should_use_llm_analysis(self, sentiment_result: Dict[str, Any], bias_result: Dict[str, Any]) -> bool:
        """Determina se deve usar análise LLM"""
        # Use LLM for ambiguous cases or high-risk content
        sentiment_confidence = sentiment_result.get('confidence', 0.5)
        bias_risk = bias_result.get('overall_risk', 0.0)

        # Low confidence sentiment or high bias risk = use LLM
        return sentiment_confidence < 0.6 or bias_risk > 0.4

    def _create_llm_context(self, analysis_result: Dict[str, Any], massive_data: Optional[Dict[str, Any]]) -> str:
        """Cria contexto para análise LLM"""
        context_parts = []

        # Add sentiment context
        sentiment = analysis_result.get('sentiment_analysis', {})
        if sentiment.get('classification') != 'neutral':
            context_parts.append(f"Sentimento detectado: {sentiment.get('classification', 'indefinido')}")

        # Add bias context
        bias = analysis_result.get('bias_disinformation_analysis', {})
        if bias.get('overall_risk', 0) > 0.3:
            context_parts.append(f"Risco de viés detectado: {bias.get('overall_risk', 0):.2f}")

        # Add any available external context
        if massive_data:
            if 'topic' in massive_data:
                context_parts.append(f"Tópico: {massive_data['topic']}")

        return ' | '.join(context_parts) if context_parts else ""

    def _make_final_decision(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Toma decisão final baseada em todas as análises"""
        try:
            # Get analysis results
            sentiment = analysis_result.get('sentiment_analysis', {})
            bias = analysis_result.get('bias_disinformation_analysis', {})
            llm = analysis_result.get('llm_reasoning_analysis', {})
            contextual = analysis_result.get('contextual_analysis', {})
            rule_decision = analysis_result.get('rule_decision', {})

            # Calculate composite confidence
            confidences = [
                sentiment.get('confidence', 0.5) * 0.2,  # 20% weight
                (1.0 - bias.get('overall_risk', 0.5)) * 0.3,  # 30% weight (inverted risk)
                llm.get('llm_confidence', 0.5) * 0.3,  # 30% weight
                contextual.get('contextual_confidence', 0.5) * 0.2  # 20% weight
            ]

            final_confidence = sum(confidences)

            # Apply rule engine decision if applicable
            if rule_decision.get('status') in ['approved', 'rejected']:
                status = rule_decision['status']
                reason = rule_decision['reason']
            else:
                # Use confidence thresholds for decision
                if self.confidence_thresholds.should_approve(final_confidence):
                    status = 'approved'
                    reason = 'Aprovado com base na análise combinada'
                elif self.confidence_thresholds.should_reject(final_confidence):
                    status = 'rejected'
                    reason = 'Rejeitado com base na análise combinada'
                else:
                    # Default to rejection for ambiguous cases (safer)
                    status = 'rejected'
                    reason = 'Rejeitado por ambiguidade - política de segurança'

            # Create comprehensive decision result
            decision = {
                'status': status,
                'reason': reason,
                'final_confidence': final_confidence,
                'confidence_breakdown': {
                    'sentiment_contribution': sentiment.get('confidence', 0.5) * 0.2,
                    'bias_contribution': (1.0 - bias.get('overall_risk', 0.5)) * 0.3,
                    'llm_contribution': llm.get('llm_confidence', 0.5) * 0.3,
                    'contextual_contribution': contextual.get('contextual_confidence', 0.5) * 0.2
                },
                'decision_factors': {
                    'sentiment_classification': sentiment.get('classification', 'neutral'),
                    'bias_risk_level': 'high' if bias.get('overall_risk', 0) > 0.6 else 'medium' if bias.get('overall_risk', 0) > 0.3 else 'low',
                    'llm_recommendation': llm.get('llm_recommendation', 'NÃO_EXECUTADO'),
                    'rule_triggered': rule_decision.get('triggered_rules', [])
                },
                'analysis_summary': {
                    'total_flags': (
                        len(bias.get('detected_bias_keywords', [])) +
                        len(bias.get('detected_disinformation_patterns', [])) +
                        len(contextual.get('context_flags', []))
                    ),
                    'sentiment_polarity': sentiment.get('polarity', 0.0),
                    'overall_risk_score': bias.get('overall_risk', 0.0),
                    'contextual_consistency': contextual.get('consistency_score', 0.5)
                },
                'processing_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '3.0.0',
                    'confidence_threshold_used': self.confidence_thresholds.get_threshold('approval')
                }
            }

            return decision

        except Exception as e:
            logger.error(f"Erro na decisão final: {e}")
            return {
                'status': 'rejected',
                'reason': f'Erro no processamento: {str(e)}',
                'final_confidence': 0.0,
                'error': True
            }

    def _create_validation_error_result(self, item_data: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Cria resultado para item que falhou na validação"""
        return {
            'item_id': item_data.get('id', 'sem_id'),
            'original_item': item_data,
            'processing_timestamp': datetime.now().isoformat(),
            'ai_review': {
                'status': 'rejected',
                'reason': f'Erro de validação: {reason}',
                'final_confidence': 0.0,
                'error': True,
                'validation_error': True
            },
            'sentiment_analysis': {},
            'bias_disinformation_analysis': {},
            'llm_reasoning_analysis': {},
            'contextual_analysis': {},
            'processing_time_seconds': 0.0
        }

    def _create_insufficient_content_result(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cria resultado para item com conteúdo insuficiente"""
        return {
            'item_id': item_data.get('id', 'sem_id'),
            'original_item': item_data,
            'processing_timestamp': datetime.now().isoformat(),
            'ai_review': {
                'status': 'rejected',
                'reason': 'Conteúdo textual insuficiente para análise',
                'final_confidence': 0.0,
                'error': False,
                'insufficient_content': True
            },
            'sentiment_analysis': {},
            'bias_disinformation_analysis': {},
            'llm_reasoning_analysis': {},
            'contextual_analysis': {},
            'processing_time_seconds': 0.0
        }

    def _create_error_result(self, item_data: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Cria resultado para erro de processamento"""
        return {
            'item_id': item_data.get('id', 'sem_id'),
            'original_item': item_data,
            'processing_timestamp': datetime.now().isoformat(),
            'ai_review': {
                'status': 'rejected',
                'reason': f'Erro no processamento: {error_message}',
                'final_confidence': 0.0,
                'error': True
            },
            'error_details': error_message,
            'processing_time_seconds': 0.0
        }

    def _update_stats(self, status: str, processing_time: float):
        """Atualiza estatísticas de processamento"""
        self.stats['total_processed'] += 1
        self.stats['processing_times'].append(processing_time)

        if status == 'approved':
            self.stats['approved'] += 1
        elif status == 'rejected':
            self.stats['rejected'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de processamento"""
        total_time = (datetime.now() - self.stats['start_time']).total_seconds()
        avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0

        return {
            'total_processed': self.stats['total_processed'],
            'approved': self.stats['approved'],
            'rejected': self.stats['rejected'],
            'approval_rate': self.stats['approved'] / max(self.stats['total_processed'], 1),
            'total_runtime_seconds': total_time,
            'average_processing_time_seconds': avg_processing_time,
            'items_per_second': self.stats['total_processed'] / max(total_time, 1)
        }

    def find_consolidacao_file(self, session_id: str) -> Optional[str]:
        """Busca automaticamente o arquivo de consolidação da etapa 1 para a sessão especificada"""
        try:
            # Diretório base onde os arquivos são salvos
            base_paths = [
                f"../src/relatorios_intermediarios/workflow/{session_id}",
                f"src/relatorios_intermediarios/workflow/{session_id}",
                f"relatorios_intermediarios/workflow/{session_id}",
                f"../relatorios_intermediarios/workflow/{session_id}"
            ]

            for base_path in base_paths:
                if os.path.exists(base_path):
                    # Busca por arquivos de consolidação
                    pattern = f"{base_path}/consolidacao_etapa1_final_*.json"
                    files = glob.glob(pattern)

                    if files:
                        # Pega o arquivo mais recente
                        latest_file = max(files, key=os.path.getmtime)
                        self.logger.info(f"✅ Arquivo de consolidação encontrado: {latest_file}")
                        return latest_file

            # Se não encontrou, busca em todo o projeto
            search_patterns = [
                f"**/consolidacao_etapa1_final_*{session_id}*.json",
                f"**/consolidacao_etapa1_final_*.json"
            ]

            for pattern in search_patterns:
                files = glob.glob(pattern, recursive=True)
                if files:
                    # Filtra por sessão se possível
                    session_files = [f for f in files if session_id in f]
                    if session_files:
                        latest_file = max(session_files, key=os.path.getmtime)
                        self.logger.info(f"✅ Arquivo de consolidação encontrado (busca recursiva): {latest_file}")
                        return latest_file
                    else:
                        # Pega o mais recente se não conseguir filtrar por sessão
                        latest_file = max(files, key=os.path.getmtime)
                        self.logger.info(f"⚠️ Usando arquivo mais recente: {latest_file}")
                        return latest_file

            self.logger.warning(f"❌ Nenhum arquivo de consolidação encontrado para sessão {session_id}")
            return None

        except Exception as e:
            self.logger.error(f"❌ Erro ao buscar arquivo de consolidação: {e}")
            return None

    def load_consolidacao_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Carrega dados do arquivo de consolidação da etapa 1"""
        try:
            file_path = self.find_consolidacao_file(session_id)

            if not file_path:
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.info(f"📄 Dados de consolidação carregados: {len(data.get('data', {}).get('dados_web', []))} itens web")
            return data

        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados de consolidação: {e}")
            return None

    def convert_consolidacao_to_analysis_format(self, consolidacao_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Converte dados de consolidação para formato de análise do External AI Verifier"""
        try:
            if not consolidacao_data:
                return {'items': [], 'context': {}}

            data_section = consolidacao_data.get('data', {})
            dados_web = data_section.get('dados_web', [])

            # Converte cada item para o formato esperado
            items = []
            for idx, item in enumerate(dados_web):
                converted_item = {
                    'id': f"web_{idx+1:03d}",
                    'content': item.get('titulo', ''),
                    'title': item.get('titulo', ''),
                    'source': item.get('url', ''),
                    'url': item.get('url', ''),
                    'author': item.get('fonte', 'Desconhecido'),
                    'timestamp': datetime.now().isoformat(),
                    'category': 'web_content',
                    'relevancia': item.get('relevancia', 0.5),
                    'conteudo_tamanho': item.get('conteudo_tamanho', 0),
                    'engagement': item.get('engagement', {}),
                    'metadata': {
                        'session_id': session_id,
                        'fonte_original': item.get('fonte', ''),
                        'tipo_dado': data_section.get('tipo', ''),
                        'processado_em': datetime.now().isoformat()
                    }
                }
                items.append(converted_item)

            # Contextualiza a análise
            context = {
                'topic': data_section.get('tipo', 'analise_dados_web'),
                'analysis_type': 'verificacao_consolidacao_etapa1',
                'session_id': session_id,
                'source_file': 'consolidacao_etapa1',
                'total_items_originais': len(dados_web),
                'processamento_timestamp': datetime.now().isoformat(),
                **self.config.get('context', {})
            }

            self.logger.info(f"✅ Conversão concluída: {len(items)} itens preparados para análise")

            return {
                'items': items,
                'context': context
            }

        except Exception as e:
            self.logger.error(f"❌ Erro ao converter dados de consolidação: {e}")
            return {'items': [], 'context': {}}

    def analyze_session_consolidacao(self, session_id: str) -> Dict[str, Any]:
        """Analisa automaticamente os dados de consolidação de uma sessão"""
        try:
            self.logger.info(f"🔍 Iniciando análise da consolidação para sessão: {session_id}")

            # Carrega dados de consolidação
            consolidacao_data = self.load_consolidacao_data(session_id)

            if not consolidacao_data:
                return {
                    'success': False,
                    'error': f'Arquivo de consolidação não encontrado para sessão {session_id}',
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                }

            # Converte para formato de análise
            analysis_data = self.convert_consolidacao_to_analysis_format(consolidacao_data, session_id)

            if not analysis_data.get('items'):
                return {
                    'success': False,
                    'error': 'Nenhum item válido encontrado nos dados de consolidação',
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                }

            # Executa análise
            result = self.analyze_content_batch(analysis_data)

            # Adiciona informações da sessão ao resultado
            result['session_analysis'] = {
                'session_id': session_id,
                'consolidacao_source': True,
                'items_analisados': len(analysis_data.get('items', [])),
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            self.logger.error(f"❌ Erro na análise da sessão {session_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }

    def process_batch(self, items: List[Dict[str, Any]], massive_data: Optional[Dict[str, Any]] = None, 
                     batch_size: int = 10) -> Dict[str, Any]:
        """
        Processa uma lista de itens em lotes para melhor performance
        
        Args:
            items: Lista de itens para processar
            massive_data: Contexto adicional
            batch_size: Tamanho do lote para processamento
            
        Returns:
            Dict com resultados do processamento em lote
        """
        logger.info(f"🚀 Iniciando processamento em lote: {len(items)} itens, lotes de {batch_size}")
        
        all_results = []
        approved_items = []
        rejected_items = []
        
        # Calcular total de lotes antecipadamente
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        # Processar em lotes
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"📦 Processando lote {batch_num}/{total_batches} ({len(batch)} itens)")
            
            batch_results = []
            for item in batch:
                result = self.process_item(item, massive_data)
                batch_results.append(result)
                all_results.append(result)
                
                # Categorizar resultado
                if hasattr(result, 'get') and callable(getattr(result, 'get')):
                    status = result.get('ai_review', {}).get('status', 'rejected')
                else:
                    # Se result for uma Exception ou objeto sem get()
                    status = 'rejected'
                if status == 'approved':
                    approved_items.append(result)
                else:
                    rejected_items.append(result)
            
            logger.info(f"✅ Lote {batch_num} concluído: {len([r for r in batch_results if r.get('ai_review', {}).get('status') == 'approved'])} aprovados")
        
        # Compilar estatísticas finais
        stats = self.get_statistics()
        
        return {
            'all_results': all_results,
            'approved_items': approved_items,
            'rejected_items': rejected_items,
            'statistics': stats,
            'batch_info': {
                'total_items': len(items),
                'batch_size': batch_size,
                'total_batches': total_batches,
                'approved_count': len(approved_items),
                'rejected_count': len(rejected_items),
                'approval_rate': len(approved_items) / len(items) if items else 0
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '3.0.0',
                'processing_mode': 'batch'
            }
        }

    async def process_batch_async(self, items: List[Dict[str, Any]], massive_data: Optional[Dict[str, Any]] = None,
                                 max_concurrent: int = 5) -> Dict[str, Any]:
        """
        Processa itens de forma assíncrona para melhor performance
        
        Args:
            items: Lista de itens para processar
            massive_data: Contexto adicional
            max_concurrent: Máximo de processamentos simultâneos
            
        Returns:
            Dict com resultados do processamento assíncrono
        """
        logger.info(f"⚡ Iniciando processamento assíncrono: {len(items)} itens, máx {max_concurrent} simultâneos")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_item(item):
            async with semaphore:
                # Como process_item não é async, executar em thread
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.process_item, item, massive_data)
        
        # Processar todos os itens
        tasks = [process_single_item(item) for item in items]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Processar resultados e exceções
        valid_results = []
        approved_items = []
        rejected_items = []
        errors = []
        
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.error(f"❌ Erro no item {i}: {result}")
                errors.append({'item_index': i, 'error': str(result)})
                # Criar resultado de erro
                error_result = self._create_error_result(items[i], str(result))
                valid_results.append(error_result)
                rejected_items.append(error_result)
            else:
                valid_results.append(result)
                if hasattr(result, 'get') and callable(getattr(result, 'get')):
                    status = result.get('ai_review', {}).get('status', 'rejected')
                else:
                    # Se result for uma Exception ou objeto sem get()
                    status = 'rejected'
                if status == 'approved':
                    approved_items.append(result)
                else:
                    rejected_items.append(result)
        
        stats = self.get_statistics()
        
        return {
            'all_results': valid_results,
            'approved_items': approved_items,
            'rejected_items': rejected_items,
            'errors': errors,
            'statistics': stats,
            'async_info': {
                'total_items': len(items),
                'max_concurrent': max_concurrent,
                'approved_count': len(approved_items),
                'rejected_count': len(rejected_items),
                'error_count': len(errors),
                'approval_rate': len(approved_items) / len(items) if items else 0
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '3.0.0',
                'processing_mode': 'async'
            }
        }

    def analyze_content_batch(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa lote de conteúdo"""
        try:
            items = input_data.get('items', [])
            context = input_data.get('context', {})

            if not items:
                return {
                    'success': False,
                    'error': 'Nenhum item fornecido para análise',
                    'timestamp': datetime.now().isoformat()
                }

            self.logger.info(f"🔍 Iniciando análise de {len(items)} itens")

            results = []
            total_items = len(items)

            for idx, item in enumerate(items):
                self.logger.info(f"📊 Analisando item {idx + 1}/{total_items}: {item.get('id', 'N/A')}")

                try:
                    result = self.process_item(item, context) # Use process_item directly
                    results.append(result)

                    # Pequeno delay entre análises
                    if idx < total_items - 1:
                        import time
                        time.sleep(0.5)

                except Exception as e:
                    self.logger.error(f"❌ Erro ao analisar item {item.get('id', 'N/A')}: {e}")
                    results.append({
                        'item_id': item.get('id', 'N/A'),
                        'status': 'error',
                        'error': str(e),
                        'confidence_score': 0.0
                    })

            # Gera estatísticas finais
            stats = self._generate_batch_statistics(results)

            return {
                'success': True,
                'total_items': total_items,
                'results': results,
                'statistics': stats,
                'processing_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'agent_version': '3.0',
                    'batch_size': total_items
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Erro na análise em lote: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gera estatísticas para uma análise em lote."""
        total_items = len(results)
        approved_count = sum(1 for r in results if r.get('ai_review', {}).get('status') == 'approved')
        rejected_count = total_items - approved_count
        error_count = sum(1 for r in results if r.get('ai_review', {}).get('error'))
        
        total_processing_time = sum(r.get('processing_time_seconds', 0) for r in results)
        avg_processing_time = total_processing_time / total_items if total_items > 0 else 0

        return {
            'total_items': total_items,
            'approved': approved_count,
            'rejected': rejected_count,
            'errors': error_count,
            'approval_rate': approved_count / total_items if total_items > 0 else 0,
            'total_processing_time_seconds': total_processing_time,
            'average_processing_time_seconds': avg_processing_time
        }


def run_external_review(input_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Função principal de entrada para o módulo externo

    Args:
        input_data (Dict[str, Any]): Dados de entrada contendo itens para análise
        config_path (Optional[str]): Caminho para arquivo de configuração

    Returns:
        Dict[str, Any]: Resultados da análise e itens processados
    """
    try:
        logger.info("🚀 Iniciando External AI Verifier...")

        # Initialize review agent
        review_agent = ExternalReviewAgent(config_path)

        # Extract items to process
        items = input_data.get('items', [])
        massive_data = input_data.get('context', {})

        if not items:
            logger.warning("Nenhum item fornecido para análise")
            return {
                'items': [],
                'statistics': {'total_processed': 0, 'error': 'Nenhum item fornecido'},
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '3.0.0'
                }
            }

        logger.info(f"Processando {len(items)} itens...")

        # Process each item
        processed_items = []
        approved_items = []
        rejected_items = []

        for item in items:
            result = review_agent.process_item(item, massive_data)
            processed_items.append(result)

            # Separate approved/rejected for easier consumption
            if result['ai_review']['status'] == 'approved':
                approved_items.append(result)
            else:
                rejected_items.append(result)

        # Compile final results
        final_result = {
            'items': approved_items,  # Only approved items by default
            'all_items': processed_items,  # All items with full analysis
            'rejected_items': rejected_items,  # Rejected items separately
            'statistics': review_agent.get_statistics(),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '3.0.0',
                'total_input_items': len(items),
                'approved_count': len(approved_items),
                'rejected_count': len(rejected_items)
            }
        }

        logger.info(f"✅ Processamento concluído: {len(approved_items)} aprovados, {len(rejected_items)} rejeitados")

        return final_result

    except Exception as e:
        logger.error(f"Erro crítico no External AI Verifier: {e}")
        return {
            'items': [],
            'statistics': {'error': str(e), 'total_processed': 0},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '3.0.0',
                'error': True
            }
        }

# Instância global do External AI Verifier
external_ai_verifier = ExternalReviewAgent()