#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v3.0 - External Contextual Analyzer
Analisador contextual para o módulo externo de verificação
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class ExternalContextualAnalyzer:
    """Analisador contextual externo independente"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializa o analisador contextual"""
        self.config = config.get('contextual_analysis', {})
        self.enabled = self.config.get('enabled', True)
        self.check_consistency = self.config.get('check_consistency', True)
        self.analyze_source_reliability = self.config.get('analyze_source_reliability', True)
        self.verify_temporal_coherence = self.config.get('verify_temporal_coherence', True)
        
        # Initialize context cache for cross-item analysis
        self.context_cache = {
            'processed_items': [],
            'source_patterns': {},
            'content_patterns': {},
            'temporal_markers': []
        }
        
        logger.info(f"✅ External Contextual Analyzer inicializado")
        logger.debug(f"Configurações: consistency={self.check_consistency}, source={self.analyze_source_reliability}, temporal={self.verify_temporal_coherence}")
    
    def analyze_context(self, item_data: Dict[str, Any], massive_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analisa o item em contexto mais amplo
        
        Args:
            item_data (Dict[str, Any]): Dados do item individual
            massive_data (Optional[Dict[str, Any]]): Dados contextuais mais amplos
            
        Returns:
            Dict[str, Any]: Análise contextual
        """
        if not self.enabled:
            return self._get_neutral_result()
        
        try:
            # Initialize context analysis result
            context_result = {
                'contextual_confidence': 0.5,
                'consistency_score': 0.5,
                'source_reliability_score': 0.5,
                'temporal_coherence_score': 0.5,
                'context_flags': [],
                'context_insights': [],
                'adjustment_factor': 0.0
            }
            
            # Extract text content for analysis
            text_content = self._extract_text_content(item_data)
            
            # Perform different types of contextual analysis
            if self.check_consistency:
                consistency_analysis = self._analyze_consistency(text_content, item_data, massive_data)
                context_result.update(consistency_analysis)
            
            if self.analyze_source_reliability:
                source_analysis = self._analyze_source_reliability(item_data, massive_data)
                context_result.update(source_analysis)
            
            if self.verify_temporal_coherence:
                temporal_analysis = self._analyze_temporal_coherence(text_content, item_data)
                context_result.update(temporal_analysis)
            
            # Calculate overall contextual confidence
            context_result['contextual_confidence'] = self._calculate_contextual_confidence(context_result)
            
            # Update context cache for future analysis
            self._update_context_cache(item_data, context_result)
            
            logger.debug(f"Context analysis: confidence={context_result['contextual_confidence']:.3f}")
            
            return context_result
            
        except Exception as e:
            logger.error(f"Erro na análise contextual: {e}")
            return self._get_neutral_result()
    
    def _extract_text_content(self, item_data: Dict[str, Any]) -> str:
        """Extrai conteúdo textual relevante do item"""
        content_fields = ['content', 'text', 'title', 'description', 'summary']
        
        text_content = ""
        for field in content_fields:
            if field in item_data and item_data[field]:
                text_content += f" {item_data[field]}"
        
        return text_content.strip()
    
    def _analyze_consistency(self, text_content: str, item_data: Dict[str, Any], massive_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa consistência interna e externa"""
        consistency_result = {
            'consistency_score': 0.5,
            'consistency_flags': [],
            'consistency_insights': []
        }
        
        try:
            score = 0.5
            flags = []
            insights = []
            
            # Check internal consistency
            internal_score, internal_flags = self._check_internal_consistency(text_content)
            score = (score + internal_score) / 2
            flags.extend(internal_flags)
            
            # Check consistency with previous items if available
            if self.context_cache['processed_items']:
                external_score, external_flags = self._check_external_consistency(text_content, item_data)
                score = (score + external_score) / 2
                flags.extend(external_flags)
                
                if external_score < 0.3:
                    insights.append("Conteúdo inconsistente com padrões anteriores")
                elif external_score > 0.8:
                    insights.append("Conteúdo altamente consistente com padrões estabelecidos")
            
            consistency_result.update({
                'consistency_score': score,
                'consistency_flags': flags,
                'consistency_insights': insights
            })
            
        except Exception as e:
            logger.warning(f"Erro na análise de consistência: {e}")
        
        return consistency_result
    
    def _check_internal_consistency(self, text_content: str) -> tuple:
        """Verifica consistência interna do texto"""
        score = 0.7  # Start with good assumption
        flags = []
        
        if not text_content or len(text_content.strip()) < 10:
            return 0.3, ["Conteúdo muito curto para análise de consistência"]
        
        # Check for contradictory statements
        contradiction_patterns = [
            (r'sempre.*nunca', "Contradição: 'sempre' e 'nunca' no mesmo contexto"),
            (r'todos?.*ninguém', "Contradição: generalização conflitante"),
            (r'impossível.*possível', "Contradição: possibilidade conflitante"),
            (r'verdade.*mentira', "Contradição: veracidade conflitante")
        ]
        
        for pattern, flag_msg in contradiction_patterns:
            if re.search(pattern, text_content.lower()):
                score -= 0.2
                flags.append(flag_msg)
        
        # Check for temporal inconsistencies
        temporal_patterns = [
            r'ontem.*amanhã',
            r'passado.*futuro.*hoje',
            r'antes.*depois.*simultaneamente'
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, text_content.lower()):
                score -= 0.1
                flags.append("Possível inconsistência temporal")
        
        return max(score, 0.0), flags
    
    def _check_external_consistency(self, text_content: str, item_data: Dict[str, Any]) -> tuple:
        """Verifica consistência com itens processados anteriormente"""
        score = 0.5
        flags = []
        
        try:
            # Compare with recent processed items
            recent_items = self.context_cache['processed_items'][-5:]  # Last 5 items
            
            if not recent_items:
                return 0.5, []
            
            # Simple keyword-based similarity check
            current_words = set(text_content.lower().split())
            
            similarity_scores = []
            for prev_item in recent_items:
                prev_words = set(prev_item.get('text', '').lower().split())
                if prev_words:
                    intersection = len(current_words & prev_words)
                    union = len(current_words | prev_words)
                    similarity = intersection / union if union > 0 else 0
                    similarity_scores.append(similarity)
            
            if similarity_scores:
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                
                # Very high similarity might indicate duplication
                if avg_similarity > 0.9:
                    score = 0.3
                    flags.append("Conteúdo muito similar a itens anteriores (possível duplicação)")
                # Very low similarity might be inconsistent
                elif avg_similarity < 0.1:
                    score = 0.4
                    flags.append("Conteúdo muito diferente do padrão estabelecido")
                else:
                    score = 0.7  # Good consistency
                    
        except Exception as e:
            logger.warning(f"Erro na verificação de consistência externa: {e}")
        
        return score, flags
    
    def _analyze_source_reliability(self, item_data: Dict[str, Any], massive_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa confiabilidade da fonte"""
        source_result = {
            'source_reliability_score': 0.5,
            'source_flags': [],
            'source_insights': []
        }
        
        try:
            score = 0.5
            flags = []
            insights = []
            
            # Extract source information
            source_info = self._extract_source_info(item_data)
            
            if not source_info:
                score = 0.3
                flags.append("Fonte não identificada")
                return {**source_result, 'source_reliability_score': score, 'source_flags': flags}
            
            # Check source patterns
            source_domain = source_info.get('domain', '').lower()
            
            # Known reliable patterns
            reliable_indicators = [
                '.edu', '.gov', '.org',
                'academia', 'university', 'instituto',
                'pesquisa', 'ciencia', 'journal'
            ]
            
            unreliable_indicators = [
                'blog', 'forum', 'social',
                'fake', 'rumor', 'gossip'
            ]
            
            for indicator in reliable_indicators:
                if indicator in source_domain:
                    score += 0.2
                    insights.append(f"Fonte contém indicador confiável: {indicator}")
                    break
            
            for indicator in unreliable_indicators:
                if indicator in source_domain:
                    score -= 0.3
                    flags.append(f"Fonte contém indicador de baixa confiabilidade: {indicator}")
                    break
            
            # Check source history in cache
            if source_domain in self.context_cache['source_patterns']:
                source_stats = self.context_cache['source_patterns'][source_domain]
                avg_quality = source_stats.get('avg_quality', 0.5)
                
                if avg_quality > 0.7:
                    score += 0.1
                    insights.append("Fonte com histórico positivo")
                elif avg_quality < 0.4:
                    score -= 0.1
                    flags.append("Fonte com histórico problemático")
            
            score = min(max(score, 0.0), 1.0)
            
            source_result.update({
                'source_reliability_score': score,
                'source_flags': flags,
                'source_insights': insights
            })
            
        except Exception as e:
            logger.warning(f"Erro na análise de fonte: {e}")
        
        return source_result
    
    def _extract_source_info(self, item_data: Dict[str, Any]) -> Dict[str, str]:
        """Extrai informações da fonte"""
        source_fields = ['source', 'url', 'domain', 'author', 'publisher']
        source_info = {}
        
        for field in source_fields:
            if field in item_data and item_data[field]:
                source_info[field] = str(item_data[field])
        
        # Extract domain from URL if available
        if 'url' in source_info and 'domain' not in source_info:
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(source_info['url'])
                source_info['domain'] = parsed.netloc
            except:
                pass
        
        return source_info
    
    def _analyze_temporal_coherence(self, text_content: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa coerência temporal"""
        temporal_result = {
            'temporal_coherence_score': 0.5,
            'temporal_flags': [],
            'temporal_insights': []
        }
        
        try:
            score = 0.7
            flags = []
            insights = []
            
            # Extract temporal markers
            temporal_markers = self._extract_temporal_markers(text_content)
            
            # Check for temporal inconsistencies
            if len(temporal_markers) > 1:
                coherent, issues = self._check_temporal_coherence(temporal_markers)
                if not coherent:
                    score -= 0.3
                    flags.extend(issues)
                else:
                    insights.append("Marcadores temporais coerentes")
            
            # Check against item timestamp if available
            if 'timestamp' in item_data or 'date' in item_data:
                item_time = item_data.get('timestamp') or item_data.get('date')
                temporal_consistency = self._check_item_temporal_consistency(temporal_markers, item_time)
                if temporal_consistency < 0.5:
                    score -= 0.2
                    flags.append("Inconsistência entre conteúdo e timestamp do item")
            
            temporal_result.update({
                'temporal_coherence_score': max(score, 0.0),
                'temporal_flags': flags,
                'temporal_insights': insights
            })
            
        except Exception as e:
            logger.warning(f"Erro na análise temporal: {e}")
        
        return temporal_result
    
    def _extract_temporal_markers(self, text: str) -> List[str]:
        """Extrai marcadores temporais do texto"""
        temporal_patterns = [
            r'(?:ontem|hoje|amanhã)',
            r'(?:esta|próxima|passada)\s+(?:semana|segunda|terça|quarta|quinta|sexta|sábado|domingo)',
            r'(?:este|próximo|passado)\s+(?:mês|ano)',
            r'(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)',
            r'(?:2019|2020|2021|2022|2023|2024|2025)',
            r'há\s+\d+\s+(?:dias?|meses?|anos?)',
            r'em\s+\d+\s+(?:dias?|meses?|anos?)'
        ]
        
        markers = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text.lower())
            markers.extend(matches)
        
        return markers
    
    def _check_temporal_coherence(self, markers: List[str]) -> tuple:
        """Verifica coerência entre marcadores temporais"""
        # Simple coherence check - this could be made more sophisticated
        issues = []
        
        # Check for obvious contradictions
        if any('ontem' in m for m in markers) and any('amanhã' in m for m in markers):
            issues.append("Contradição temporal: 'ontem' e 'amanhã' no mesmo contexto")
        
        # Check for year contradictions
        years = [m for m in markers if re.search(r'20\d{2}', m)]
        if len(set(years)) > 2:
            issues.append("Múltiplos anos mencionados - possível inconsistência")
        
        return len(issues) == 0, issues
    
    def _check_item_temporal_consistency(self, markers: List[str], item_time: str) -> float:
        """Verifica consistência temporal com timestamp do item"""
        try:
            # Simple check - this could be enhanced
            current_year = datetime.now().year
            
            # Check if markers mention current year
            mentions_current_year = any(str(current_year) in m for m in markers)
            mentions_old_years = any(str(year) in m for m in markers if year < current_year - 1 for year in range(2015, current_year))
            
            if mentions_current_year:
                return 0.8
            elif mentions_old_years:
                return 0.4
            else:
                return 0.6
                
        except Exception as e:
            logger.warning(f"Erro na verificação de consistência temporal: {e}")
            return 0.5
    
    def _calculate_contextual_confidence(self, context_result: Dict[str, Any]) -> float:
        """Calcula confiança contextual geral"""
        try:
            scores = [
                context_result.get('consistency_score', 0.5),
                context_result.get('source_reliability_score', 0.5),
                context_result.get('temporal_coherence_score', 0.5)
            ]
            
            # Weight the scores
            weights = [0.4, 0.4, 0.2]  # Consistency and source are more important
            
            weighted_score = sum(score * weight for score, weight in zip(scores, weights))
            
            # Penalty for flags
            total_flags = (
                len(context_result.get('consistency_flags', [])) +
                len(context_result.get('source_flags', [])) +
                len(context_result.get('temporal_flags', []))
            )
            
            flag_penalty = min(total_flags * 0.1, 0.3)
            
            final_score = max(weighted_score - flag_penalty, 0.0)
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de confiança contextual: {e}")
            return 0.5
    
    def _update_context_cache(self, item_data: Dict[str, Any], context_result: Dict[str, Any]):
        """Atualiza cache de contexto com informações do item atual"""
        try:
            # Add to processed items (keep last 20)
            item_summary = {
                'text': self._extract_text_content(item_data)[:500],  # First 500 chars
                'context_score': context_result.get('contextual_confidence', 0.5),
                'timestamp': datetime.now().isoformat()
            }
            
            self.context_cache['processed_items'].append(item_summary)
            if len(self.context_cache['processed_items']) > 20:
                self.context_cache['processed_items'] = self.context_cache['processed_items'][-20:]
            
            # Update source patterns
            source_info = self._extract_source_info(item_data)
            if source_info.get('domain'):
                domain = source_info['domain']
                if domain not in self.context_cache['source_patterns']:
                    self.context_cache['source_patterns'][domain] = {
                        'count': 0,
                        'total_quality': 0.0,
                        'avg_quality': 0.5
                    }
                
                stats = self.context_cache['source_patterns'][domain]
                stats['count'] += 1
                stats['total_quality'] += context_result.get('contextual_confidence', 0.5)
                stats['avg_quality'] = stats['total_quality'] / stats['count']
                
        except Exception as e:
            logger.warning(f"Erro na atualização do cache de contexto: {e}")
    
    def _get_neutral_result(self) -> Dict[str, Any]:
        """Retorna resultado neutro padrão"""
        return {
            'contextual_confidence': 0.5,
            'consistency_score': 0.5,
            'source_reliability_score': 0.5,
            'temporal_coherence_score': 0.5,
            'context_flags': [],
            'context_insights': [],
            'adjustment_factor': 0.0
        }