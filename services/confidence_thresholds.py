#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v3.0 - External Confidence Thresholds
Gerenciador de limiares de confiança para o módulo externo
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ExternalConfidenceThresholds:
    """Gerenciador de limiares de confiança externo"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializa os limiares de confiança"""
        self.thresholds = config.get('thresholds', {})
        
        # Default thresholds if not provided
        self.default_thresholds = {
            'approval': 0.75,
            'rejection': 0.35,
            'high_confidence': 0.85,
            'low_confidence': 0.5,
            'sentiment_neutral': 0.1,
            'bias_high_risk': 0.7,
            'llm_minimum': 0.6
        }
        
        # Merge with defaults
        for key, default_value in self.default_thresholds.items():
            if key not in self.thresholds:
                self.thresholds[key] = default_value
        
        logger.info(f"✅ External Confidence Thresholds inicializado com {len(self.thresholds)} limiares")
        logger.debug(f"Thresholds: {self.thresholds}")
    
    def get_threshold(self, score_type: str, default: Optional[float] = None) -> float:
        """
        Obtém o limiar para um tipo de pontuação específico
        
        Args:
            score_type (str): Tipo de pontuação (e.g., 'approval', 'rejection')
            default (Optional[float]): Valor padrão se não encontrado
            
        Returns:
            float: Limiar de confiança
        """
        if score_type in self.thresholds:
            return self.thresholds[score_type]
        
        if default is not None:
            return default
        
        # Return a reasonable default based on score type
        if 'approval' in score_type.lower():
            return 0.7
        elif 'rejection' in score_type.lower():
            return 0.3
        elif 'high' in score_type.lower():
            return 0.8
        elif 'low' in score_type.lower():
            return 0.4
        else:
            return 0.5
    
    def should_approve(self, confidence: float) -> bool:
        """Verifica se deve aprovar baseado na confiança"""
        return confidence >= self.get_threshold('approval')
    
    def should_reject(self, confidence: float) -> bool:
        """Verifica se deve rejeitar baseado na confiança"""
        return confidence <= self.get_threshold('rejection')
    
    def is_ambiguous(self, confidence: float) -> bool:
        """Verifica se está em faixa ambígua (entre rejection e approval)"""
        rejection_threshold = self.get_threshold('rejection')
        approval_threshold = self.get_threshold('approval')
        return rejection_threshold < confidence < approval_threshold
    
    def is_high_confidence(self, confidence: float) -> bool:
        """Verifica se é alta confiança"""
        return confidence >= self.get_threshold('high_confidence')
    
    def is_low_confidence(self, confidence: float) -> bool:
        """Verifica se é baixa confiança"""
        return confidence <= self.get_threshold('low_confidence')
    
    def is_high_bias_risk(self, risk_score: float) -> bool:
        """Verifica se é alto risco de viés"""
        return risk_score >= self.get_threshold('bias_high_risk')
    
    def classify_confidence_level(self, confidence: float) -> str:
        """
        Classifica o nível de confiança
        
        Args:
            confidence (float): Pontuação de confiança
            
        Returns:
            str: Nível de confiança ('high', 'medium', 'low')
        """
        if self.is_high_confidence(confidence):
            return 'high'
        elif self.is_low_confidence(confidence):
            return 'low'
        else:
            return 'medium'
    
    def get_decision_recommendation(self, confidence: float, risk_score: float = 0.0) -> Dict[str, Any]:
        """
        Recomenda uma decisão baseada na confiança e risco
        
        Args:
            confidence (float): Pontuação de confiança
            risk_score (float): Pontuação de risco
            
        Returns:
            Dict[str, Any]: Recomendação de decisão
        """
        # High risk overrides high confidence
        if self.is_high_bias_risk(risk_score):
            return {
                'decision': 'reject',
                'reason': 'Alto risco de viés/desinformação detectado',
                'confidence_level': self.classify_confidence_level(confidence),
                'risk_level': 'high',
                'requires_llm_analysis': False
            }
        
        # Clear approval
        if self.should_approve(confidence):
            return {
                'decision': 'approve',
                'reason': 'Alta confiança na qualidade do conteúdo',
                'confidence_level': self.classify_confidence_level(confidence),
                'risk_level': 'low' if risk_score < 0.3 else 'medium',
                'requires_llm_analysis': False
            }
        
        # Clear rejection
        if self.should_reject(confidence):
            return {
                'decision': 'reject',
                'reason': 'Baixa confiança na qualidade do conteúdo',
                'confidence_level': self.classify_confidence_level(confidence),
                'risk_level': 'low' if risk_score < 0.3 else 'medium',
                'requires_llm_analysis': False
            }
        
        # Ambiguous case - might need LLM analysis
        return {
            'decision': 'ambiguous',
            'reason': 'Confiança em faixa ambígua - requer análise adicional',
            'confidence_level': self.classify_confidence_level(confidence),
            'risk_level': 'medium' if risk_score < 0.5 else 'high',
            'requires_llm_analysis': True
        }
    
    def update_threshold(self, score_type: str, new_value: float):
        """
        Atualiza um limiar específico
        
        Args:
            score_type (str): Tipo de pontuação
            new_value (float): Novo valor do limiar
        """
        if 0.0 <= new_value <= 1.0:
            self.thresholds[score_type] = new_value
            logger.info(f"Threshold '{score_type}' atualizado para {new_value}")
        else:
            logger.warning(f"Valor inválido para threshold '{score_type}': {new_value} (deve estar entre 0.0 e 1.0)")
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Retorna todos os limiares configurados"""
        return self.thresholds.copy()
    
    def validate_thresholds(self) -> bool:
        """
        Valida se os limiares estão configurados corretamente
        
        Returns:
            bool: True se válidos, False caso contrário
        """
        try:
            # Check that rejection < approval
            rejection = self.get_threshold('rejection')
            approval = self.get_threshold('approval')
            
            if rejection >= approval:
                logger.error(f"Configuração inválida: rejection ({rejection}) deve ser menor que approval ({approval})")
                return False
            
            # Check that all thresholds are in valid range
            for key, value in self.thresholds.items():
                if not (0.0 <= value <= 1.0):
                    logger.error(f"Threshold '{key}' fora do range válido (0.0-1.0): {value}")
                    return False
            
            logger.info("✅ Todos os thresholds são válidos")
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação dos thresholds: {e}")
            return False