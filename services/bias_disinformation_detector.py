#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v3.0 - External Bias & Disinformation Detector
Módulo independente para detecção de viés e desinformação
"""

import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ExternalBiasDisinformationDetector:
    """Detector de viés e desinformação externo independente"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializa o detector de viés e desinformação"""
        self.config = config.get('bias_detection', {})
        self.enabled = self.config.get('enabled', True)
        
        # Load configuration
        self.bias_keywords = self.config.get('bias_keywords', [])
        self.disinformation_patterns = self.config.get('disinformation_patterns', [])
        self.rhetoric_devices = self.config.get('rhetoric_devices', [])
        
        logger.info(f"✅ External Bias & Disinformation Detector inicializado")
        logger.debug(f"Bias keywords: {len(self.bias_keywords)}, Patterns: {len(self.disinformation_patterns)}")
    
    def detect_bias_disinformation(self, text: str) -> Dict[str, float]:
        """
        Detecta padrões de viés e desinformação no texto
        
        Args:
            text (str): Texto para análise
            
        Returns:
            Dict[str, float]: Resultados da detecção
        """
        if not self.enabled or not text or not text.strip():
            return self._get_neutral_result()
        
        try:
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            text_lower = cleaned_text.lower()
            
            results = {
                'bias_score': 0.0,
                'disinformation_score': 0.0,
                'rhetoric_score': 0.0,
                'overall_risk': 0.0,
                'detected_bias_keywords': [],
                'detected_disinformation_patterns': [],
                'detected_rhetoric_devices': [],
                'confidence': 0.0,
                'analysis_details': {
                    'total_words': len(cleaned_text.split()),
                    'bias_matches': 0,
                    'disinformation_matches': 0,
                    'rhetoric_matches': 0
                }
            }
            
            # Detect bias keywords
            bias_analysis = self._detect_bias_keywords(text_lower)
            results.update(bias_analysis)
            
            # Detect disinformation patterns
            disinformation_analysis = self._detect_disinformation_patterns(text_lower)
            results.update(disinformation_analysis)
            
            # Detect rhetoric devices
            rhetoric_analysis = self._detect_rhetoric_devices(text_lower)
            results.update(rhetoric_analysis)
            
            # Calculate overall risk
            results['overall_risk'] = self._calculate_overall_risk(results)
            
            # Calculate confidence
            results['confidence'] = self._calculate_confidence(results)
            
            logger.debug(f"Bias/Disinformation analysis: risk={results['overall_risk']:.3f}, confidence={results['confidence']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na detecção de viés/desinformação: {e}")
            return self._get_neutral_result()
    
    def _clean_text(self, text: str) -> str:
        """Limpa o texto para análise"""
        if not text:
            return ""
        
        # Remove URLs, mentions, hashtags but keep text structure
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[@#](\w+)', r'\\1', text)
        text = re.sub(r'\\s+', ' ', text).strip()
        
        return text
    
    def _detect_bias_keywords(self, text_lower: str) -> Dict[str, Any]:
        """Detecta palavras-chave de viés"""
        detected_keywords = []
        bias_score = 0.0
        
        for keyword in self.bias_keywords:
            if keyword.lower() in text_lower:
                detected_keywords.append(keyword)
                bias_score += 0.1  # Each bias keyword adds 0.1 to score
        
        # Normalize score (cap at 1.0)
        bias_score = min(bias_score, 1.0)
        
        return {
            'bias_score': bias_score,
            'detected_bias_keywords': detected_keywords,
            'analysis_details': {'bias_matches': len(detected_keywords)}
        }
    
    def _detect_disinformation_patterns(self, text_lower: str) -> Dict[str, Any]:
        """Detecta padrões de desinformação"""
        detected_patterns = []
        disinformation_score = 0.0
        
        for pattern in self.disinformation_patterns:
            if pattern.lower() in text_lower:
                detected_patterns.append(pattern)
                disinformation_score += 0.15  # Each pattern adds more weight
        
        # Additional pattern detection with regex
        # Look for vague authority claims
        authority_patterns = [
            r'especialistas? (?:afirmam?|dizem?|garantem?)',
            r'estudos? (?:comprovam?|mostram?|indicam?)',
            r'pesquisas? (?:revelam?|demonstram?|apontam?)',
            r'cientistas? (?:descobriram?|provaram?|confirmaram?)'
        ]

        # Look for strawman fallacy
        strawman_patterns = [
            r'(?:eles|eles dizem|a esquerda|a direita) querem? (?:destruir|acabar com|impor) (?:nossa cultura|nossos valores|a família)',
            r'(?:argumento do espantalho|distorcem|exageram) (?:o que eu disse|nossas palavras)'
        ]

        # Look for ad hominem attacks
        ad_hominem_patterns = [
            r'(?:ele|ela|você) não tem moral para falar',
            r'(?:ignorante|burro|mentiroso|hipócrita) (?:para acreditar|para defender)'
        ]

        # Look for false dichotomy
        false_dichotomy_patterns = [
            r'(?:ou é|ou você está com|ou você apoia) (?:nós|eles) (?:ou contra nós|ou contra eles)',
            r'(?:só existem|apenas duas? opções?)'
        ]

        # Look for appeal to emotion
        appeal_to_emotion_patterns = [
            r'(?:pense nas crianças|e se fosse você|você não se importa)',
            r'(?:chocante|absurdo|inacreditável|revoltante)'
        ]

        all_regex_patterns = authority_patterns + strawman_patterns + ad_hominem_patterns + false_dichotomy_patterns + appeal_to_emotion_patterns


        for pattern in all_regex_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                detected_patterns.extend(matches)
                disinformation_score += len(matches) * 0.1
        

        
        # Normalize score
        disinformation_score = min(disinformation_score, 1.0)
        
        return {
            'disinformation_score': disinformation_score,
            'detected_disinformation_patterns': detected_patterns,
            'analysis_details': {'disinformation_matches': len(detected_patterns)}
        }
    
    def _detect_rhetoric_devices(self, text_lower: str) -> Dict[str, Any]:
        """Detecta dispositivos retóricos"""
        detected_devices = []
        rhetoric_score = 0.0
        
        # Detect emotional manipulation patterns
        emotional_patterns = {
            'apelo ao medo': [r'perig(o|oso|osa)', r'risco', r'ameaça', r'catástrofe'],
            'apelo à emoção': [r'imaginem?', r'pensem?', r'sintam?'],
            'generalização': [r'todos? (?:sabem?|fazem?)', r'ninguém', r'sempre', r'nunca'],
            'falsa dicotomia': [r'ou (?:você|vocês?)', r'apenas duas? opç']
        }
        
        for device_name, patterns in emotional_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_devices.append(device_name)
                    rhetoric_score += 0.1
                    break  # Only count each device type once
        
        # Check configured rhetoric devices
        for device in self.rhetoric_devices:
            if device.lower() in text_lower:
                detected_devices.append(device)
                rhetoric_score += 0.1
        
        # Normalize score
        rhetoric_score = min(rhetoric_score, 1.0)
        
        return {
            'rhetoric_score': rhetoric_score,
            'detected_rhetoric_devices': list(set(detected_devices)),  # Remove duplicates
            'analysis_details': {'rhetoric_matches': len(detected_devices)}
        }
    
    def _calculate_overall_risk(self, results: Dict[str, Any]) -> float:
        """Calcula o risco geral"""
        # Weighted combination of different risk factors
        bias_weight = 0.3
        disinformation_weight = 0.4
        rhetoric_weight = 0.3
        
        overall_risk = (
            results.get('bias_score', 0.0) * bias_weight +
            results.get('disinformation_score', 0.0) * disinformation_weight +
            results.get('rhetoric_score', 0.0) * rhetoric_weight
        )
        
        return min(overall_risk, 1.0)
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calcula a confiança da análise"""
        try:
            total_words = results.get('analysis_details', {}).get('total_words', 1)
            total_matches = (
                len(results.get('detected_bias_keywords', [])) +
                len(results.get('detected_disinformation_patterns', [])) +
                len(results.get('detected_rhetoric_devices', []))
            )
            
            # Base confidence on detection density and text length
            if total_words < 10:
                return 0.3  # Low confidence for very short text
            
            detection_density = total_matches / total_words
            
            # Higher density = higher confidence in detection
            confidence = min(0.5 + (detection_density * 5), 1.0)
            
            # If no matches found, still have some confidence it's clean
            if total_matches == 0 and total_words > 20:
                confidence = 0.7
            elif total_matches == 0:
                confidence = 0.5
            
            return max(confidence, 0.1)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de confiança: {e}")
            return 0.5
    
    def _get_neutral_result(self) -> Dict[str, float]:
        """Retorna resultado neutro padrão"""
        return {
            'bias_score': 0.0,
            'disinformation_score': 0.0,
            'rhetoric_score': 0.0,
            'overall_risk': 0.0,
            'detected_bias_keywords': [],
            'detected_disinformation_patterns': [],
            'detected_rhetoric_devices': [],
            'confidence': 0.1,
            'analysis_details': {
                'total_words': 0,
                'bias_matches': 0,
                'disinformation_matches': 0,
                'rhetoric_matches': 0
            }
        }