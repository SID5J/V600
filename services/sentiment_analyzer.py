#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v3.0 - External Sentiment Analyzer
Módulo independente para análise de sentimento e polaridade
"""

import logging
from typing import Dict, Any, Optional
import re

# Try to import TextBlob, fallback if not available
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob não disponível. Usando análise básica.")

# Try to import VADER, fallback if not available
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER Sentiment não disponível.")

logger = logging.getLogger(__name__)

class ExternalSentimentAnalyzer:
    """Analisador de sentimento externo independente"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializa o analisador de sentimento"""
        self.config = config.get('sentiment_analysis', {})
        self.enabled = self.config.get('enabled', True)
        self.use_vader = self.config.get('use_vader', True) and VADER_AVAILABLE
        self.use_textblob = self.config.get('use_textblob', True) and TEXTBLOB_AVAILABLE
        self.polarity_weights = self.config.get('polarity_weights', {
            'positive': 1.1,
            'negative': 0.8,
            'neutral': 1.0
        })
        
        # Initialize VADER if available and enabled
        if self.use_vader:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Palavras para análise básica quando bibliotecas não estão disponíveis
        self.positive_words = {"bom", "ótimo", "excelente", "maravilhoso", "perfeito", "incrível", "fantástico", "amor", "feliz", "alegre", "positivo", "sucesso", "ganho", "oportunidade", "melhor", "bem", "confiável", "seguro", "verdadeiro", "justo", "aprovado"}
        self.negative_words = {"ruim", "péssimo", "terrível", "horrível", "odiar", "triste", "raiva", "problema", "erro", "falha", "negativo", "fracasso", "perda", "ameaça", "pior", "mal", "duvidoso", "inseguro", "falso", "injusto", "rejeitado", "viés", "desinformação"}
        
        logger.info(f"✅ External Sentiment Analyzer inicializado (VADER: {self.use_vader}, TextBlob: {self.use_textblob})")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analisa o sentimento do texto fornecido
        
        Args:
            text (str): Texto para análise
            
        Returns:
            Dict[str, float]: Resultados da análise de sentimento
        """
        if not self.enabled or not text or not text.strip():
            return self._get_neutral_sentiment()
        
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            results = {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0,
                'classification': 'neutral',
                'confidence': 0.0,
                'analysis_methods': []
            }
            
            # TextBlob Analysis
            if self.use_textblob:
                textblob_results = self._analyze_with_textblob(cleaned_text)
                results.update(textblob_results)
                results['analysis_methods'].append('textblob')
            
            # VADER Analysis
            if self.use_vader:
                vader_results = self._analyze_with_vader(cleaned_text)
                # Combine VADER results with TextBlob
                results['compound'] = vader_results['compound']
                results['positive'] = vader_results['pos']
                results['negative'] = vader_results['neg'] 
                results['neutral'] = vader_results['neu']
                results['analysis_methods'].append('vader')
                
                # Use VADER for final classification if available
                results['classification'] = self._classify_sentiment_vader(vader_results)
            
            # Apply polarity weights
            results = self._apply_polarity_weights(results)
            
            # Calculate final confidence
            results['confidence'] = self._calculate_confidence(results)
            
            logger.debug(f"Sentiment analysis completed: {results['classification']} (confidence: {results['confidence']:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na análise de sentimento: {e}")
            return self._get_neutral_sentiment()
    
    def _clean_text(self, text: str) -> str:
        """Limpa o texto para análise"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (keep the content)
        text = re.sub(r'[@#](\w+)', r'\\1', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        
        return text
    
    def _analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """Análise com TextBlob ou análise básica"""
        try:
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                return {
                    'polarity': blob.sentiment.polarity,  # -1 to 1
                    'subjectivity': blob.sentiment.subjectivity  # 0 to 1
                }
            else:
                # Análise básica usando palavras-chave
                return self._basic_sentiment_analysis(text)
        except Exception as e:
            logger.warning(f"Erro na análise de sentimento: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def _basic_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Análise básica de sentimento usando palavras-chave"""
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Calcular polaridade básica
        total_words = len(text_lower.split())
        if total_words == 0:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        polarity = (positive_count - negative_count) / max(total_words, 1)
        polarity = max(-1.0, min(1.0, polarity * 10))  # Normalizar entre -1 e 1
        
        # Subjetividade baseada na quantidade de palavras emocionais
        subjectivity = min((positive_count + negative_count) / max(total_words, 1) * 5, 1.0)
        
        return {'polarity': polarity, 'subjectivity': subjectivity}
    
    def _analyze_with_vader(self, text: str) -> Dict[str, float]:
        """Análise com VADER"""
        try:
            return self.vader_analyzer.polarity_scores(text)
        except Exception as e:
            logger.warning(f"Erro no VADER: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
    
    def _classify_sentiment_vader(self, vader_scores: Dict[str, float]) -> str:
        """Classifica sentimento baseado nos scores do VADER"""
        compound = vader_scores.get('compound', 0.0)
        
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _apply_polarity_weights(self, results: Dict[str, float]) -> Dict[str, float]:
        """Aplica pesos de polaridade configurados"""
        classification = results.get('classification', 'neutral')
        weight = self.polarity_weights.get(classification, 1.0)
        
        # Adjust polarity and compound scores
        if 'polarity' in results:
            results['polarity'] *= weight
        if 'compound' in results:
            results['compound'] *= weight
            
        return results
    
    def _calculate_confidence(self, results: Dict[str, float]) -> float:
        """Calcula confiança da análise"""
        try:
            # Base confidence on the strength of sentiment indicators
            polarity_abs = abs(results.get('polarity', 0.0))
            compound_abs = abs(results.get('compound', 0.0))
            subjectivity = results.get('subjectivity', 0.0)
            
            # Higher absolute values indicate stronger sentiment (more confident)
            sentiment_strength = max(polarity_abs, compound_abs)
            
            # Subjectivity can indicate confidence (highly subjective = less reliable)
            subjectivity_penalty = subjectivity * 0.2
            
            # Method bonus (more methods = higher confidence)
            method_count = len(results.get('analysis_methods', []))
            method_bonus = min(method_count * 0.1, 0.2)
            
            confidence = min(sentiment_strength + method_bonus - subjectivity_penalty, 1.0)
            confidence = max(confidence, 0.1)  # Minimum confidence
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de confiança: {e}")
            return 0.5
    
    def _get_neutral_sentiment(self) -> Dict[str, float]:
        """Retorna resultado neutro padrão"""
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0,
            'classification': 'neutral',
            'confidence': 0.1,
            'analysis_methods': []
        }