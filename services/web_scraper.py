"""
Serviço de Web Scraping HÍBRIDO
Combina Alibaba WebSailor V2 + Fallbacks Robustos
Sistema 100% REAL - Sem dados simulados
"""

import os
import json
import time
import logging
import requests
import urllib.parse
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from .api_rotation_manager import get_api_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class WebScraperService:
    """Serviço de scraping com múltiplas APIs e fallbacks automáticos"""
    
    def __init__(self):
        # Configurar API rotation manager
        self.api_manager = get_api_manager()
        
        # Configurar LLMs com rotação de APIs
        self.llm_providers = []
        
        # Tentar configurar Gemini
        gemini_key_obj = self.api_manager.get_api_key('gemini')
        if gemini_key_obj and gemini_key_obj.key:
            try:
                genai.configure(api_key=gemini_key_obj.key)
                self.llm_providers.append({
                    'name': 'Gemini',
                    'model': genai.GenerativeModel('gemini-2.0-flash-exp'),
                    'type': 'gemini',
                    'api_name': gemini_key_obj.name
                })
                logger.info(f"[OK] Gemini configurado (API: {gemini_key_obj.name})")
            except Exception as e:
                logger.warning(f"Erro ao configurar Gemini: {e}")
        
        # Tentar configurar OpenRouter
        openrouter_key_obj = self.api_manager.get_api_key('openrouter')
        if openrouter_key_obj and openrouter_key_obj.key:
            self.llm_providers.append({
                'name': 'OpenRouter',
                'api_key': openrouter_key_obj.key,
                'type': 'openrouter',
                'api_name': openrouter_key_obj.name
            })
            logger.info(f"[OK] OpenRouter configurado (API: {openrouter_key_obj.name})")
        
        # Tentar configurar OpenAI
        openai_key_obj = self.api_manager.get_api_key('openai')
        if openai_key_obj and openai_key_obj.key:
            self.llm_providers.append({
                'name': 'OpenAI',
                'api_key': openai_key_obj.key,
                'type': 'openai',
                'api_name': openai_key_obj.name
            })
            logger.info(f"[OK] OpenAI configurado (API: {openai_key_obj.name})")
        
        # Carregar múltiplas API keys para rotação
        self.api_keys = self._load_api_keys()
        self.current_api_index = {key: 0 for key in self.api_keys.keys()}
        
        # Ordem de tentativa das APIs
        self.search_services = [
            self._search_serper,
            self._search_jina,
            self._search_bing,
            self._search_google_cse,
            self._search_google_direct,
            self._search_yahoo
        ]
        
        logger.info(f"[OK] Web Scraper inicializado com {len(self.search_services)} serviços")
    
    def _load_api_keys(self) -> Dict:
        """Carrega múltiplas chaves de API para rotação"""
        api_keys = {
            'serper': [],
            'jina': [],
            'google_cse': []
        }
        
        # Serper - múltiplas chaves
        main_key = os.getenv('SERPER_API_KEY')
        if main_key:
            api_keys['serper'].append(main_key.strip())
        
        for i in range(1, 4):
            key = os.getenv(f'SERPER_API_KEY_{i}')
            if key:
                api_keys['serper'].append(key.strip())
        
        # Jina - múltiplas chaves
        for i in range(1, 6):
            key = os.getenv(f'JINA_API_KEY_{i}') or (os.getenv('JINA_API_KEY') if i == 1 else None)
            if key:
                api_keys['jina'].append(key.strip())
        
        # Google CSE
        google_key = os.getenv('GOOGLE_SEARCH_KEY')
        google_cse = os.getenv('GOOGLE_CSE_ID')
        if google_key and google_cse:
            api_keys['google_cse'].append({'key': google_key, 'cse_id': google_cse})
        
        return api_keys
    
    def _get_next_api_key(self, service: str) -> str:
        """Rotaciona chaves de API"""
        if service not in self.api_keys or not self.api_keys[service]:
            return None
        
        keys = self.api_keys[service]
        current_index = self.current_api_index[service]
        key = keys[current_index]
        
        # Avançar para próxima chave
        self.current_api_index[service] = (current_index + 1) % len(keys)
        
        return key
    
    def _search_serper(self, query: str, num_results: int) -> List[Dict]:
        """Fallback 1: Serper API"""
        logger.info("[FALLBACK 1] Tentando Serper...")
        
        try:
            api_key = self._get_next_api_key('serper')
            if not api_key:
                raise Exception("Sem API key")
            
            url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }
            data = {
                'q': query,
                'num': num_results,
                'gl': 'br',
                'hl': 'pt-br'
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('organic', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
            
            if results:
                logger.info(f"[OK] Serper: {len(results)} resultados")
                return results[:num_results]
            
            raise Exception("Nenhum resultado")
            
        except Exception as e:
            logger.warning(f"[FALHA] Serper: {e}")
            return []
    
    def _search_jina(self, query: str, num_results: int) -> List[Dict]:
        """Fallback 2: Jina AI Search"""
        logger.info("[FALLBACK 2] Tentando Jina...")
        
        try:
            api_key = self._get_next_api_key('jina')
            if not api_key:
                raise Exception("Sem API key")
            
            url = f"https://s.jina.ai/{urllib.parse.quote(query)}"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('data', [])[:num_results]:
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('content', '')[:200]
                })
            
            if results:
                logger.info(f"[OK] Jina: {len(results)} resultados")
                return results
            
            raise Exception("Nenhum resultado")
            
        except Exception as e:
            logger.warning(f"[FALHA] Jina: {e}")
            return []
    
    def _search_bing(self, query: str, num_results: int) -> List[Dict]:
        """Fallback 3: Bing Search"""
        logger.info("[FALLBACK 3] Tentando Bing...")
        
        try:
            url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&count=50"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Language': 'pt-BR,pt;q=0.9'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('li', class_='b_algo'):
                try:
                    h2 = result.find('h2')
                    if not h2:
                        continue
                    
                    link = h2.find('a')
                    if not link:
                        continue
                    
                    title = link.get_text(strip=True)
                    url = link.get('href', '')
                    
                    if url and title and url.startswith('http'):
                        results.append({'title': title, 'url': url, 'snippet': ''})
                        
                        if len(results) >= num_results:
                            break
                except:
                    continue
            
            if results:
                logger.info(f"[OK] Bing: {len(results)} resultados")
                return results
            
            raise Exception("Nenhum resultado")
            
        except Exception as e:
            logger.warning(f"[FALHA] Bing: {e}")
            return []
    
    def _search_google_cse(self, query: str, num_results: int) -> List[Dict]:
        """Fallback 4: Google Custom Search"""
        logger.info("[FALLBACK 4] Tentando Google CSE...")
        
        try:
            cse_config = self._get_next_api_key('google_cse')
            if not cse_config:
                raise Exception("Sem API key")
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': cse_config['key'],
                'cx': cse_config['cse_id'],
                'q': query,
                'num': min(10, num_results)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
            
            if results:
                logger.info(f"[OK] Google CSE: {len(results)} resultados")
                return results
            
            raise Exception("Nenhum resultado")
            
        except Exception as e:
            logger.warning(f"[FALHA] Google CSE: {e}")
            return []
    
    def _search_google_direct(self, query: str, num_results: int) -> List[Dict]:
        """Fallback 5: Google direto"""
        logger.info("[FALLBACK 5] Tentando Google direto...")
        
        try:
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&num=50"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Language': 'pt-BR,pt;q=0.9'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='g'):
                try:
                    link = result.find('a')
                    if not link:
                        continue
                    
                    url = link.get('href', '')
                    if not url or not url.startswith('http'):
                        continue
                    
                    h3 = result.find('h3')
                    title = h3.get_text(strip=True) if h3 else url
                    
                    results.append({'title': title, 'url': url, 'snippet': ''})
                    
                    if len(results) >= num_results:
                        break
                except:
                    continue
            
            if results:
                logger.info(f"[OK] Google: {len(results)} resultados")
                return results
            
            raise Exception("Nenhum resultado")
            
        except Exception as e:
            logger.warning(f"[FALHA] Google: {e}")
            return []
    
    def _search_yahoo(self, query: str, num_results: int) -> List[Dict]:
        """Fallback 6: Yahoo Search"""
        logger.info("[FALLBACK 6] Tentando Yahoo...")
        
        try:
            url = f"https://search.yahoo.com/search?p={urllib.parse.quote(query)}&n=50"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='algo'):
                try:
                    link = result.find('a')
                    if not link:
                        continue
                    
                    url = link.get('href', '')
                    title = link.get_text(strip=True)
                    
                    if url and title and url.startswith('http'):
                        results.append({'title': title, 'url': url, 'snippet': ''})
                        
                        if len(results) >= num_results:
                            break
                except:
                    continue
            
            if results:
                logger.info(f"[OK] Yahoo: {len(results)} resultados")
                return results
            
            raise Exception("Nenhum resultado")
            
        except Exception as e:
            logger.warning(f"[FALHA] Yahoo: {e}")
            return []
    
    def search_web(self, query: str, num_results: int = 10) -> List[Dict]:
        """Busca com fallback automático"""
        logger.info(f"[SEARCH] Buscando {num_results} resultados para: {query}")
        
        for i, search_func in enumerate(self.search_services, 1):
            try:
                results = search_func(query, num_results)
                if results and len(results) > 0:
                    logger.info(f"[SUCCESS] Fallback {i} funcionou!")
                    return results
            except Exception as e:
                logger.warning(f"[ERRO] Fallback {i}: {e}")
                continue
        
        logger.error("[ERRO] Todos fallbacks falharam")
        return []
    
    def scrape_url(self, url: str) -> Dict:
        """Extrai conteúdo de URL com fallbacks para 403/bloqueios"""
        # Lista de User-Agents para rotação
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        
        for attempt, ua in enumerate(user_agents):
            try:
                headers = {
                    'User-Agent': ua,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                
                # Adicionar delay entre tentativas
                if attempt > 0:
                    time.sleep(2 * attempt)
                
                response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                
                # Se 403, tentar próximo User-Agent
                if response.status_code == 403:
                    logger.warning(f"[403] Tentativa {attempt + 1} bloqueada para {url}")
                    continue
                
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                    script.decompose()
                
                text = soup.get_text(separator=' ', strip=True)
                text = ' '.join(text.split())
                
                logger.info(f"[OK] Scraping bem-sucedido: {url} (tentativa {attempt + 1})")
                return {
                    'title': soup.title.string if soup.title else '',
                    'url': url,
                    'content': text[:5000],
                    'success': True
                }
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    logger.warning(f"[403] {url} bloqueado (tentativa {attempt + 1})")
                    continue
                else:
                    logger.error(f"[HTTP {e.response.status_code}] {url}: {e}")
                    break
            except Exception as e:
                logger.error(f"[ERRO] {url} (tentativa {attempt + 1}): {e}")
                if attempt == len(user_agents) - 1:  # Última tentativa
                    break
                continue
        
        # Se todas as tentativas falharam
        logger.error(f"[FALHA] Todas as tentativas falharam para: {url}")
        return {
            'title': '',
            'url': url,
            'content': '',
            'success': False,
            'error': 'Todas as tentativas de scraping falharam (possível bloqueio 403)'
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Chama LLM com fallback"""
        
        for provider in self.llm_providers:
            try:
                logger.info(f"[LLM] Tentando {provider['name']}...")
                
                if provider['type'] == 'gemini':
                    response = provider['model'].generate_content(prompt)
                    logger.info(f"[OK] {provider['name']} respondeu")
                    return response.text.strip()
                
                elif provider['type'] == 'openrouter':
                    headers = {
                        'Authorization': f"Bearer {provider['api_key']}",
                        'Content-Type': 'application/json'
                    }
                    data = {
                        'model': 'google/gemini-2.0-flash-exp:free',
                        'messages': [{'role': 'user', 'content': prompt}]
                    }
                    response = requests.post(
                        'https://openrouter.ai/api/v1/chat/completions',
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()['choices'][0]['message']['content']
                    logger.info(f"[OK] {provider['name']} respondeu")
                    return result.strip()
                
                elif provider['type'] == 'openai':
                    headers = {
                        'Authorization': f"Bearer {provider['api_key']}",
                        'Content-Type': 'application/json'
                    }
                    data = {
                        'model': 'gpt-4o-mini',
                        'messages': [{'role': 'user', 'content': prompt}]
                    }
                    response = requests.post(
                        'https://api.openai.com/v1/chat/completions',
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()['choices'][0]['message']['content']
                    logger.info(f"[OK] {provider['name']} respondeu")
                    return result.strip()
                    
            except Exception as e:
                logger.warning(f"[FALHA] {provider['name']}: {e}")
                continue
        
        raise Exception("Todos LLMs falharam")
    
    def generate_training_data(self, scraped_data: List[Dict], query: str) -> List[Dict]:
        """Gera dados de treinamento"""
        logger.info(f"[AI] Gerando dados de treinamento...")
        
        training_examples = []
        
        for idx, item in enumerate(scraped_data):
            if not item.get('success'):
                continue
            
            try:
                prompt = f"""Analise o seguinte conteúdo web e gere MÚLTIPLOS dados de treinamento no formato pergunta-resposta para fine-tuning de modelos de linguagem.

Query original: {query}
Título: {item['title']}
URL: {item['url']}
Conteúdo: {item['content'][:3000]}

Para CADA tópico ou conceito importante do conteúdo, gere um objeto JSON com:
- pergunta: uma pergunta natural e específica sobre o tópico (em português)
- resposta: uma resposta completa e informativa baseada no conteúdo (em português)
- contexto: breve contexto sobre o tópico
- categoria: categoria do conhecimento (ex: "tecnologia", "saúde", "educação", etc.)
- dificuldade: "básico", "intermediário" ou "avançado"
- fonte: URL da fonte
- confiabilidade: número de 0.0 a 1.0 indicando a confiabilidade da informação

IMPORTANTE:
- Gere perguntas variadas: "O que é...", "Como funciona...", "Quais são os benefícios de...", "Por que...", etc.
- As respostas devem ser informativas, precisas e baseadas no conteúdo
- Use linguagem natural e clara
- Foque em informações úteis e práticas
- Evite perguntas muito genéricas

Gere entre 4-6 objetos JSON. Retorne um array JSON. APENAS o JSON, sem texto adicional."""

                response_text = self._call_llm(prompt)
                
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                data = json.loads(response_text)
                
                if isinstance(data, list):
                    training_examples.extend(data)
                    logger.info(f"[OK] Item {idx+1}/{len(scraped_data)}: {len(data)} exemplos")
                else:
                    training_examples.append(data)
                    logger.info(f"[OK] Item {idx+1}/{len(scraped_data)}: 1 exemplo")
                
                time.sleep(1.5)
                
            except Exception as e:
                logger.error(f"[ERRO] Item {idx+1}: {e}")
                
                # Fallback: gerar dados básicos baseados no conteúdo
                content_words = item['content'].split()
                content_preview = ' '.join(content_words[:100])
                
                fallback_examples = [
                    {
                        'pergunta': f"O que você pode me dizer sobre {query}?",
                        'resposta': f"Com base nas informações disponíveis: {content_preview}...",
                        'contexto': f"Informações sobre {query} extraídas de {item['title']}",
                        'categoria': 'geral',
                        'dificuldade': 'básico',
                        'fonte': item['url'],
                        'confiabilidade': 0.6
                    },
                    {
                        'pergunta': f"Quais são os pontos principais sobre {query}?",
                        'resposta': f"Os principais pontos incluem: {content_preview}...",
                        'contexto': f"Resumo de informações sobre {query}",
                        'categoria': 'geral',
                        'dificuldade': 'intermediário',
                        'fonte': item['url'],
                        'confiabilidade': 0.6
                    },
                    {
                        'pergunta': f"Como posso aprender mais sobre {query}?",
                        'resposta': f"Para aprender mais sobre {query}, você pode consultar fontes como {item['title']} que contém informações relevantes sobre o tema.",
                        'contexto': f"Orientações para aprofundar conhecimento em {query}",
                        'categoria': 'educação',
                        'dificuldade': 'básico',
                        'fonte': item['url'],
                        'confiabilidade': 0.7
                    }
                ]
                
                training_examples.extend(fallback_examples)
                logger.info(f"[OK] Item {idx+1}: 3 exemplos fallback")
        
        logger.info(f"[TOTAL] {len(training_examples)} exemplos")
        return training_examples
    
    def scrape_and_generate(self, query: str, num_results: int = 10) -> Dict:
        """Pipeline completo"""
        logger.info(f"[START] Pipeline para: {query}")
        
        try:
            search_results = self.search_web(query, num_results)
            
            if not search_results:
                return {'success': False, 'error': 'Nenhum resultado'}
            
            scraped_data = []
            for result in search_results:
                logger.info(f"[SCRAPE] {result['url']}")
                data = self.scrape_url(result['url'])
                data['title'] = result['title']
                scraped_data.append(data)
                time.sleep(0.5)
            
            training_data = self.generate_training_data(scraped_data, query)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_query = ''.join(c if c.isalnum() or c in (' ', '_') else '_' for c in query)
            safe_query = safe_query.replace(' ', '_')[:50]
            filename = f"dados/{safe_query}_{timestamp}.json"
            
            os.makedirs('dados', exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[OK] Pipeline concluído: {len(training_data)} exemplos")
            logger.info(f"[OK] Salvo em: {filename}")
            
            return {
                'success': True,
                'filename': filename,
                'num_examples': len(training_data),
                'num_urls': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"[ERRO] Pipeline falhou: {e}")
            return {'success': False, 'error': str(e)}
