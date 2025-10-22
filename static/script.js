// Sistema de Fine-Tuning - Frontend UNIFICADO E COMPLETO
console.log('🔧 DEBUG: Script carregado!');

let socket;
let appState = {
    modelLoaded: false,
    dataLoaded: false,
    trainingActive: false,
    modelPath: null,
    dataPath: null,
    dataFile: null
};

// Aguardar DOM carregar
document.addEventListener('DOMContentLoaded', () => {
    console.log('🔧 DEBUG: DOM carregado, inicializando elementos');
    console.log('🔧 DEBUG: Testando JavaScript - FUNCIONANDO!');
    
    // Inicializar Socket.IO após DOM carregar
    socket = io();
    console.log('🔧 DEBUG: Socket.IO inicializado:', socket);
    
    // Elementos DOM
    const modelPath = document.getElementById('modelPath');
    const dataPath = document.getElementById('dataPath');
    const loadModelBtn = document.getElementById('loadModelBtn');
    const loadDataBtn = document.getElementById('loadDataBtn');
    const startTraining = document.getElementById('startTraining');
    const saveModel = document.getElementById('executeSave');
    const messageInput = document.getElementById('messageInput');
    const sendMessage = document.getElementById('sendMessage');
    const chatMessages = document.getElementById('chatMessages');
    const modelStatus = document.getElementById('modelStatus');
    const createModelBtn = document.getElementById('createModel');
    const modelFile = document.getElementById('modelFile');
    const dataFile = document.getElementById('dataFile');
    
    console.log('🔧 DEBUG: Elementos encontrados:', {
        modelPath: !!modelPath,
        dataPath: !!dataPath,
        loadModelBtn: !!loadModelBtn,
        loadDataBtn: !!loadDataBtn,
        startTraining: !!startTraining,
        saveModel: !!saveModel,
        messageInput: !!messageInput,
        sendMessage: !!sendMessage,
        chatMessages: !!chatMessages,
        modelStatus: !!modelStatus,
        createModelBtn: !!createModelBtn
    });

    // ============================================================
    // SELETOR DE PASTA PARA SALVAR
    // ============================================================
    const savePathFile = document.getElementById('savePathFile');
    const savePath = document.getElementById('savePath');
    
    if (savePathFile && savePath) {
        savePathFile.addEventListener('change', (event) => {
            const files = event.target.files;
            if (files.length > 0) {
                // Pegar o caminho da primeira pasta selecionada
                const firstFile = files[0];
                const folderPath = firstFile.webkitRelativePath.split('/')[0];
                savePath.value = `modelos/${folderPath}`;
                console.log('🔧 DEBUG: Pasta selecionada:', folderPath);
            }
        });
    }
    
    // ============================================================
    // CRIAR MODELO DO ZERO
    // ============================================================
    console.log('🔧 DEBUG: Verificando botão createModel:', createModelBtn);
    console.log('🔧 DEBUG: Tipo do elemento:', typeof createModelBtn);
    console.log('🔧 DEBUG: ID do elemento:', createModelBtn ? createModelBtn.id : 'N/A');
    
    if (createModelBtn) {
        console.log('🔧 DEBUG: Botão criar modelo encontrado, adicionando evento');
        
        // Adicionar evento de click
        createModelBtn.addEventListener('click', function(event) {
            console.log('🔧 DEBUG: *** EVENTO CLICK DISPARADO ***', event);
            event.preventDefault();
            
            // Teste simples primeiro
            console.log('🔧 TESTE: Botão clicado! JavaScript funcionando!');
            alert('Botão clicado! Iniciando criação do modelo...');
            
            console.log('🔧 DEBUG: Iniciando requisição diretamente');
            
            addLog('✨ Iniciando criação de modelo do zero...');
            createModelBtn.classList.add('loading');
            createModelBtn.disabled = true;
            
            console.log('🔧 DEBUG: Fazendo requisição POST para /create_model');
            fetch('/create_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: 'usbabc_model_ptbr',
                    vocab_size: 12000,
                    hidden_size: 512,
                    num_layers: 8,
                    num_heads: 8,
                    max_length: 2048
                })
            }).then(function(response) {
                console.log('🔧 DEBUG: Resposta recebida:', response.status);
                return response.json();
            }).then(function(result) {
                console.log('🔧 DEBUG: Resultado:', result);
                
                if (result.success) {
                    addLog('🚀 Criação iniciada! Aguarde...');
                } else {
                    throw new Error(result.error || 'Erro desconhecido');
                }
            }).catch(function(error) {
                console.error('🔧 DEBUG: Erro na requisição:', error);
                addLog('❌ Erro: ' + error.message);
            }).finally(function() {
                createModelBtn.classList.remove('loading');
                createModelBtn.disabled = false;
            });
        });
    } else {
        console.log('🔧 DEBUG: Botão criar modelo não encontrado!');
    }

    // ============================================================
    // CARREGAMENTO DO MODELO POR CAMINHO
    // ============================================================
    if (loadModelBtn && modelPath) {
        loadModelBtn.addEventListener('click', async () => {
            const path = modelPath.value.trim();
            if (!path) {
                console.log('❌ Digite o caminho do modelo!');
                modelPath.focus();
                return;
            }

            try {
                addLog(`📥 Carregando modelo: ${path}`);
                updateStatus('Carregando modelo...', false);
                loadModelBtn.disabled = true;
                loadModelBtn.innerHTML = '⏳ Carregando...';
                
                const response = await fetch('/load_model_path', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        model_path: path
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    appState.modelLoaded = true;
                    appState.modelPath = result.path;
                    document.getElementById('quantModelPath').value = result.path;
                    const fileName = result.path.split(/[/\\]/).pop().replace('.gguf', '');
                    document.getElementById('quantOutputPath').value = `modelos/${fileName}_q4_k_m.gguf`;
                    updateStatus('Modelo carregado', true);
                    
                    // Verificar status do chat após carregamento
                    setTimeout(async () => {
                        try {
                            const statusResponse = await fetch('/model_status');
                            const status = await statusResponse.json();
                            if (status.chat_available) {
                                enableChat();
                                addLog('✅ ✓ Modelo carregado e chat habilitado!');
                            } else {
                                addLog('⚠️ Modelo carregado mas chat não disponível');
                            }
                        } catch (e) {
                            console.error('Erro ao verificar status do chat:', e);
                            enableChat(); // Tentar habilitar mesmo assim
                            addLog('✅ ✓ Modelo carregado!');
                        }
                    }, 1000);
                    
                    checkTrainingReady();
                } else {
                    throw new Error(result.error || 'Erro ao carregar modelo');
                }
                
            } catch (error) {
                addLog(`❌ Erro: ${error.message}`);
                updateStatus('Erro no carregamento', false);
            } finally {
                loadModelBtn.disabled = false;
                loadModelBtn.innerHTML = '📂 Carregar Modelo';
            }
        });
    }

    // ============================================================
    // CARREGAMENTO DOS DADOS POR CAMINHO
    // ============================================================
    if (loadDataBtn && dataPath) {
        loadDataBtn.addEventListener('click', async () => {
            const path = dataPath.value.trim();
            if (!path) {
                console.log('❌ Digite o caminho dos dados!');
                dataPath.focus();
                return;
            }

            try {
                addLog(`📊 Carregando dados: ${path}`);
                loadDataBtn.disabled = true;
                loadDataBtn.innerHTML = '⏳ Carregando...';
                
                const response = await fetch('/load_data_path', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        data_path: path
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    appState.dataLoaded = true;
                    appState.dataPath = result.path;
                    appState.dataFile = result.filename;
                    checkTrainingReady();
                    addLog(`✅ ✓ Dados carregados: ${result.samples} amostras (${result.filename})`);
                } else {
                    throw new Error(result.error || 'Erro ao carregar dados');
                }
                
            } catch (error) {
                addLog(`❌ Erro: ${error.message}`);
            } finally {
                loadDataBtn.disabled = false;
                loadDataBtn.innerHTML = '📊 Carregar Dados';
            }
        });
    }

    // ============================================================
    // TREINAMENTO
    // ============================================================
    if (startTraining) {
        startTraining.addEventListener('click', async () => {
            if (!appState.modelLoaded || !appState.dataLoaded) {
                console.log('❌ Carregue um modelo e dados primeiro!');
                return;
            }

            try {
                appState.trainingActive = true;
                startTraining.disabled = true;
                startTraining.textContent = 'Treinando...';
                addLog('🚀 Iniciando treinamento...');

                const response = await fetch('/start_training', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        model_path: appState.modelPath,
                        data_file: appState.dataFile
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    addLog('✅ ✓ Treinamento iniciado!');
                } else {
                    throw new Error(result.error || 'Erro ao iniciar treinamento');
                }

            } catch (error) {
                addLog(`❌ Erro: ${error.message}`);
                appState.trainingActive = false;
                startTraining.disabled = false;
                startTraining.textContent = 'Iniciar Treinamento';
            }
        });
    }

    // ============================================================
    // SALVAR MODELO
    // ============================================================
    if (saveModel) {
        saveModel.addEventListener('click', async () => {
            const saveModelName = document.getElementById('savePath');
            const saveModelFormat = document.getElementById('saveFormat');
            
            if (!saveModelName || !saveModelFormat) {
                console.log('❌ Campos de salvamento não encontrados!');
                return;
            }

            const modelName = saveModelName.value.trim();
            const modelFormat = saveModelFormat.value;

            if (!modelName) {
                console.log('❌ Digite um nome para o modelo!');
                saveModelName.focus();
                return;
            }

            try {
                saveModel.disabled = true;
                saveModel.innerHTML = '<span class="btn-text">💾 Salvando...</span>';
                
                addLog(`💾 Salvando modelo "${modelName}" em formato ${modelFormat.toUpperCase()}...`);
                
                const response = await fetch('/save_model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        save_path: modelName,
                        save_format: modelFormat
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    addLog(`✅ Modelo salvo com sucesso!`);
                    addLog(`📁 Caminho: ${result.path}`);
                    addLog(`📦 Formato: ${result.format.toUpperCase()}`);
                    if (result.size_mb) {
                        addLog(`📊 Tamanho: ${result.size_mb} MB`);
                    }
                    
                    // Limpar campo nome
                    saveModelName.value = '';
                } else {
                    throw new Error(result.error || 'Erro ao salvar modelo');
                }

            } catch (error) {
                addLog(`❌ Erro ao salvar: ${error.message}`);
            } finally {
                saveModel.disabled = false;
                saveModel.innerHTML = '<span class="btn-text">💾 Salvar Modelo</span>';
            }
        });
    }

    // ============================================================
    // CHAT
    // ============================================================
    if (sendMessage && messageInput) {
        sendMessage.addEventListener('click', sendChatMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }

    async function sendChatMessage() {
        if (!appState.modelLoaded) {
            console.log('❌ Carregue um modelo primeiro!');
            return;
        }

        const message = messageInput.value.trim();
        if (!message) return;

        try {
            // Adicionar mensagem do usuário
            addChatMessage('user', message);
            messageInput.value = '';
            sendMessage.disabled = true;

            // Enviar para o servidor
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    model_path: appState.modelPath
                })
            });

            const result = await response.json();
            
            if (result.success) {
                addChatMessage('assistant', result.response);
            } else {
                throw new Error(result.error || 'Erro na resposta');
            }

        } catch (error) {
            addChatMessage('system', `❌ Erro: ${error.message}`);
        } finally {
            sendMessage.disabled = false;
        }
    }

    // ============================================================
    // QUANTIZAÇÃO
    // ============================================================
    const quantizeBtn = document.getElementById('startQuantization');
    if (quantizeBtn) {
        quantizeBtn.addEventListener('click', async () => {
            const modelPath = document.getElementById('quantModelPath').value;
            const outputPath = document.getElementById('quantOutputPath').value;
            const quantType = document.getElementById('quantType').value;

            if (!modelPath) {
                console.log('❌ Especifique o caminho do modelo!');
                return;
            }

            try {
                quantizeBtn.disabled = true;
                quantizeBtn.textContent = 'Quantizando...';
                addLog('🔧 Iniciando quantização...');

                const response = await fetch('/start_quantization', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        model_path: modelPath,
                        output_path: outputPath,
                        quantization_type: quantType
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    addLog(`✅ ✓ Quantização concluída: ${result.output_path}`);
                } else {
                    throw new Error(result.error || 'Erro na quantização');
                }

            } catch (error) {
                addLog(`❌ Erro: ${error.message}`);
            } finally {
                quantizeBtn.disabled = false;
                quantizeBtn.textContent = 'Quantizar Modelo';
            }
        });
    }

    // ============================================================
    // WEBSOCKET EVENTS
    // ============================================================
    socket.on('training_progress', (data) => {
        addLog(`📈 Progresso: ${data.message}`);
        if (data.progress) {
            updateProgress(data.progress);
        }
    });

    socket.on('training_complete', (data) => {
        addLog('🎉 Treinamento concluído!');
        appState.trainingActive = false;
        if (startTraining) {
            startTraining.disabled = false;
            startTraining.textContent = 'Iniciar Treinamento';
        }
        
        // PRIORIDADE 4: Habilitar botão "Salvar Modelo" após treinamento
        const saveModelBtn = document.getElementById('executeSave');
        if (saveModelBtn) {
            saveModelBtn.disabled = false;
            addLog('💾 ✓ Botão "Salvar Modelo" habilitado! Você pode salvar o modelo treinado agora.');
        }
    });

    socket.on('training_error', (data) => {
        addLog(`❌ Erro no treinamento: ${data.error}`);
        appState.trainingActive = false;
        if (startTraining) {
            startTraining.disabled = false;
            startTraining.textContent = 'Iniciar Treinamento';
        }
    });

    socket.on('model_created', (data) => {
        addLog('🎉 Modelo criado com sucesso!');
        if (data.path) {
            appState.modelLoaded = true;
            appState.modelPath = data.path;
            updateStatus('Modelo criado', true);
            enableChat();
        }
        const createBtn = document.getElementById('createModel');
        if (createBtn) {
            createBtn.classList.remove('loading');
            createBtn.disabled = false;
        }
    });

    socket.on('error', (data) => {
        addLog(`❌ Erro: ${data.message}`);
        const createBtn = document.getElementById('createModel');
        if (createBtn) {
            createBtn.classList.remove('loading');
            createBtn.disabled = false;
        }
    });

    // ============================================================
    // FUNÇÕES AUXILIARES
    // ============================================================
    function addLog(text) {
        const logDiv = document.createElement('div');
        logDiv.className = 'log-item';
        const timestamp = new Date().toLocaleTimeString();
        logDiv.textContent = `[${timestamp}] ${text}`;
        
        const logsContainer = document.getElementById('logs');
        if (logsContainer) {
            logsContainer.appendChild(logDiv);
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
    }

    function updateStatus(text, active) {
        if (modelStatus) {
            const statusSpan = modelStatus.querySelector('span:last-child');
            if (statusSpan) {
                statusSpan.textContent = text;
            }
            modelStatus.className = active ? 'status-indicator active' : 'status-indicator';
        }
    }

    function updateProgress(percent) {
        const progressBar = document.getElementById('progress');
        if (progressBar) {
            progressBar.style.width = `${percent}%`;
        }
    }

    function enableChat() {
        if (messageInput) messageInput.disabled = false;
        if (sendMessage) sendMessage.disabled = false;
        if (chatMessages) {
            chatMessages.innerHTML = '<div class="message system"><p>✅ ✓ Modelo carregado! Você pode conversar agora.</p></div>';
        }
    }

    function checkTrainingReady() {
        if (appState.modelLoaded && appState.dataLoaded && startTraining) {
            startTraining.disabled = false;
            addLog('✅ ✓ Pronto para treinar!');
        }
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    function updateModelInfo(filename, size) {
        const modelInfo = document.getElementById('modelInfo');
        if (modelInfo) {
            modelInfo.textContent = `${filename} (${formatBytes(size)})`;
            modelInfo.style.display = 'block';
        }
    }

    function updateDataInfo(filename, size, samples) {
        const dataInfo = document.getElementById('dataInfo');
        if (dataInfo) {
            dataInfo.textContent = `${filename} (${formatBytes(size)}, ${samples} amostras)`;
            dataInfo.style.display = 'block';
        }
    }

    function addChatMessage(role, content) {
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const contentP = document.createElement('p');
        contentP.textContent = content;
        messageDiv.appendChild(contentP);
        
        chatMessages.appendChild(messageDiv);
        
        // Limitar número de mensagens para melhorar performance
        const maxMessages = 100;
        const messages = chatMessages.querySelectorAll('.message');
        if (messages.length > maxMessages) {
            // Remover mensagens mais antigas
            const toRemove = messages.length - maxMessages;
            for (let i = 0; i < toRemove; i++) {
                messages[i].remove();
            }
        }
        
        // Scroll suave para o final
        requestAnimationFrame(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
    }

    // ============================================================
    // DRAG AND DROP
    // ============================================================
    ['modelUpload', 'dataUpload'].forEach(id => {
        const zone = document.getElementById(id);
        const input = id === 'modelUpload' ? modelFile : dataFile;

        if (zone && input) {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.style.borderColor = '#6366F1';
            });

            zone.addEventListener('dragleave', () => {
                zone.style.borderColor = '#e5e7eb';
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.style.borderColor = '#e5e7eb';
                const files = e.dataTransfer.files;
                if (files.length) {
                    input.files = files;
                    input.dispatchEvent(new Event('change'));
                }
            });
        }
    });

    // ============================================================
    // UPLOAD DE ARQUIVOS
    // ============================================================
    if (modelFile) {
        modelFile.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            console.log('🔧 DEBUG: Arquivo de modelo selecionado:', file.name);
            
            const formData = new FormData();
            formData.append('model', file);

            try {
                const response = await fetch('/upload_model', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (result.success) {
                    appState.modelLoaded = true;
                    appState.modelPath = result.path;
                    updateModelInfo(file.name, file.size);
                    addLog(`✅ Modelo carregado: ${file.name}`);
                    checkTrainingReady();
                } else {
                    throw new Error(result.error || 'Erro ao carregar modelo');
                }
            } catch (error) {
                console.error('❌ Erro ao carregar modelo:', error);
                addLog(`❌ Erro ao carregar modelo: ${error.message}`);
            }
        });
    }

    if (dataFile) {
        dataFile.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            console.log('🔧 DEBUG: Arquivo de dados selecionado:', file.name);
            
            const formData = new FormData();
            formData.append('data', file);

            try {
                const response = await fetch('/upload_data', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (result.success) {
                    appState.dataLoaded = true;
                    appState.dataPath = result.path;
                    appState.dataFile = result.filename;
                    updateDataInfo(file.name, file.size, result.samples);
                    addLog(`✅ Dados carregados: ${result.samples} amostras (${file.name})`);
                    checkTrainingReady();
                } else {
                    throw new Error(result.error || 'Erro ao carregar dados');
                }
            } catch (error) {
                console.error('❌ Erro ao carregar dados:', error);
                addLog(`❌ Erro ao carregar dados: ${error.message}`);
            }
        });
    }

    // ============================================================
    // WEB SCRAPING
    // ============================================================
    const startScrapingBtn = document.getElementById('startScraping');
    const searchQuery = document.getElementById('scrapingQuery');
    const numResults = document.getElementById('numResults');
    const scrapingStatus = document.getElementById('scrapingStatus');
    
    if (startScrapingBtn && searchQuery && numResults) {
        console.log('🔧 DEBUG: Elementos de web scraping encontrados');
        
        startScrapingBtn.addEventListener('click', async () => {
            console.log('🔧 DEBUG: Botão de web scraping clicado');
            
            const query = searchQuery.value.trim();
            const results = parseInt(numResults.value) || 10;
            
            if (!query) {
                console.log('❌ Por favor, insira uma query de busca!');
                return;
            }
            
            try {
                // Desabilitar botão e mostrar status
                startScrapingBtn.disabled = true;
                startScrapingBtn.classList.add('loading');
                scrapingStatus.style.display = 'block';
                scrapingStatus.innerHTML = '🔍 Iniciando busca na web...';
                scrapingStatus.style.backgroundColor = '#dbeafe';
                scrapingStatus.style.color = '#1e40af';
                
                addLog(`🔍 Iniciando web scraping: "${query}" (${results} resultados)`);
                
                const response = await fetch('/start_scraping', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: query,
                        num_results: results
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    scrapingStatus.innerHTML = '✅ Busca iniciada com sucesso! Aguarde...';
                    scrapingStatus.style.backgroundColor = '#dcfce7';
                    scrapingStatus.style.color = '#166534';
                    addLog('✅ Web scraping iniciado com sucesso');
                } else {
                    throw new Error(result.error || 'Erro desconhecido');
                }
                
            } catch (error) {
                console.error('Erro no web scraping:', error);
                scrapingStatus.innerHTML = `❌ Erro: ${error.message}`;
                scrapingStatus.style.backgroundColor = '#fef2f2';
                scrapingStatus.style.color = '#dc2626';
                addLog(`❌ Erro no web scraping: ${error.message}`);
            } finally {
                // Reabilitar botão após 3 segundos
                setTimeout(() => {
                    startScrapingBtn.disabled = false;
                    startScrapingBtn.classList.remove('loading');
                }, 3000);
            }
        });
    }
    
    // Socket events para web scraping
    socket.on('scraping_progress', (data) => {
        console.log('📡 Progresso do scraping:', data);
        if (scrapingStatus) {
            scrapingStatus.innerHTML = `🔍 ${data.message}`;
            scrapingStatus.style.backgroundColor = '#dbeafe';
            scrapingStatus.style.color = '#1e40af';
        }
        addLog(`🔍 ${data.message}`);
    });
    
    socket.on('scraping_complete', (data) => {
        console.log('📡 Scraping completo:', data);
        if (scrapingStatus) {
            scrapingStatus.innerHTML = `✅ Concluído! ${data.total_results} resultados processados. Arquivo: ${data.filename}`;
            scrapingStatus.style.backgroundColor = '#dcfce7';
            scrapingStatus.style.color = '#166534';
        }
        addLog(`✅ Web scraping concluído: ${data.total_results} resultados salvos em ${data.filename}`);
    });
    
    socket.on('scraping_error', (data) => {
        console.log('📡 Erro no scraping:', data);
        if (scrapingStatus) {
            scrapingStatus.innerHTML = `❌ Erro: ${data.error}`;
            scrapingStatus.style.backgroundColor = '#fef2f2';
            scrapingStatus.style.color = '#dc2626';
        }
        addLog(`❌ Erro no web scraping: ${data.error}`);
    });

    // ============================================================
    // INICIALIZAÇÃO
    // ============================================================
    window.addEventListener('load', async () => {
        try {
            const response = await fetch('/model_status');
            const status = await response.json();
            
            console.log('🔧 DEBUG: Status do modelo:', status);
            
            if (status.loaded) {
                appState.modelLoaded = true;
                appState.modelPath = status.path;
                const quantModelPath = document.getElementById('quantModelPath');
                if (quantModelPath) {
                    quantModelPath.value = status.path;
                }
                const fileName = status.path.split(/[/\\]/).pop().replace('.gguf', '');
                const quantOutputPath = document.getElementById('quantOutputPath');
                if (quantOutputPath) {
                    quantOutputPath.value = `modelos/${fileName}_q4_k_m.gguf`;
                }
                updateStatus('Modelo carregado', true);
                
                // Verificar se chat está disponível
                if (status.chat_available) {
                    enableChat();
                    addLog('✅ ✓ Modelo carregado e chat disponível');
                } else {
                    addLog('⚠️ Modelo carregado mas chat não disponível');
                }
            } else {
                addLog('ℹ️ Nenhum modelo carregado');
            }
        } catch (error) {
            console.error('Erro ao verificar status:', error);
            addLog('❌ Erro ao verificar status do modelo');
        }
        
        addLog('✅ Sistema pronto');
    });

    // ============================================================
    // CONVERSOR SAFETENSORS → GGUF
    // ============================================================
    const startConversion = document.getElementById('startConversion');
    const chooseSafetensorsPath = document.getElementById('chooseSafetensorsPath');
    const safetensorsPath = document.getElementById('safetensorsPath');
    const convertQuantType = document.getElementById('convertQuantType');
    const convertOutputPath = document.getElementById('convertOutputPath');
    const conversionStatus = document.getElementById('conversionStatus');
    
    // Elementos para carregamento de modelo principal
    const chooseModelBtn = document.getElementById('chooseModelBtn');
    const chooseQuantModelPath = document.getElementById('chooseQuantModelPath');
    const quantModelPath = document.getElementById('quantModelPath');

    if (chooseSafetensorsPath) {
        chooseSafetensorsPath.addEventListener('click', async () => {
            try {
                const response = await fetch('/open_file_dialog', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        file_types: ['safetensors', 'bin', 'pt', 'pth'],
                        title: 'Selecionar Modelo SafeTensors/PyTorch',
                        initial_dir: 'modelos'
                    })
                });
                const result = await response.json();
                if (result.success && result.path) {
                    safetensorsPath.value = result.path;
                    addLog(`✅ Modelo selecionado: ${result.name} (${(result.size / (1024**3)).toFixed(2)} GB)`);
                    
                    // Auto-gerar nome de saída
                    const baseName = result.name.replace(/\.(safetensors|bin|pt|pth)$/, '');
                    convertOutputPath.value = `modelos/${baseName}_converted.gguf`;
                    
                    // Validar modelo
                    validateSafeTensorsModel(result.path);
                } else {
                    addLog('❌ ' + (result.error || 'Nenhum arquivo selecionado'));
                }
            } catch (error) {
                console.error('Erro ao escolher arquivo:', error);
                addLog('❌ Erro ao escolher arquivo SafeTensors');
            }
        });
    }

    // Event listener para botão "PROC MOD" - carrega modelo diretamente
    if (chooseModelBtn) {
        chooseModelBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/open_file_dialog', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        file_types: ['gguf', 'safetensors', 'bin', 'pt', 'pth', 'zip'],
                        title: 'Selecionar Modelo de IA',
                        initial_dir: 'modelos'
                    })
                });
                const result = await response.json();
                if (result.success && result.path) {
                    addLog(`📁 Modelo selecionado: ${result.name} (${(result.size / (1024**3)).toFixed(2)} GB)`);
                    
                    // Carregar modelo diretamente
                    chooseModelBtn.disabled = true;
                    chooseModelBtn.innerHTML = '⏳<br>LOAD<br>ING';
                    
                    const loadResponse = await fetch('/load_model_path', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ model_path: result.path })
                    });
                    
                    const loadResult = await loadResponse.json();
                    if (loadResult.success) {
                        addLog('✅ ' + loadResult.message);
                        // Atualizar campo de quantização automaticamente
                        if (quantModelPath) {
                            quantModelPath.value = result.path;
                        }
                        // Atualizar info do modelo na área de drag and drop
                        const modelInfo = document.getElementById('modelInfo');
                        if (modelInfo) {
                            modelInfo.textContent = `✅ ${result.name}`;
                            modelInfo.style.color = '#4CAF50';
                        }
                    } else {
                        addLog('❌ ' + loadResult.error);
                    }
                } else {
                    addLog('❌ ' + (result.error || 'Nenhum arquivo selecionado'));
                }
            } catch (error) {
                console.error('Erro ao carregar modelo:', error);
                addLog('❌ Erro ao carregar modelo');
            } finally {
                chooseModelBtn.disabled = false;
                chooseModelBtn.innerHTML = '📁<br>PROC<br>MOD';
            }
        });
    }



    // Event listener para botão de procurar modelo para quantização
    if (chooseQuantModelPath) {
        chooseQuantModelPath.addEventListener('click', async () => {
            try {
                const response = await fetch('/open_file_dialog', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        file_types: ['gguf', 'safetensors', 'bin', 'pt', 'pth'],
                        title: 'Selecionar Modelo para Quantização',
                        initial_dir: 'modelos'
                    })
                });
                const result = await response.json();
                if (result.success && result.path) {
                    quantModelPath.value = result.path;
                    addLog(`✅ Modelo para quantização: ${result.name} (${(result.size / (1024**3)).toFixed(2)} GB)`);
                } else {
                    addLog('❌ ' + (result.error || 'Nenhum arquivo selecionado'));
                }
            } catch (error) {
                console.error('Erro ao escolher modelo:', error);
                addLog('❌ Erro ao abrir seletor de arquivo');
            }
        });
    }

    if (startConversion) {
        startConversion.addEventListener('click', async () => {
            const modelPath = safetensorsPath.value.trim();
            const outputPath = convertOutputPath.value.trim();
            const quantType = convertQuantType.value;

            if (!modelPath) {
                console.log('Por favor, selecione um modelo SafeTensors');
                return;
            }

            if (!outputPath) {
                console.log('Por favor, especifique o arquivo de saída');
                return;
            }

            try {
                startConversion.disabled = true;
                startConversion.innerHTML = '<span class="btn-text">🔄 Convertendo...</span>';
                
                if (conversionStatus) {
                    conversionStatus.style.display = 'block';
                    conversionStatus.innerHTML = '🔄 Iniciando conversão...';
                    conversionStatus.style.backgroundColor = '#fef3c7';
                    conversionStatus.style.color = '#92400e';
                }

                const response = await fetch('/convert_safetensors', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_path: modelPath,
                        output_path: outputPath,
                        quant_type: quantType
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    addLog(`✅ Conversão concluída: ${result.output_path}`);
                    if (result.size_mb) {
                        addLog(`📦 Tamanho do arquivo: ${result.size_mb.toFixed(1)} MB`);
                    }
                } else {
                    addLog(`❌ Erro na conversão: ${result.error}`);
                }

            } catch (error) {
                console.error('Erro na conversão:', error);
                addLog('❌ Erro na conversão SafeTensors → GGUF');
                
                if (conversionStatus) {
                    conversionStatus.innerHTML = `❌ Erro: ${error.message}`;
                    conversionStatus.style.backgroundColor = '#fef2f2';
                    conversionStatus.style.color = '#dc2626';
                }
            } finally {
                startConversion.disabled = false;
                startConversion.innerHTML = '<span class="btn-text">🔄 Converter para GGUF</span>';
            }
        });
    }

    // Função para validar modelo SafeTensors
    async function validateSafeTensorsModel(modelPath) {
        try {
            const response = await fetch('/validate_safetensors', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_path: modelPath })
            });
            
            const result = await response.json();
            
            if (result.valid) {
                addLog(`✅ Modelo válido: ${result.model_type || 'Desconhecido'}`);
                if (result.vocab_size) {
                    addLog(`📊 Vocabulário: ${result.vocab_size} tokens`);
                }
                if (!result.supported) {
                    addLog('⚠️ Arquitetura pode não ser totalmente suportada');
                }
            } else {
                addLog(`❌ Modelo inválido: ${result.error}`);
            }
        } catch (error) {
            console.error('Erro na validação:', error);
        }
    }

    // Socket events para conversão
    socket.on('conversion_progress', (data) => {
        console.log('🔄 Progresso conversão:', data);
        if (conversionStatus) {
            conversionStatus.innerHTML = `🔄 ${data.message} (${data.percent}%)`;
            conversionStatus.style.backgroundColor = '#fef3c7';
            conversionStatus.style.color = '#92400e';
        }
        addLog(`🔄 ${data.message} (${data.percent}%)`);
    });

    socket.on('conversion_complete', (data) => {
        console.log('✅ Conversão completa:', data);
        if (conversionStatus) {
            if (data.success) {
                conversionStatus.innerHTML = `✅ Conversão concluída! Arquivo: ${data.output_path}`;
                conversionStatus.style.backgroundColor = '#dcfce7';
                conversionStatus.style.color = '#166534';
                if (data.size_mb) {
                    conversionStatus.innerHTML += ` (${data.size_mb.toFixed(1)} MB)`;
                }
            } else {
                conversionStatus.innerHTML = `❌ Erro: ${data.error}`;
                conversionStatus.style.backgroundColor = '#fef2f2';
                conversionStatus.style.color = '#dc2626';
            }
        }
    });
});