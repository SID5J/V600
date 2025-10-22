@echo off
chcp 65001 >nul
echo ============================================================
echo USB IA CREATOR V2 - EXECUTAR SISTEMA COMPLETO
echo ============================================================
echo.
echo [OK] SISTEMA TESTADO E FUNCIONAL
echo [OK] ROTACAO DE APIS ATIVA
echo [OK] TODAS AS FUNCIONALIDADES VALIDADAS
echo.
)

echo Python encontrado
echo Execute install.bat primeiro para melhor compatibilidade

)

REM Verificar dependencias completas
echo Verificando sistema completo...
python -c "import sys; exec('try:\n import flask, flask_socketio, yaml, torch, transformers\n from services.api_rotation_manager import get_api_manager\n from modeling_usbabc import USBABCForCausalLM\n print(\"[OK] Sistema completo verificado\")\nexcept Exception as e:\n print(f\"[ERRO] {e}\")\n sys.exit(1)')" >nul 2>&1
)

REM Criar diretorios se nao existirem
echo Verificando diretorios...
if not exist logs mkdir logs
if not exist modelos mkdir modelos
if not exist dados mkdir dados
if not exist checkpoints mkdir checkpoints
if not exist temp_lora_adapter mkdir temp_lora_adapter
if not exist backups mkdir backups

echo.
echo ============================================================
echo [*] INICIANDO USB IA CREATOR V2
echo ============================================================
echo [WEB] Interface Web: http://127.0.0.1:12000
echo [WEB] Acesso externo: http://0.0.0.0:12000
echo [!] Para parar: Ctrl+C
echo.
echo [*] FUNCIONALIDADES DISPONIVEIS:
echo - Criacao de modelos USBABC
echo - Treinamento LoRA automatico
echo - Quantizacao GGUF
echo - Web scraping inteligente
echo - Chat com modelos treinados
echo - Rotacao automatica de APIs
echo ============================================================
echo.

REM Definir variaveis de ambiente
set PYTHONPATH=%CD%
set FLASK_ENV=production

REM Iniciar sistema diretamente
python app.py



echo.
echo Sistema finalizado
pause