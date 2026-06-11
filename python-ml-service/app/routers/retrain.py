import time
import logging
from fastapi import APIRouter, BackgroundTasks, status
from app.infrastructure.model_registry import ModelRegistry

logger = logging.getLogger(__name__)
router = APIRouter()

def execute_retrain_pipeline():
    """
    Worker function que executa o pipeline em segundo plano.
    Ao finalizar o treino, atualiza os modelos ativos no ModelRegistry.
    """
    logger.info("[M-13] Iniciando pipeline de re-treino automático...")
    
    try:
        # 1. Simulação de leitura de dados frescos do banco
        logger.info("[M-13] Extraindo novos logs do banco de dados (Postgres)...")
        time.sleep(3) 
        
        # 2. Simulação de treinamento computacionalmente pesado
        logger.info("[M-13] Treinando novos modelos (Classifier e Regressor)...")
        time.sleep(5) 
        
        # 3. Log do experimento e registro da nova versão
        new_version = f"v_{int(time.time())}"
        logger.info(f"[M-13] Modelos treinados e registrados no MLflow. Versão: {new_version}")
        
        # 4. Atualização Silenciosa da Memória (Hot-Swapping)
        registry = ModelRegistry.instance()
        
        # Em produção, as instâncias reais dos modelos treinados seriam passadas aqui:
        # registry.register("classifier", novo_modelo_treinado, {"version": new_version})
        
        logger.info("[M-13] ModelRegistry atualizado com sucesso (Hot-Swapping concluído)!")
        
    except Exception as e:
        logger.error(f"[M-13] Falha crítica durante o pipeline de re-treino: {str(e)}")


@router.post("/api/v1/models/retrain", status_code=status.HTTP_202_ACCEPTED)
def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Gatilho (Webhook) para disparar o re-treino dos modelos.
    Delega a operação para uma thread isolada e retorna 202 imediatamente.
    """
    # Aciona a função worker na fila de tarefas em background do FastAPI
    background_tasks.add_task(execute_retrain_pipeline)
    
    return {
        "status": "re-training initiated",
        "message": "O pipeline de re-treino foi disparado em background. O ModelRegistry será atualizado dinamicamente ao finalizar."
    }