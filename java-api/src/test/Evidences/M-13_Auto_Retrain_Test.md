# Documentação de Teste e Arquitetura: Re-treino Automático (M-13)

Este documento serve como evidência técnica da conclusão da tarefa M-13 (Sprint 1), focada na implementação do pipeline de re-treino automático e assíncrono dos modelos de Machine Learning no serviço Python (FastAPI).

## 1. Resumo da Arquitetura
Para garantir que os modelos possam ser atualizados com novos dados sem interromper a disponibilidade do sistema, foi adotada uma arquitetura orientada a endpoints (estilo Webhook) com processamento assíncrono. A API expõe uma rota que aceita a requisição imediatamente e delega a execução pesada do treino para um operário (*worker*) em segundo plano utilizando `BackgroundTasks` nativo do FastAPI.

## 2. Modificações no Código-Fonte

### 2.1. Criação do Router de Re-treino (`retrain.py`)
O endpoint `POST /api/v1/models/retrain` foi isolado na camada de rotas.
* **Gatilho Assíncrono:** O método recebe a chamada e adiciona a tarefa `execute_retrain_pipeline` à fila de segundo plano.
* **Resposta Imediata:** Devolve o HTTP Status Code **`202 Accepted`** ao cliente em poucos milissegundos, evitando *timeouts* de rede enquanto o modelo processa.

### 2.2. Integração com o Model Registry (Hot-Swapping)
Assim que o pipeline termina o treino em background e registra as métricas no MLflow, ele aciona o Singleton `ModelRegistry.instance()`. O novo modelo substitui o antigo na memória de forma *Thread-Safe*, garantindo que as próximas predições já utilizem a versão atualizada instantaneamente.

## 3. Protocolo de Teste End-to-End (E2E)
Validação do comportamento assíncrono efetuada através do Postman e monitoramento dos logs em tempo real na console do Docker:

| Etapa | Ação Executada | Resultado Obtido |
| :--- | :--- | :--- |
| **1. Disparo do Gatilho** | `POST http://localhost:8000/api/v1/models/retrain` | **Status 202 Accepted.** A API respondeu imediatamente com o JSON confirmando a inicialização do processo em background. |
| **2. Execução Assíncrona** | Monitoramento via `docker compose logs -f` | **Sucesso.** O terminal demonstrou os passos do pipeline sendo executados em background enquanto a API continuava livre para receber novas predições. |
| **3. Hot-Swapping** | Verificação pós-treino no Registry | **Sucesso.** O modelo em memória foi substituído silenciosamente pela nova versão gerada sem necessidade de reiniciar o container. |

## 4. Conclusão
A esteira de MLOps cumpre todos os requisitos de automação da tarefa M-13. O ciclo de vida do modelo foi automatizado de ponta a ponta de forma resiliente, assíncrona e sem impacto no tempo de resposta para os usuários finais. A tarefa está formalmente homologada.