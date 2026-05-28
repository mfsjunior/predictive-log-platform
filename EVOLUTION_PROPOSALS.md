# EVOLUTION PROPOSALS ? Predictive Log Intelligence Platform (PLIP)

> Propostas de evolu��o organizadas em m�dulos independentes para trabalho em duplas.
> Cada m�dulo tem escopo delimitado, arquivos afetados mapeados e crit�rios de aceita��o claros.
> Os m�dulos s�o ordenados por **depend�ncia t�cnica**: resolva d�bitos de base antes de features novas.

---

## COMO USAR ESTE DOCUMENTO

- Cada m�dulo � autossuficiente: uma dupla pode peg�-lo sem bloquear outra
- Leia o `PROJECT_CONTEXT.md` antes de iniciar qualquer m�dulo
- Siga o fluxo do `ENGINEERING_GUIDE.md` antes de codar
- Ao concluir um m�dulo, marque o status e registre decis�es no `PROJECT_CONTEXT.md` (se��o 12)

---

## �NDICE DE M�DULOS

| ID | M�dulo | Camada | Prioridade | Pr�-requisito |
|---|---|---|---|---|
| M-01 | GlobalExceptionHandler | Java API | CR�TICA | ? |
| M-02 | ModelRegistry Python | Python ML | CR�TICA | ? |
| M-03 | Mapper Layer (Java) | Java API | ALTA | M-01 |
| M-04 | CORS + Network Isolation | Infra / Python | ALTA | ? |
| M-05 | Pagina��o de Logs e Predi��es | Java API | ALTA | M-01, M-03 |
| M-06 | Soft Delete nas Entidades | Java API | M�DIA | M-03 |
| M-07 | Rate Limiting na API Java | Java API | M�DIA | M-01 |
| M-08 | Filtros Avan�ados de Estat�sticas | Java API + Python | M�DIA | M-05 |
| M-09 | Health Checks Detalhados | Infra / Java | M�DIA | ? |
| M-10 | Testes de Integra��o Java | Java API | ALTA | M-01, M-03 |
| M-11 | Testes Python ? Cobertura Router | Python ML | ALTA | M-02 |
| M-12 | Endpoint de Hist�rico de Predi��es | Java API | BAIXA | M-05 |
| M-13 | Re-treino Autom�tico Agendado | Python ML | BAIXA | M-02 |
| M-14 | Alertas via WebSocket | Python ML | BAIXA | M-02 |

---

---

## M-01 ? GlobalExceptionHandler (Java API)

**Prioridade**: CR�TICA
**Dupla**: Desenvolvimento backend Java
**Esfor�o estimado**: pequeno (1 sess�o)

### Contexto

Atualmente cada controller captura exce��es individualmente com `try/catch` manual:

```java
// LogController.java ? padr�o atual (repetido em 3 controllers)
try {
    ...
} catch (IllegalArgumentException e) {
    return ResponseEntity.badRequest().body(...);
} catch (Exception e) {
    log.error("Upload failed", e);
    return ResponseEntity.internalServerError().body(...);
}
```

Isso causa: duplica��o de c�digo, respostas de erro com formatos diferentes por controller, aus�ncia de tratamento de erros de valida��o Bean Validation (`@Valid`), e impossibilidade de centralizar log de exce��es.

### Objetivo

Criar um handler global `@RestControllerAdvice` que padronize **todas** as respostas de erro da API, e remover os `try/catch` dos controllers ap�s a cria��o.

### Arquivos a criar

```
java-api/src/main/java/com/logplatform/
??? exception/
    ??? GlobalExceptionHandler.java   ? @RestControllerAdvice
    ??? ApiErrorResponse.java         ? DTO padr�o de erro
    ??? BusinessException.java        ? exce��o de dom�nio base
    ??? ResourceNotFoundException.java ? 404 sem�ntico
```

### Arquivos a modificar

```
controller/LogController.java       ? remover try/catch, simplificar para 1 linha
controller/PredictController.java   ? remover try/catch, simplificar para 1 linha
controller/StatsController.java     ? sem altera��o (n�o tem try/catch)
```

### Contrato do DTO de erro (n�o alterar ap�s definido)

```json
{
  "timestamp": "2026-05-14T10:30:00",
  "status": 400,
  "error": "Bad Request",
  "message": "File must be a CSV file",
  "path": "/logs/upload"
}
```

### Crit�rios de aceita��o

- [ ] `GET /stats/summary` com banco vazio retorna `200` com dados zerados (comportamento atual preservado)
- [ ] `POST /logs/upload` com arquivo n�o-CSV retorna `400` com body no formato `ApiErrorResponse`
- [ ] `POST /logs/upload` com arquivo v�lido continua retornando `200` (sem regress�o)
- [ ] `POST /predict/error` com body inv�lido retorna `400` com mensagem de valida��o
- [ ] Nenhum controller cont�m `try/catch` ap�s o refactor
- [ ] Testes unit�rios dos controllers passam sem altera��o de l�gica

---

---

## M-02 ? ModelRegistry Centralizado (Python ML)

**Prioridade**: CR�TICA
**Dupla**: Desenvolvimento Python / ML
**Esfor�o estimado**: pequeno-m�dio (1?2 sess�es)

### Contexto

O `predict.py` acessa o modelo treinado via importa��o direta de vari�vel do m�dulo `train.py`:

```python
# predict.py ? problema atual
from app.routers.train import classifier_pipeline  # estado global mut�vel
```

Isso cria acoplamento entre routers, risco de race condition em requisi��es concorrentes, e impede testes isolados.

O projeto j� possui `app/infrastructure/model_registry.py` ? mas ainda n�o � usado consistentemente por todos os routers.

### Objetivo

Garantir que `ModelRegistry` seja a **�nica fonte de verdade** dos modelos carregados, eliminar todas as importa��es de vari�veis globais entre routers, e validar com testes.

### Arquivos a verificar/completar

```
python-ml-service/app/
??? infrastructure/
?   ??? model_registry.py       ? verificar implementa��o atual e completar se necess�rio
??? routers/
?   ??? predict.py              ? substituir import global por ModelRegistry.instance()
?   ??? anomaly.py              ? idem
?   ??? train.py                ? garantir que persiste modelos via registry ap�s treino
```

### Padr�o esperado ap�s o m�dulo

```python
# predict.py ? padr�o correto
from app.infrastructure.model_registry import ModelRegistry

@router.post("/predict/error", response_model=ErrorPredictionResponse)
async def predict_error(request: ErrorPredictionRequest):
    registry = ModelRegistry.instance()
    classifier = registry.get_classifier()
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modelo n�o treinado.")
    ...
```

### Crit�rios de aceita��o

- [ ] Nenhum router importa vari�veis diretamente de outro router
- [ ] `POST /predict/error` sem modelo treinado retorna `503` com mensagem clara
- [ ] `POST /train` seguido de `POST /predict/error` funciona corretamente
- [ ] `ModelRegistry` � singleton (n�o recria inst�ncia a cada request)
- [ ] Testes em `test_predict.py` n�o precisam mais do `fixture` que altera `train.classifier_pipeline` diretamente

---

---

## M-03 ? Camada de Mapper (Java API)

**Prioridade**: ALTA
**Dupla**: Desenvolvimento backend Java
**Pr�-requisito**: M-01 conclu�do
**Esfor�o estimado**: m�dio (2 sess�es)

### Contexto

Atualmente a convers�o entre `WebLog` (entity JPA) e DTOs pode estar ocorrendo dentro dos services ou de forma inline. N�o existe camada de mapper expl�cita, o que viola SRP e acopla o contrato de API ao modelo de banco.

### Objetivo

Criar mappers dedicados para cada entidade, desacoplando a evolu��o do schema do banco da evolu��o dos contratos de API.

### Arquivos a criar

```
java-api/src/main/java/com/logplatform/
??? mapper/
    ??? WebLogMapper.java         ? WebLog ? WebLogDomain ? DTO
    ??? PredictionMapper.java     ? Prediction ? PredictionResult ? DTO
```

### Conven��o do mapper

Mappers s�o classes `@Component` com m�todos est�ticos ou de inst�ncia. **N�o usar MapStruct neste momento** para n�o adicionar depend�ncia desnecess�ria ? implementar manualmente seguindo o padr�o existente do projeto.

```java
@Component
public class WebLogMapper {

    public WebLogDomain toDomain(WebLog entity) { ... }

    public WebLog toEntity(WebLogDomain domain) { ... }

    public WebLogResponse toResponse(WebLogDomain domain) { ... }
}
```

### Crit�rios de aceita��o

- [ ] `LogIngestionService` n�o manipula campos de `WebLog` diretamente em l�gica de neg�cio
- [ ] `StatisticsService` n�o retorna campos de entity JPA nos DTOs de resposta
- [ ] Mappers possuem testes unit�rios com casos de campo nulo
- [ ] Nenhum campo de `WebLog` (ex: `createdAt` interno) vaza para DTOs de resposta p�blica

---

---

## M-04 ? CORS Restrito + Isolamento de Rede (Infra / Python)

**Prioridade**: ALTA
**Dupla**: Infraestrutura / seguran�a
**Esfor�o estimado**: pequeno (1 sess�o)

### Contexto

O Python ML Service tem CORS aberto (`allow_origins=["*"]`) e n�o possui autentica��o. Em produ��o, o servi�o deve ser acess�vel **apenas** pela Java API dentro da rede Docker.

### Objetivo

1. Restringir CORS no Python para aceitar apenas origens conhecidas
2. Garantir que o servi�o Python n�o seja exposto publicamente no `docker-compose.yml`
3. Adicionar valida��o de `X-Internal-Request` header para bloquear chamadas diretas externas

### Arquivos a modificar

```
python-ml-service/app/main.py        ? ajustar allow_origins para vari�vel de ambiente
python-ml-service/app/config.py      ? adicionar CORS_ORIGINS: list[str]
docker-compose.yml                   ? remover bind 0.0.0.0 do python-ml (porta s� interna)
```

### Configura��o esperada

```python
# config.py
CORS_ORIGINS: list[str] = Field(
    default=["http://java-api:8080", "http://localhost:8080"],
    description="Allowed CORS origins"
)
```

```python
# main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # n�o mais ["*"]
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

### Crit�rios de aceita��o

- [ ] `docker compose up` sobe todos os servi�os sem erro
- [ ] Java API consegue chamar `POST /predict/error` no Python normalmente
- [ ] Porta `8000` n�o est� vinculada a `0.0.0.0` no ambiente de produ��o (apenas interna Docker)
- [ ] Vari�vel `CORS_ORIGINS` sobrep�e o default via vari�vel de ambiente no `docker-compose.yml`

---

---

## M-05 ? Pagina��o de Logs e Predi��es (Java API)

**Prioridade**: ALTA
**Dupla**: Desenvolvimento backend Java
**Pr�-requisito**: M-01, M-03
**Esfor�o estimado**: m�dio (2 sess�es)

### Contexto

N�o h� endpoint para listar logs ou hist�rico de predi��es com pagina��o. O `StatisticsService` faz `webLogRepository.count()` e queries de agrega��o ? sem risco de OOM. Mas futuras queries de listagem sem pagina��o representam risco real.

### Objetivo

Criar endpoints de listagem paginada para `WebLog` e `Prediction`, usando `Pageable` do Spring Data.

### Arquivos a criar

```
controller/LogQueryController.java      ? GET /logs?page=0&size=20&sort=timestamp,desc
controller/PredictionQueryController.java ? GET /predictions?page=0&size=20
dto/PagedResponse.java                   ? wrapper gen�rico de pagina��o
dto/WebLogResponse.java                  ? resposta p�blica de um log (sem createdAt interno)
dto/PredictionResponse.java              ? resposta p�blica de uma predi��o
```

### Arquivos a modificar

```
repository/WebLogRepository.java       ? adicionar query paginada com filtros
repository/PredictionRepository.java   ? idem
```

### Contrato esperado

```
GET /logs?page=0&size=20&sort=timestamp,desc&method=GET&statusCode=500
```

```json
{
  "content": [...],
  "page": 0,
  "size": 20,
  "totalElements": 1500,
  "totalPages": 75,
  "last": false
}
```

### Crit�rios de aceita��o

- [ ] `GET /logs` sem par�metros retorna p�gina 0 com 20 itens
- [ ] `GET /logs?size=200` � rejeitado com `400` (size m�ximo = 100)
- [ ] Filtro por `method` e `statusCode` funciona corretamente
- [ ] `GET /predictions?page=1&size=10` retorna a segunda p�gina
- [ ] Nenhuma query carrega todos os registros em mem�ria
- [ ] Endpoint documentado no Swagger com exemplos

---

---

## M-06 ? Soft Delete nas Entidades (Java API)

**Prioridade**: M�DIA
**Dupla**: Desenvolvimento backend Java
**Pr�-requisito**: M-03
**Esfor�o estimado**: pequeno (1 sess�o)

### Contexto

`WebLog` e `Prediction` n�o possuem campo de dele��o l�gica. Qualquer `DELETE` remove o dado fisicamente, comprometendo auditoria.

### Objetivo

Adicionar `deletedAt` e `@SQLRestriction` (Hibernate 6) para que queries autom�ticas do Spring Data excluam registros deletados logicamente.

### Arquivos a modificar

```
entity/WebLog.java          ? adicionar LocalDateTime deletedAt
entity/Prediction.java      ? idem
repository/WebLogRepository.java     ? adicionar deleteById l�gico
repository/PredictionRepository.java ? idem
```

### Migration SQL necess�ria

```sql
-- Adicionar em postgres/migrations/V2__add_soft_delete.sql
ALTER TABLE web_logs ADD COLUMN deleted_at TIMESTAMP DEFAULT NULL;
ALTER TABLE predictions ADD COLUMN deleted_at TIMESTAMP DEFAULT NULL;
```

### Padr�o esperado na entidade

```java
@SQLRestriction("deleted_at IS NULL")   // Hibernate 6 ? filtra automaticamente queries
@Column(name = "deleted_at")
private LocalDateTime deletedAt;
```

### Crit�rios de aceita��o

- [ ] `DELETE /logs/{id}` preenche `deleted_at` sem remover o registro
- [ ] `GET /logs` n�o retorna registros com `deleted_at` preenchido
- [ ] Contagem estat�stica (`/stats/summary`) exclui registros deletados
- [ ] Registro f�sico permanece no banco para auditoria

---

---

## M-07 ? Rate Limiting na API Java

**Prioridade**: M�DIA
**Dupla**: Seguran�a / backend Java
**Pr�-requisito**: M-01
**Esfor�o estimado**: pequeno-m�dio (1?2 sess�es)

### Contexto

N�o h� prote��o contra abuso de endpoints. Um cliente pode fazer upload de CSV ilimitado ou esgotar o servi�o ML com predi��es em loop.

### Objetivo

Implementar rate limiting por IP usando `Bucket4j` com Redis como backend distribu�do (compat�vel com m�ltiplas inst�ncias).

### Depend�ncia a adicionar no `pom.xml`

```xml
<dependency>
    <groupId>com.bucket4j</groupId>
    <artifactId>bucket4j-core</artifactId>
    <version>8.10.1</version>
</dependency>
<dependency>
    <groupId>com.bucket4j</groupId>
    <artifactId>bucket4j-redis</artifactId>
    <version>8.10.1</version>
</dependency>
```                 

### Arquivos a criar

```
config/RateLimitConfig.java            ? configura��o dos buckets por endpoint
infrastructure/adapter/RateLimitFilter.java ? OncePerRequestFilter com l�gica de bucket
exception/RateLimitExceededException.java   ? 429 Too Many Requests
```

### Limites esperados

| Endpoint | Limite | Janela |
|---|---|---|
| `POST /logs/upload` | 5 req | por IP / minuto |
| `POST /predict/error` | 30 req | por IP / minuto |
| `POST /predict/response-time` | 30 req | por IP / minuto |
| `POST /auth/login` | 10 req | por IP / minuto |

### Crit�rios de aceita��o

- [ ] 6� chamada ao `/logs/upload` em 1 minuto retorna `429`
- [ ] Header `X-RateLimit-Remaining` presente em todas as respostas
- [ ] Header `X-RateLimit-Reset` indica quando o bucket recarrega
- [ ] Limites configur�veis via `application.yml` (n�o hardcoded)
- [ ] Rate limit reseta corretamente ap�s a janela de tempo

---

---

## M-08 ? Filtros Avan�ados de Estat�sticas (Java + Python)

**Prioridade**: M�DIA
**Dupla**: Desenvolvimento fullstack (Java + Python)
**Pr�-requisito**: M-05
**Esfor�o estimado**: m�dio (2?3 sess�es)

### Contexto

`GET /stats/summary` retorna um resumo est�tico de **todos** os logs. N�o � poss�vel filtrar por per�odo, m�todo HTTP ou faixa de status code.

### Objetivo

Adicionar suporte a filtros de per�odo e dimens�es nas estat�sticas, sem quebrar o endpoint atual.

### Parte Java ? Novo endpoint (n�o substituir o atual)

```
GET /stats/summary?from=2026-05-01&to=2026-05-14&method=GET
```

Arquivos a criar/modificar:
```
dto/StatsFilterRequest.java         ? par�metros de filtro (query params)
dto/StatsSummary.java               ? adicionar campo period (opcional)
repository/WebLogRepository.java    ? queries JPQL com filtros opcionais
service/StatisticsService.java      ? overload de computeSummary(StatsFilterRequest)
```

### Parte Python ? Estat�sticas por segmento

```
GET /stats/by-hour         ? distribui��o por hora do dia
GET /stats/by-method       ? agrupamento por m�todo HTTP
GET /stats/by-status-class ? 2xx, 3xx, 4xx, 5xx
```

Arquivos a criar:
```
python-ml-service/app/routers/stats.py    ? novo router
```

### Crit�rios de aceita��o

- [ ] `GET /stats/summary` sem filtro continua funcionando (sem regress�o)
- [ ] `GET /stats/summary?from=2026-05-01&to=2026-05-14` retorna apenas dados do per�odo
- [ ] `GET /stats/summary?method=POST` filtra corretamente
- [ ] Cache Redis invalida quando filtros s�o diferentes (chave inclui par�metros)
- [ ] Python retorna distribui��o por hora do dia corretamente

---

---

## M-09 ? Health Checks Detalhados (Infra / Java)

**Prioridade**: M�DIA
**Dupla**: Infraestrutura / SRE
**Esfor�o estimado**: pequeno (1 sess�o)

### Contexto

`/actuator/health` retorna `{"status": "UP"}` gen�rico. N�o indica sa�de individual de depend�ncias (PostgreSQL, Redis, Kafka, Python ML).

### Objetivo

Criar health indicators customizados para cada depend�ncia, expondo um endpoint rico que o Grafana e o Prometheus possam usar para alertas de degrada��o parcial.

### Arquivos a criar

```
config/HealthConfig.java                         ? registrar beans de health
infrastructure/health/
    ??? MlServiceHealthIndicator.java            ? chama GET /health do Python
    ??? KafkaHealthIndicator.java                ? verifica conex�o com broker
    ??? RedisHealthIndicator.java                ? ping Redis
```

### Resposta esperada

```json
GET /actuator/health

{
  "status": "UP",
  "components": {
    "db": { "status": "UP", "details": { "database": "PostgreSQL 16" } },
    "redis": { "status": "UP" },
    "kafka": { "status": "UP", "details": { "topics": ["log.events"] } },
    "mlService": { "status": "UP", "details": { "url": "http://python-ml:8000" } },
    "diskSpace": { "status": "UP" }
  }
}
```

### Crit�rios de aceita��o

- [ ] Quando Python ML est� down, `mlService.status` � `DOWN` mas `overall.status` � `DEGRADED` (n�o `DOWN`)
- [ ] Quando Kafka est� down, o servi�o Java continua funcionando com `kafka.status: DOWN`
- [ ] Endpoint documentado e acess�vel sem autentica��o
- [ ] Timeout de verifica��o do Python ML: m�ximo 2s para n�o travar o health check

---

---

## M-10 ? Testes de Integra��o Java (Testcontainers)

**Prioridade**: ALTA
**Dupla**: Qualidade / backend Java
**Pr�-requisito**: M-01, M-03
**Esfor�o estimado**: m�dio-grande (2?3 sess�es)

### Contexto

Existem testes de controller com `@WebMvcTest` mas sem banco real. O `IntegrationTest.java` existe mas n�o est� completo. N�o h� testes que verifiquem o fluxo completo: upload CSV ? salvar banco ? consultar estat�sticas.

### Objetivo

Criar testes de integra��o com Testcontainers (PostgreSQL + Redis) que cubram os fluxos cr�ticos end-to-end na camada de API Java.

### Arquivos a criar

```
src/test/java/com/logplatform/
??? integration/
?   ??? LogIngestionIntegrationTest.java    ? upload CSV ? verifica banco
?   ??? StatisticsIntegrationTest.java      ? ingest�o ? computeSummary()
?   ??? PredictionAuditIntegrationTest.java ? predi��o salva no banco
??? fixture/
?   ??? WebLogFixture.java                 ? factory de objetos de teste
?   ??? CsvFixture.java                    ? CSVs v�lidos e inv�lidos para teste
??? config/
    ??? TestContainersConfig.java           ? @SpringBootTest + Testcontainers
```

### Depend�ncias j� presentes no `pom.xml`

```xml
<!-- testcontainers.version=1.19.5 j� declarada no pom.xml -->
<dependency>
    <groupId>org.testcontainers</groupId>
    <artifactId>postgresql</artifactId>
    <scope>test</scope>
</dependency>
```

### Crit�rios de aceita��o

- [ ] `mvn test` executa todos os testes de integra��o sem banco local instalado
- [ ] Fluxo completo: upload de `data/web_logs.csv` ? `count()` > 0 no banco de teste
- [ ] `computeSummary()` ap�s ingest�o retorna `totalRecords > 0`
- [ ] Upload de arquivo n�o-CSV retorna `400` via integra��o real
- [ ] Cobertura de linha dos services >= 70% (JaCoCo)

---

---

## M-11 ? Cobertura de Testes Python (Routers Cr�ticos)

**Prioridade**: ALTA
**Dupla**: Qualidade / Python ML
**Pr�-requisito**: M-02
**Esfor�o estimado**: m�dio (2 sess�es)

### Contexto

Existem `test_predict.py`, `test_anomaly.py` e `test_pipeline.py`, mas os testes dependem de importar `train.classifier_pipeline` diretamente (estado global). Ap�s M-02, os testes precisam ser adaptados. Tamb�m n�o h� testes para os routers `monitor.py` e `websocket.py`.

### Objetivo

Adaptar testes existentes para usar `ModelRegistry`, adicionar testes para `monitor.py` e garantir cobertura >= 80% nos routers cr�ticos.

### Arquivos a criar/modificar

```
python-ml-service/tests/
??? test_predict.py       ? adaptar para ModelRegistry (remover acesso a train.*)
??? test_anomaly.py       ? idem
??? test_pipeline.py      ? idem
??? test_monitor.py       ? NOVO: testa drift detection com dados mock
??? test_train.py         ? NOVO: testa pipeline de treino isolado
??? conftest.py           ? NOVO: fixtures compartilhadas (modelos treinados, DB mock)
```

### Padr�o de fixture esperado

```python
# conftest.py
@pytest.fixture(scope="session")
def trained_registry():
    """Registry com modelos treinados para reuso entre testes."""
    registry = ModelRegistry.instance()
    df = generate_synthetic_dataset(n_records=500, seed=42)
    # ... treino ...
    return registry
```

### Crit�rios de aceita��o

- [ ] `pytest tests/ -v` executa sem acessar banco real (mocks de DB)
- [ ] Nenhum teste acessa `train.classifier_pipeline` diretamente
- [ ] `test_monitor.py` testa `GET /monitor/drift` com dados de refer�ncia e dados atuais
- [ ] Cobertura >= 80% nos routers `predict.py`, `train.py` e `anomaly.py`
- [ ] `pytest --cov=app tests/` reporta cobertura

---

---

## M-12 ? Endpoint de Hist�rico de Predi��es (Java API)

**Prioridade**: BAIXA
**Dupla**: Desenvolvimento backend Java
**Pr�-requisito**: M-05
**Esfor�o estimado**: pequeno (1 sess�o)

### Contexto

A tabela `predictions` j� existe no banco com `input_data` (JSONB) e `result` (JSONB), e o `PredictionService` j� salva cada predi��o. N�o h� endpoint para consultar esse hist�rico.

### Objetivo

Expor o hist�rico de predi��es com pagina��o e filtro por tipo (`error` / `response-time`).

### Arquivos a criar

```
controller/PredictionQueryController.java   ? GET /predictions
dto/PredictionResponse.java                 ? resposta p�blica (sem dados sens�veis internos)
```

### Contrato esperado

```
GET /predictions?type=error&page=0&size=20
```

```json
{
  "content": [
    {
      "id": 1,
      "predictionType": "error",
      "inputData": { "method": "GET", "hour": 14, ... },
      "result": { "errorProbability": 0.12, "riskLevel": "LOW" },
      "modelVersion": "v1.2.3",
      "latencyMs": 3.4,
      "createdAt": "2026-05-14T10:30:00"
    }
  ],
  "page": 0,
  "totalElements": 230
}
```

### Crit�rios de aceita��o

- [ ] `GET /predictions` retorna lista paginada
- [ ] Filtro `?type=error` retorna apenas predi��es de erro
- [ ] `inputData` e `result` s�o desserializados corretamente do JSONB
- [ ] Endpoint requer autentica��o JWT
- [ ] Documentado no Swagger com exemplos de resposta

---

---

## M-13 ? Re-treino Autom�tico Agendado (Python ML)

**Prioridade**: BAIXA
**Dupla**: ML Engineering / Python
**Pr�-requisito**: M-02
**Esfor�o estimado**: m�dio (2 sess�es)

### Contexto

O `scheduler.py` existe mas n�o est� integrado ao pipeline de re-treino. O modelo s� � retreinado quando algu�m chama `POST /train` manualmente.

### Objetivo

Configurar o scheduler para disparar `POST /train` automaticamente quando:
1. O volume de novos dados desde o �ltimo treino ultrapassar um threshold configur�vel
2. O drift detectado pelo Evidently AI ultrapassar um score configur�vel

### Arquivos a modificar/criar

```
python-ml-service/app/
??? scheduler.py                        ? completar com APScheduler jobs
??? infrastructure/
?   ??? retrain_trigger.py              ? l�gica de decis�o de re-treino
??? config.py                           ? RETRAIN_THRESHOLD_RECORDS, DRIFT_THRESHOLD_SCORE
```

### Crit�rios de aceita��o

- [ ] Scheduler inicia junto com a aplica��o FastAPI (via `lifespan`)
- [ ] Verifica��o de threshold acontece a cada N minutos (configur�vel)
- [ ] Re-treino n�o bloqueia a API durante execu��o (task ass�ncrona)
- [ ] Log estruturado registra motivo do re-treino (`"reason": "drift_detected"`)
- [ ] Re-treino manual via `POST /train` continua funcionando normalmente

---

---

## M-14 ? Alertas via WebSocket (Python ML)

**Prioridade**: BAIXA
**Dupla**: Fullstack / Python
**Pr�-requisito**: M-02
**Esfor�o estimado**: m�dio (2 sess�es)

### Contexto

O `websocket.py` existe mas os alertas n�o s�o disparados automaticamente quando anomalias s�o detectadas ou quando o drift ultrapassa thresholds.

### Objetivo

Fazer o WebSocket emitir alertas em tempo real quando:
- Uma predi��o retorna `risk_level == "CRITICAL"`
- O score de drift ultrapassa `0.3`
- Um batch de anomalias detecta > 5% de outliers

### Arquivos a modificar

```
python-ml-service/app/
??? routers/websocket.py    ? implementar broadcast de alertas
??? routers/predict.py      ? publicar alerta em CRITICAL predictions
??? routers/anomaly.py      ? publicar alerta quando outlier rate > threshold
??? routers/monitor.py      ? publicar alerta quando drift score > threshold
```

### Crit�rios de aceita��o

- [ ] Cliente WebSocket conectado em `ws://localhost:8000/ws/alerts` recebe mensagem quando predi��o � CRITICAL
- [ ] Formato da mensagem: `{"type": "alert", "level": "CRITICAL", "source": "predict_error", "timestamp": "...", "detail": {...}}`
- [ ] Clientes desconectados n�o causam erro no servidor
- [ ] WebSocket n�o bloqueia o pipeline de predi��o (fire-and-forget)

---

---

## ATRIBUI��O DAS 5 DUPLAS ? 3 SPRINTS PARALELOS

> Regra: **toda dupla termina o sprint atual antes de avan�ar para o pr�ximo.**
> Sprint 1 � a base de todos ? ningu�m avan�a sem ele estar pronto.

| Dupla | Sprint 1 ? Funda��o (todos fazem) | Sprint 2 ? Constru��o | Sprint 3 ? Features |
|---|---|---|---|
| **Geovana, Hugo e Lucas ** | M-01 GlobalExceptionHandler + M-02 ModelRegistry Python | M-07 Rate Limiting | M-13 Re-treino Autom�tico |
| **Agenor e Arthur** | M-01 GlobalExceptionHandler + M-02 ModelRegistry Python | M-03 Mapper Layer | M-10 Testes Integra��o Java |
| **Matheus e Matheus** | M-01 GlobalExceptionHandler + M-02 ModelRegistry Python | M-04 CORS + Network | M-14 Alertas WebSocket |
| **Marcos e Jo�o Paulo 
** | M-01 GlobalExceptionHandler + M-02 ModelRegistry Python | M-09 Health Checks + M-06 Soft Delete | M-05 Pagina��o |
| **Gabriel e Gabriel ** | M-01 GlobalExceptionHandler + M-02 ModelRegistry Python | M-11 Testes Python | M-08 Filtros Estat�sticas + M-12 Hist�rico de Predi��es |

> Sprint 1 � igual para todas as duplas: as duas entregas base sem as quais nada mais funciona.
