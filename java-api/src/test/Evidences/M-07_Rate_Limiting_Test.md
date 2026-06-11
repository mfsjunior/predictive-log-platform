# Documentação de Teste e Arquitetura: Rate Limiting (M-07)

Este documento serve como evidência técnica da conclusão da tarefa M-07 (Sprint 1), focada na proteção da API Spring Boot (Java 21) da Predictive Log Platform contra abusos, tráfego malicioso e ataques de força bruta.

## 1. Resumo da Arquitetura
Para evitar a sobrecarga do servidor e proteger a porta de entrada do sistema (especialmente as rotas de autenticação), foi implementado um limitador de taxa (*Rate Limiting*) global. A solução arquitetural utiliza a biblioteca **Bucket4j**, que implementa o algoritmo de *Token Bucket* (Balde de Fichas).

## 2. Modificações no Código-Fonte

### 2.1. O Filtro Global Interceptador (`RateLimitFilter.java`)
Foi criado um filtro que estende `OncePerRequestFilter`, anotado com `@Component` para interceptar todas as requisições antes que estas alcancem os controladores.
* **Capacidade do Balde:** 20 tokens (requisições) permitidos por minuto.
* **Taxa de Reposição:** Reposição gradual utilizando a estratégia *Greedy*.

## 3. Protocolo de Teste End-to-End (E2E)
O teste de estresse foi executado localmente através do Postman, utilizando um payload otimizado de array vazio (`[]`) para simular requisições consecutivas rápidas e validar as duas camadas de defesa da API.

| Etapa | Ação (Postman) | Resultado Obtido |
| :--- | :--- | :--- |
| **1. Validação de Carga (M-01)** | `POST http://localhost:8080/auth/login` <br> Payload: `[]` *(Cliques 1 a 20)* | **Status 400 Bad Request.** O Spring Boot interceptou a estrutura incorreta na camada de validação sintática antes do processamento de negócio. |
| **2. Ativação do Rate Limiting (M-07)** | `POST http://localhost:8080/auth/login` <br> Payload: `[]` *(Clique 21+ no mesmo minuto)* | **Status 429 Too Many Requests.** O filtro interceptor cortou a ligação prematuramente, retornando a mensagem padrão: `{"error": "Too many requests. Please try again later."}`. |

## 4. Conclusão
A infraestrutura de segurança foi validada com sucesso. O mecanismo de Rate Limiting garante a estabilidade e a alta disponibilidade da API Java contra ataques coordenados de negação de serviço e força bruta. A tarefa M-07 está formalmente homologada.