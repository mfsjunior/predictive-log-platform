# Documentação de Teste e Arquitetura: Global Exception Handler (M-01)

Este documento serve como evidência técnica da conclusão da tarefa M-01 (Sprint 1), focada na reestruturação da camada de validação e tratamento de erros globais da API Spring Boot (Java 21) da Predictive Log Platform.

## 1. Resumo da Refatoração
A implementação anterior do endpoint de autenticação utilizava estruturas genéricas (`Map<String, String>`) para recepção de payloads, o que impedia o framework de aplicar reflexões e validações automáticas do ecossistema Jakarta Validation. Como consequência, requisições malformadas alcançavam a camada de negócio, resultando em retornos HTTP 401 ou 500 inadequados. A solução arquitetural consistiu em tipar as entradas via DTOs (Java Records), aplicar a diretiva `@Valid` e centralizar o tratamento de exceções via `@RestControllerAdvice`.

## 2. Modificações no Código-Fonte

### 2.1. Criação do DTO tipado (`LoginRequest.java`)
O `Map` genérico foi substituído por um Java Record imutável, garantindo a integridade dos dados desde a entrada. Regras de negócio (`@NotBlank`) foram atreladas aos atributos.
```java
public record LoginRequest(
    @NotBlank(message = "O usuário é obrigatório") 
    String username,
    
    @NotBlank(message = "A senha é obrigatória") 
    String password
) {}

2.2. Blindagem do Controlador (AuthController.java)

A assinatura do método de login foi atualizada com a diretiva de validação, instruindo o Spring Boot a barrar a requisição antes da execução do método caso as regras do Record não sejam satisfeitas.

@PostMapping("/login")
public ResponseEntity<?> login(@Valid @RequestBody LoginRequest request) {
    // ... lógica de autenticação com dados seguros ...
}

2.3. Interceptação Centralizada (GlobalExceptionHandler.java)

O tratador global foi configurado para escutar a exceção MethodArgumentNotValidException, extrair as mensagens amigáveis dos campos falhos e padronizar o retorno no contrato institucional ApiErrorResponse.

3. Evidência de Teste End-to-End (E2E)

Para homologar a barreira de validação e o roteamento do Exception Handler, simulamos uma requisição de cliente malformada através do Postman:
Etapa	Ação (Postman)	Resultado Obtido

1. Envio de Payload Vazio	

POST http://localhost:8080/auth/login

Payload (JSON): {}
	Intercepção bem-sucedida. O Spring Boot barrou a requisição antes de acionar a camada de serviço.

2. Retorno do Global Handler	Validação da Resposta HTTP	Status 400 Bad Request. O sistema retornou a estrutura padrão ApiErrorResponse contendo o array validationErrors, especificando exatamente a ausência obrigatória de username e password.

4. Conclusão

A API Java agora possui uma camada de proteção robusta e centralizada. A tipagem estrita via Records em conjunto com o Global Exception Handler impede que dados sujos transitem pela aplicação, melhorando a segurança, previsibilidade e a experiência de integração para o Front-end. A tarefa M-01 está formalmente homologada.


5. Resolução de Conflitos de Build (DevOps)

Durante o processo de testes locais (Windows), observou-se que o Docker Daemon retinha o cache das camadas antigas da imagem, ignorando o novo código do AuthController.

Para contornar esse falso-positivo e garantir que o artefato (.jar) refletisse as anotações @Valid, adotou-se a estratégia de build em ambiente isolado utilizando a imagem maven:3.9-eclipse-temurin-21.

O comando abaixo foi estabelecido como padrão para recompilação limpa da API antes do deploy via Docker Compose:
docker run --rm -v "${PWD}:/app" -w /app maven:3.9-eclipse-temurin-21 mvn clean package -DskipTests