package com.logplatform.exception;

import com.logplatform.dto.ApiErrorResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.context.request.WebRequest;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

import java.util.stream.Collectors;

/**
 * Manipulador global de exceções para a API REST.
 * 
 * Teoria para aula:
 * - @RestControllerAdvice: Intercepta exceções em todos os @RestControllers automaticamente.
 * - Centraliza a lógica de tratamento de erro — elimina try/catch repetido nos controllers.
 * - ResponseEntity: Permite controlar status HTTP, headers e body da resposta.
 * - Todos os erros retornam no formato ApiErrorResponse — contrato consistente.
 * 
 * Fluxo:
 * 1. Exceção lançada em um controller
 * 2. Spring procura por @ExceptionHandler que trata esse tipo
 * 3. Este handler transforma a exceção em ResponseEntity com status/body apropriados
 * 4. Cliente recebe JSON padronizado
 */
@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

    /**
     * Trata exceções de negócio (domínio) com status HTTP customizado.
     */
    @ExceptionHandler(BusinessException.class)
    public ResponseEntity<ApiErrorResponse> handleBusinessException(
            BusinessException ex,
            WebRequest request) {
        
        log.warn("Business exception: {}", ex.getMessage());
        
        ApiErrorResponse errorResponse = ApiErrorResponse.of(
                ex.getHttpStatus(),
                "Business Error",
                ex.getMessage(),
                request.getDescription(false).replace("uri=", "")
        );
        
        return ResponseEntity
                .status(ex.getHttpStatus())
                .body(errorResponse);
    }

    /**
     * Trata validações de Bean Validation (@Valid, @NotBlank, etc).
     * Agrupa mensagens de erro de múltiplos campos.
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ApiErrorResponse> handleValidationException(
            MethodArgumentNotValidException ex,
            WebRequest request) {
        
        String errors = ex.getBindingResult()
                .getFieldErrors()
                .stream()
                .map(error -> error.getField() + ": " + error.getDefaultMessage())
                .collect(Collectors.joining(", "));
        
        log.warn("Validation failed: {}", errors);
        
        ApiErrorResponse errorResponse = ApiErrorResponse.of(
                HttpStatus.BAD_REQUEST.value(),
                "Validation Error",
                "Invalid request: " + errors,
                request.getDescription(false).replace("uri=", "")
        );
        
        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(errorResponse);
    }

    /**
     * Trata upload de arquivos acima do limite de tamanho.
     */
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ApiErrorResponse> handleMaxUploadSizeExceeded(
            MaxUploadSizeExceededException ex,
            WebRequest request) {
        
        log.warn("Upload size exceeded: {}", ex.getMessage());
        
        ApiErrorResponse errorResponse = ApiErrorResponse.of(
                HttpStatus.PAYLOAD_TOO_LARGE.value(),
                "Upload Size Exceeded",
                "File is too large. Maximum allowed size is 10 MB.",
                request.getDescription(false).replace("uri=", "")
        );
        
        return ResponseEntity
                .status(HttpStatus.PAYLOAD_TOO_LARGE)
                .body(errorResponse);
    }

    /**
     * Trata exceções genéricas não previstas.
     * Sempre retorna 500 (Internal Server Error).
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiErrorResponse> handleGenericException(
            Exception ex,
            WebRequest request) {
        
        log.error("Unexpected error", ex);
        
        ApiErrorResponse errorResponse = ApiErrorResponse.of(
                HttpStatus.INTERNAL_SERVER_ERROR.value(),
                "Internal Server Error",
                "An unexpected error occurred. Please contact support.",
                request.getDescription(false).replace("uri=", "")
        );
        
        return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(errorResponse);
    }

    /**
     * Trata ArgumentException (validação inline de argumentos).
     * Comum em services quando detectam argumentos inválidos.
     */
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ApiErrorResponse> handleIllegalArgumentException(
            IllegalArgumentException ex,
            WebRequest request) {
        
        log.warn("Illegal argument: {}", ex.getMessage());
        
        ApiErrorResponse errorResponse = ApiErrorResponse.of(
                HttpStatus.BAD_REQUEST.value(),
                "Bad Request",
                ex.getMessage(),
                request.getDescription(false).replace("uri=", "")
        );
        
        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(errorResponse);
    }
}
