package com.logplatform.exception;

/**
 * Exceção base para erros de negócio (domain logic).
 * 
 * Teoria para aula:
 * - Exceções unchecked estendem RuntimeException, não obrigam try/catch.
 * - Convenção: exceções de negócio devem ser RuntimeException para simplicidade.
 * - GlobalExceptionHandler as captura e transforma em respostas HTTP apropriadas.
 */
public class BusinessException extends RuntimeException {
    
    private final int httpStatus;
    
    public BusinessException(String message) {
        this(message, 400);  // Bad Request por padrão
    }
    
    public BusinessException(String message, int httpStatus) {
        super(message);
        this.httpStatus = httpStatus;
    }
    
    public BusinessException(String message, Throwable cause) {
        this(message, cause, 400);
    }
    
    public BusinessException(String message, Throwable cause, int httpStatus) {
        super(message, cause);
        this.httpStatus = httpStatus;
    }
    
    public int getHttpStatus() {
        return httpStatus;
    }
}
