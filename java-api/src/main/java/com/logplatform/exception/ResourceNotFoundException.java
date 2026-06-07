package com.logplatform.exception;

/**
 * Exceção para quando um recurso solicitado não existe (404 Not Found).
 * 
 * Uso: throw new ResourceNotFoundException("Log com ID 123 não encontrado");
 */
public class ResourceNotFoundException extends BusinessException {
    
    public ResourceNotFoundException(String message) {
        super(message, 404);
    }
    
    public ResourceNotFoundException(String resourceName, Long id) {
        super(String.format("%s com ID %d não encontrado", resourceName, id), 404);
    }
}
