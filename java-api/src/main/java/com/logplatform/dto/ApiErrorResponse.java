package com.logplatform.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Padrão padronizado para resposta de erro em toda a API.
 * 
 * Teoria para aula:
 * - DTO (Data Transfer Object): Classe responsável apenas por transportar dados entre camadas.
 * - @JsonInclude: Garante que campos nulos não apareçam no JSON de resposta (limpeza).
 * - Todos os erros retornam neste formato — consistência é importante para clientes da API.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ApiErrorResponse {
    
    private String timestamp;
    private int status;
    private String error;
    private String message;
    private String path;
    private String traceId;  // Opcional: para correlação de logs em ambientes distribuídos
    
    /**
     * Factory method para criar erro com timestamp automático.
     */
    public static ApiErrorResponse of(int status, String error, String message, String path) {
        return ApiErrorResponse.builder()
                .timestamp(LocalDateTime.now().format(DateTimeFormatter.ISO_DATE_TIME))
                .status(status)
                .error(error)
                .message(message)
                .path(path)
                .build();
    }
}
