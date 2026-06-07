package com.logplatform.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.time.LocalDateTime;

/**
 * DTO para resposta de dados de logs web sem expor campos de auditoria interna.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class WebLogResponse {

    private Long id;

    private LocalDateTime timestamp;

    private String method;

    private String path;

    @JsonProperty("status_code")
    private int statusCode;

    @JsonProperty("response_time_ms")
    private double responseTimeMs;

    @JsonProperty("user_agent")
    private String userAgent;

    @JsonProperty("ip_address")
    private String ipAddress;

    @JsonProperty("bytes_sent")
    private int bytesSent;
}
