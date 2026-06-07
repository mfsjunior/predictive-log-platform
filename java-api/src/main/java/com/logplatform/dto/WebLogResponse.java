package com.logplatform.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * DTO público para expor registros de log.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class WebLogResponse {
    private Long id;
    private LocalDateTime timestamp;
    private String method;
    private String path;
    private Integer statusCode;
    private Double responseTimeMs;
    private String userAgent;
    private String ipAddress;
    private Integer bytesSent;
    private LocalDateTime createdAt;
}
