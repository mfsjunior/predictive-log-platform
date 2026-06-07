package com.logplatform.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * DTO público para expor predições auditadas.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PredictionResponse {
    private Long id;
    private String predictionType;
    private Object inputData;
    private Object result;
    private String modelVersion;
    private Double latencyMs;
    private LocalDateTime createdAt;
}
