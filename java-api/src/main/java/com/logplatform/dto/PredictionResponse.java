package com.logplatform.dto;

import com.logplatform.entity.Prediction;

import java.time.LocalDateTime;

public record PredictionResponse(
        Long id,
        String predictionType,
        String inputData,
        String result,
        String modelVersion,
        Double latencyMs,
        LocalDateTime createdAt
) {
    public static PredictionResponse from(Prediction prediction) {
        return new PredictionResponse(
                prediction.getId(),
                prediction.getPredictionType(),
                prediction.getInputData(),
                prediction.getResult(),
                prediction.getModelVersion(),
                prediction.getLatencyMs(),
                prediction.getCreatedAt()
        );
    }
}
