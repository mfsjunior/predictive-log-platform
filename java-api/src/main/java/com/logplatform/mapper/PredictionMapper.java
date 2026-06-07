package com.logplatform.mapper;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.logplatform.domain.model.PredictionResult;
import com.logplatform.dto.ErrorPredictionResponse;
import com.logplatform.dto.ResponseTimePrediction;
import com.logplatform.entity.Prediction;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * Mapper manual para converter entre Entidade JPA Prediction, Domínio PredictionResult e DTOs correspondentes.
 */
@Component
@RequiredArgsConstructor
public class PredictionMapper {

    private final ObjectMapper objectMapper;

    public PredictionResult toDomain(Prediction entity) {
        if (entity == null) {
            return null;
        }

        PredictionResult domain = new PredictionResult();
        domain.setPredictionType(entity.getPredictionType());
        domain.setModelUsed(entity.getModelVersion());
        domain.setInferenceTimeMs(entity.getLatencyMs() != null ? entity.getLatencyMs() : 0.0);

        try {
            if ("error".equals(entity.getPredictionType())) {
                ErrorPredictionResponse resp = objectMapper.readValue(entity.getResult(), ErrorPredictionResponse.class);
                domain.setValue(resp.getErrorProbability());
                domain.setRiskLevel(resp.getRiskLevel());
            } else if ("response_time".equals(entity.getPredictionType())) {
                ResponseTimePrediction resp = objectMapper.readValue(entity.getResult(), ResponseTimePrediction.class);
                domain.setValue(resp.getPredictedResponseTimeMs());
                Map<String, Object> ci = resp.getConfidenceInterval();
                if (ci != null) {
                    domain.setLowerBound(ci.get("lower_bound_ms") instanceof Number ? ((Number) ci.get("lower_bound_ms")).doubleValue() : 0.0);
                    domain.setUpperBound(ci.get("upper_bound_ms") instanceof Number ? ((Number) ci.get("upper_bound_ms")).doubleValue() : 0.0);
                    domain.setConfidenceLevel(ci.get("confidence_level") instanceof Number ? ((Number) ci.get("confidence_level")).doubleValue() : 0.95);
                } else {
                    domain.setLowerBound(0.0);
                    domain.setUpperBound(0.0);
                    domain.setConfidenceLevel(0.95);
                }
            }
        } catch (Exception e) {
            // Em caso de erro de desserialização, não lança exceção, mas deixa os campos com fallback padrão
            domain.setValue(0.0);
        }

        return domain;
    }

    public Prediction toEntity(PredictionResult domain, Object inputDto) {
        if (domain == null) {
            return null;
        }

        try {
            Object resultDto;
            if ("error".equals(domain.getPredictionType())) {
                resultDto = toErrorResponse(domain);
            } else {
                resultDto = toResponseTimeResponse(domain);
            }

            return Prediction.builder()
                    .predictionType(domain.getPredictionType())
                    .inputData(inputDto != null ? objectMapper.writeValueAsString(inputDto) : null)
                    .result(objectMapper.writeValueAsString(resultDto))
                    .modelVersion(domain.getModelUsed())
                    .latencyMs(domain.getInferenceTimeMs())
                    .build();
        } catch (Exception e) {
            throw new RuntimeException("Failed to map PredictionResult to Prediction entity", e);
        }
    }

    public ErrorPredictionResponse toErrorResponse(PredictionResult domain) {
        if (domain == null) {
            return null;
        }

        return ErrorPredictionResponse.builder()
                .errorProbability(domain.getValue())
                .riskLevel(domain.getRiskLevel())
                .modelUsed(domain.getModelUsed())
                .inferenceTimeMs(domain.getInferenceTimeMs())
                .build();
    }

    public ResponseTimePrediction toResponseTimeResponse(PredictionResult domain) {
        if (domain == null) {
            return null;
        }

        Map<String, Object> ci = Map.of(
                "lower_bound_ms", domain.getLowerBound(),
                "upper_bound_ms", domain.getUpperBound(),
                "confidence_level", domain.getConfidenceLevel()
        );

        return ResponseTimePrediction.builder()
                .predictedResponseTimeMs(domain.getValue())
                .confidenceInterval(ci)
                .modelUsed(domain.getModelUsed())
                .inferenceTimeMs(domain.getInferenceTimeMs())
                .build();
    }
}
