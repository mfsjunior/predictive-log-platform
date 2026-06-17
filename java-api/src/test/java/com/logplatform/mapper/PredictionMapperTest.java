package com.logplatform.mapper;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.logplatform.domain.model.PredictionResult;
import com.logplatform.dto.ErrorPredictionRequest;
import com.logplatform.dto.ErrorPredictionResponse;
import com.logplatform.dto.ResponseTimePrediction;
import com.logplatform.entity.Prediction;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class PredictionMapperTest {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final PredictionMapper mapper = new PredictionMapper(objectMapper);

    @Test
    void toDomain_shouldMapErrorPredictionCorrectly() throws Exception {
        ErrorPredictionResponse responseDto = ErrorPredictionResponse.builder()
                .errorProbability(0.75)
                .riskLevel("HIGH")
                .modelUsed("random_forest")
                .inferenceTimeMs(15.4)
                .build();

        Prediction entity = Prediction.builder()
                .id(1L)
                .predictionType("error")
                .modelVersion("random_forest")
                .latencyMs(15.4)
                .result(objectMapper.writeValueAsString(responseDto))
                .build();

        PredictionResult domain = mapper.toDomain(entity);

        assertNotNull(domain);
        assertEquals("error", domain.getPredictionType());
        assertEquals("random_forest", domain.getModelUsed());
        assertEquals(15.4, domain.getInferenceTimeMs());
        assertEquals(0.75, domain.getValue());
        assertEquals("HIGH", domain.getRiskLevel());
    }

    @Test
    void toDomain_shouldMapResponseTimePredictionCorrectly() throws Exception {
        ResponseTimePrediction responseDto = ResponseTimePrediction.builder()
                .predictedResponseTimeMs(230.5)
                .confidenceInterval(Map.of(
                        "lower_bound_ms", 200.0,
                        "upper_bound_ms", 260.0,
                        "confidence_level", 0.95
                ))
                .modelUsed("linear_regression")
                .inferenceTimeMs(22.1)
                .build();

        Prediction entity = Prediction.builder()
                .id(2L)
                .predictionType("response_time")
                .modelVersion("linear_regression")
                .latencyMs(22.1)
                .result(objectMapper.writeValueAsString(responseDto))
                .build();

        PredictionResult domain = mapper.toDomain(entity);

        assertNotNull(domain);
        assertEquals("response_time", domain.getPredictionType());
        assertEquals("linear_regression", domain.getModelUsed());
        assertEquals(22.1, domain.getInferenceTimeMs());
        assertEquals(230.5, domain.getValue());
        assertEquals(200.0, domain.getLowerBound());
        assertEquals(260.0, domain.getUpperBound());
        assertEquals(0.95, domain.getConfidenceLevel());
    }

    @Test
    void toDomain_shouldHandleNullsAndErrors() {
        assertNull(mapper.toDomain(null));

        Prediction entity = Prediction.builder()
                .predictionType("error")
                .result("invalid json")
                .build();

        PredictionResult domain = mapper.toDomain(entity);
        assertNotNull(domain);
        assertEquals(0.0, domain.getValue());
    }

    @Test
    void toEntity_shouldMapErrorCorrectly() {
        PredictionResult domain = PredictionResult.errorPrediction(0.8, "HIGH", "xgboost");
        domain.setInferenceTimeMs(12.5);

        ErrorPredictionRequest request = ErrorPredictionRequest.builder()
                .method("GET")
                .hour(14)
                .dayOfWeek(3)
                .historicalAvgResponse(150.0)
                .build();

        Prediction entity = mapper.toEntity(domain, request);

        assertNotNull(entity);
        assertEquals("error", entity.getPredictionType());
        assertEquals("xgboost", entity.getModelVersion());
        assertEquals(12.5, entity.getLatencyMs());
        assertTrue(entity.getInputData().contains("GET"));
        assertTrue(entity.getResult().contains("0.8"));
        assertTrue(entity.getResult().contains("HIGH"));
    }

    @Test
    void toEntity_shouldHandleNull() {
        assertNull(mapper.toEntity(null, null));
    }
}
