package com.logplatform.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.logplatform.config.SecurityConfig;
import com.logplatform.dto.ErrorPredictionRequest;
import com.logplatform.dto.ErrorPredictionResponse;
import com.logplatform.dto.ResponseTimePrediction;
import com.logplatform.service.PredictionService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.bean.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(PredictController.class)
@Import(SecurityConfig.class)
class PredictControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private PredictionService predictionService;

    @Test
    void predictError_shouldReturnPrediction() throws Exception {
        ErrorPredictionResponse response = ErrorPredictionResponse.builder()
                .errorProbability(0.27)
                .riskLevel("MEDIUM")
                .modelUsed("xgboost")
                .inferenceTimeMs(5.2)
                .build();

        when(predictionService.predictError(any())).thenReturn(response);

        ErrorPredictionRequest request = ErrorPredictionRequest.builder()
                .method("GET")
                .hour(14)
                .historicalAvgResponse(240.0)
                .build();

        mockMvc.perform(post("/predict/error")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.error_probability").value(0.27))
                .andExpect(jsonPath("$.risk_level").value("MEDIUM"))
                .andExpect(jsonPath("$.model_used").value("xgboost"));
    }

    @Test
    void predictError_shouldHandleServiceFailure() throws Exception {
        when(predictionService.predictError(any()))
                .thenThrow(new RuntimeException("ML service unavailable"));

        ErrorPredictionRequest request = ErrorPredictionRequest.builder()
                .method("GET")
                .hour(14)
                .historicalAvgResponse(240.0)
                .build();

        mockMvc.perform(post("/predict/error")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isInternalServerError())
                .andExpect(jsonPath("$.risk_level").value("UNKNOWN"));
    }

    @Test
    void predictResponseTime_shouldReturnPrediction() throws Exception {
        ResponseTimePrediction response = ResponseTimePrediction.builder()
                .predictedResponseTimeMs(285.50)
                .confidenceInterval(Map.of(
                        "lower_bound_ms", 120.0,
                        "upper_bound_ms", 450.0,
                        "confidence_level", 0.95
                ))
                .modelUsed("gradient_boosting")
                .inferenceTimeMs(3.8)
                .build();

        when(predictionService.predictResponseTime(any())).thenReturn(response);

        ErrorPredictionRequest request = ErrorPredictionRequest.builder()
                .method("POST")
                .hour(10)
                .historicalAvgResponse(300.0)
                .build();

        mockMvc.perform(post("/predict/response-time")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.predicted_response_time_ms").value(285.50))
                .andExpect(jsonPath("$.confidence_interval.confidence_level").value(0.95))
                .andExpect(jsonPath("$.model_used").value("gradient_boosting"));
    }

    @Test
    void predictError_shouldValidateInput() throws Exception {
        // Missing required method field
        String invalidJson = "{\"hour\": 14, \"historicalAvgResponse\": 240.0}";

        mockMvc.perform(post("/predict/error")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(invalidJson))
                .andExpect(status().isBadRequest());
    }
}
