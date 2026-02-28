package com.logplatform.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.logplatform.dto.ErrorPredictionRequest;
import com.logplatform.dto.ErrorPredictionResponse;
import com.logplatform.dto.ResponseTimePrediction;
import com.logplatform.entity.Prediction;
import com.logplatform.repository.PredictionRepository;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.Map;

@Service
@Slf4j
public class PredictionService {

    private final WebClient webClient;
    private final PredictionRepository predictionRepository;
    private final ObjectMapper objectMapper;
    private final Timer inferenceTimer;
    private final Counter predictionCounter;
    private final Counter predictionErrorCounter;

    public PredictionService(
            WebClient.Builder webClientBuilder,
            @Value("${ml.service.url}") String mlServiceUrl,
            @Value("${ml.service.timeout:30000}") int timeout,
            PredictionRepository predictionRepository,
            ObjectMapper objectMapper,
            MeterRegistry meterRegistry) {

        this.webClient = webClientBuilder
                .baseUrl(mlServiceUrl)
                .build();
        this.predictionRepository = predictionRepository;
        this.objectMapper = objectMapper;

        this.inferenceTimer = Timer.builder("ml.inference.latency")
                .description("ML inference latency")
                .register(meterRegistry);
        this.predictionCounter = Counter.builder("ml.predictions.total")
                .description("Total predictions made")
                .register(meterRegistry);
        this.predictionErrorCounter = Counter.builder("ml.predictions.errors")
                .description("Total prediction errors")
                .register(meterRegistry);
    }

    public ErrorPredictionResponse predictError(ErrorPredictionRequest request) {
        return inferenceTimer.record(() -> {
            try {
                Map<String, Object> mlRequest = Map.of(
                        "method", request.getMethod().toUpperCase(),
                        "hour", request.getHour(),
                        "historical_avg_response", request.getHistoricalAvgResponse(),
                        "day_of_week", request.getDayOfWeek()
                );

                ErrorPredictionResponse response = webClient.post()
                        .uri("/predict/error")
                        .bodyValue(mlRequest)
                        .retrieve()
                        .bodyToMono(ErrorPredictionResponse.class)
                        .timeout(Duration.ofSeconds(30))
                        .block();

                predictionCounter.increment();

                // Persist prediction
                savePrediction("error", mlRequest, response);

                return response;
            } catch (Exception e) {
                predictionErrorCounter.increment();
                log.error("Error prediction failed: {}", e.getMessage());
                throw new RuntimeException("ML service error prediction failed: " + e.getMessage(), e);
            }
        });
    }

    public ResponseTimePrediction predictResponseTime(ErrorPredictionRequest request) {
        return inferenceTimer.record(() -> {
            try {
                Map<String, Object> mlRequest = Map.of(
                        "method", request.getMethod().toUpperCase(),
                        "hour", request.getHour(),
                        "historical_avg_response", request.getHistoricalAvgResponse(),
                        "day_of_week", request.getDayOfWeek(),
                        "is_error", 0
                );

                ResponseTimePrediction response = webClient.post()
                        .uri("/predict/response-time")
                        .bodyValue(mlRequest)
                        .retrieve()
                        .bodyToMono(ResponseTimePrediction.class)
                        .timeout(Duration.ofSeconds(30))
                        .block();

                predictionCounter.increment();

                savePrediction("response_time", mlRequest, response);

                return response;
            } catch (Exception e) {
                predictionErrorCounter.increment();
                log.error("Response time prediction failed: {}", e.getMessage());
                throw new RuntimeException("ML service response time prediction failed: " + e.getMessage(), e);
            }
        });
    }

    private void savePrediction(String type, Object input, Object result) {
        try {
            Prediction prediction = Prediction.builder()
                    .predictionType(type)
                    .inputData(objectMapper.writeValueAsString(input))
                    .result(objectMapper.writeValueAsString(result))
                    .build();
            predictionRepository.save(prediction);
        } catch (Exception e) {
            log.warn("Failed to save prediction: {}", e.getMessage());
        }
    }
}
