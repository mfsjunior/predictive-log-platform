package com.logplatform.infrastructure.adapter;

import com.logplatform.domain.model.PredictionResult;
import com.logplatform.domain.port.MlServicePort;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.Map;

/**
 * Infrastructure adapter: implements MlServicePort via HTTP (WebClient).
 * Domain doesn't know about HTTP — this adapter translates.
 */
@Component
@Slf4j
public class MlServiceAdapter implements MlServicePort {

    private final WebClient webClient;
    private final int timeoutSeconds;

    public MlServiceAdapter(
            WebClient.Builder webClientBuilder,
            @Value("${ml.service.url}") String mlServiceUrl,
            @Value("${ml.service.timeout:30000}") int timeoutMs) {
        this.webClient = webClientBuilder.baseUrl(mlServiceUrl).build();
        this.timeoutSeconds = timeoutMs / 1000;
    }

    @Override
    public PredictionResult predictError(String method, int hour,
            double historicalAvgResponse, int dayOfWeek) {
        Map<String, Object> request = Map.of(
                "method", method.toUpperCase(),
                "hour", hour,
                "historical_avg_response", historicalAvgResponse,
                "day_of_week", dayOfWeek);

        Map response = webClient.post()
                .uri("/predict/error")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(Map.class)
                .timeout(Duration.ofSeconds(timeoutSeconds))
                .block();

        if (response == null) {
            throw new RuntimeException("Empty response from ML service");
        }

        return PredictionResult.errorPrediction(
                ((Number) response.get("error_probability")).doubleValue(),
                (String) response.get("risk_level"),
                (String) response.get("model_used"));
    }

    @Override
    public PredictionResult predictResponseTime(String method, int hour,
            double historicalAvgResponse, int dayOfWeek) {
        Map<String, Object> request = Map.of(
                "method", method.toUpperCase(),
                "hour", hour,
                "historical_avg_response", historicalAvgResponse,
                "day_of_week", dayOfWeek,
                "is_error", 0);

        Map response = webClient.post()
                .uri("/predict/response-time")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(Map.class)
                .timeout(Duration.ofSeconds(timeoutSeconds))
                .block();

        if (response == null) {
            throw new RuntimeException("Empty response from ML service");
        }

        double predicted = ((Number) response.get("predicted_response_time_ms")).doubleValue();
        Map ci = (Map) response.get("confidence_interval");
        double lower = ci != null ? ((Number) ci.get("lower_bound_ms")).doubleValue() : 0;
        double upper = ci != null ? ((Number) ci.get("upper_bound_ms")).doubleValue() : 0;
        double confidence = ci != null ? ((Number) ci.get("confidence_level")).doubleValue() : 0.95;

        return PredictionResult.responseTimePrediction(predicted, lower, upper, confidence);
    }
}
