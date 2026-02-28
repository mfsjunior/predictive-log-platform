package com.logplatform.controller;

import com.logplatform.dto.ErrorPredictionRequest;
import com.logplatform.dto.ErrorPredictionResponse;
import com.logplatform.dto.ResponseTimePrediction;
import com.logplatform.service.PredictionService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/predict")
@RequiredArgsConstructor
@Tag(name = "Predictions", description = "ML-based predictions for error probability and response time")
public class PredictController {

    private final PredictionService predictionService;

    @PostMapping("/error")
    @Operation(
            summary = "Predict error probability",
            description = "Predicts the probability of an HTTP error (4xx/5xx) based on method, hour, and historical response time"
    )
    public ResponseEntity<ErrorPredictionResponse> predictError(
            @Valid @RequestBody ErrorPredictionRequest request) {
        try {
            ErrorPredictionResponse response = predictionService.predictError(request);
            return ResponseEntity.ok(response);
        } catch (RuntimeException e) {
            return ResponseEntity.internalServerError().body(
                    ErrorPredictionResponse.builder()
                            .errorProbability(-1)
                            .riskLevel("UNKNOWN")
                            .modelUsed("error: " + e.getMessage())
                            .build()
            );
        }
    }

    @PostMapping("/response-time")
    @Operation(
            summary = "Predict response time",
            description = "Predicts the expected response time with 95% confidence interval"
    )
    public ResponseEntity<ResponseTimePrediction> predictResponseTime(
            @Valid @RequestBody ErrorPredictionRequest request) {
        try {
            ResponseTimePrediction response = predictionService.predictResponseTime(request);
            return ResponseEntity.ok(response);
        } catch (RuntimeException e) {
            return ResponseEntity.internalServerError().body(
                    ResponseTimePrediction.builder()
                            .predictedResponseTimeMs(-1)
                            .modelUsed("error: " + e.getMessage())
                            .build()
            );
        }
    }
}
