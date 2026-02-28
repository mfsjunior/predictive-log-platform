package com.logplatform.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ErrorPredictionResponse {

    @JsonProperty("error_probability")
    private double errorProbability;

    @JsonProperty("risk_level")
    private String riskLevel;

    @JsonProperty("model_used")
    private String modelUsed;

    @JsonProperty("inference_time_ms")
    private Double inferenceTimeMs;
}
