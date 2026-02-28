package com.logplatform.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ResponseTimePrediction {

    @JsonProperty("predicted_response_time_ms")
    private double predictedResponseTimeMs;

    @JsonProperty("confidence_interval")
    private Map<String, Object> confidenceInterval;

    @JsonProperty("model_used")
    private String modelUsed;

    @JsonProperty("inference_time_ms")
    private Double inferenceTimeMs;
}
