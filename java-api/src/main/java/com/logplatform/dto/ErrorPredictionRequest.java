package com.logplatform.dto;

import jakarta.validation.constraints.*;
import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ErrorPredictionRequest {

    @NotBlank(message = "HTTP method is required")
    private String method;

    @Min(value = 0, message = "Hour must be between 0 and 23")
    @Max(value = 23, message = "Hour must be between 0 and 23")
    private int hour;

    @Positive(message = "Historical average response must be positive")
    private double historicalAvgResponse;

    @Min(0) @Max(6)
    private int dayOfWeek = 2;
}
