package com.logplatform.domain.port;

import com.logplatform.domain.model.PredictionResult;
import java.util.Map;

/**
 * Port (interface) for ML service communication.
 * Domain doesn't know if it's HTTP, gRPC, or Kafka.
 */
public interface MlServicePort {

    PredictionResult predictError(String method, int hour, double historicalAvgResponse, int dayOfWeek);

    PredictionResult predictResponseTime(String method, int hour, double historicalAvgResponse, int dayOfWeek);
}
