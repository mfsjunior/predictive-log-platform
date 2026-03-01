package com.logplatform.infrastructure.adapter;

import com.logplatform.domain.port.PredictionPort;
import com.logplatform.entity.Prediction;
import com.logplatform.repository.PredictionRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * Infrastructure adapter: persists prediction audit records via JPA.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class JpaPredictionAdapter implements PredictionPort {

    private final PredictionRepository predictionRepository;

    @Override
    public void save(String type, String inputJson, String resultJson) {
        try {
            Prediction prediction = Prediction.builder()
                    .predictionType(type)
                    .inputData(inputJson)
                    .result(resultJson)
                    .build();
            predictionRepository.save(prediction);
        } catch (Exception e) {
            log.warn("Failed to persist prediction audit: {}", e.getMessage());
        }
    }
}
