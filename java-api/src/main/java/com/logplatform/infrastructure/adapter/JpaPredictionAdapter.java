package com.logplatform.infrastructure.adapter;

import com.logplatform.domain.model.PredictionResult;
import com.logplatform.domain.port.PredictionPort;
import com.logplatform.entity.Prediction;
import com.logplatform.mapper.PredictionMapper;
import com.logplatform.repository.PredictionRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * Adaptador de Infraestrutura: Persiste registros de auditoria de predição via JPA usando Mapper.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class JpaPredictionAdapter implements PredictionPort {

    private final PredictionRepository predictionRepository;
    private final PredictionMapper predictionMapper;

    @Override
    public void save(PredictionResult domain, Object input) {
        try {
            Prediction prediction = predictionMapper.toEntity(domain, input);
            predictionRepository.save(prediction);
        } catch (Exception e) {
            log.warn("Failed to persist prediction audit: {}", e.getMessage());
        }
    }
}
