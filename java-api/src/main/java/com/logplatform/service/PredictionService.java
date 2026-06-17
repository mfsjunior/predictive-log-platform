package com.logplatform.service;

import com.logplatform.domain.model.PredictionResult;
import com.logplatform.domain.port.MlServicePort;
import com.logplatform.domain.port.PredictionPort;
import com.logplatform.dto.ErrorPredictionRequest;
import com.logplatform.dto.ErrorPredictionResponse;
import com.logplatform.dto.ResponseTimePrediction;
import com.logplatform.mapper.PredictionMapper;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * Serviço de Predições de Machine Learning da camada de Aplicação.
 * 
 * Totalmente desacoplado de infraestrutura (WebClient, Repositórios JPA) através do uso de Portas e Mappers.
 */
@Service
@Slf4j
public class PredictionService {

    private final MlServicePort mlServicePort;
    private final PredictionPort predictionPort;
    private final PredictionMapper predictionMapper;
    
    private final Timer inferenceTimer;
    private final Counter predictionCounter;
    private final Counter predictionErrorCounter;

    public PredictionService(
            MlServicePort mlServicePort,
            PredictionPort predictionPort,
            PredictionMapper predictionMapper,
            MeterRegistry meterRegistry) {
        
        this.mlServicePort = mlServicePort;
        this.predictionPort = predictionPort;
        this.predictionMapper = predictionMapper;

        // Timer: Mede a latência da inferência de IA nas ferramentas de monitoramento
        this.inferenceTimer = Timer.builder("ml.inference.latency")
                .description("Latência da inferência de ML")
                .register(meterRegistry);
        // Counter: Conta o total de execuções
        this.predictionCounter = Counter.builder("ml.predictions.total")
                .description("Total de predições realizadas")
                .register(meterRegistry);
        // Counter: Conta erros de predição
        this.predictionErrorCounter = Counter.builder("ml.predictions.errors")
                .description("Total prediction errors")
                .register(meterRegistry);
    }

    public ErrorPredictionResponse predictError(ErrorPredictionRequest request) {
        return inferenceTimer.record(() -> {
            try {
                // 1. Realiza a inferência através da porta de domínio
                PredictionResult result = mlServicePort.predictError(
                        request.getMethod().toUpperCase(),
                        request.getHour(),
                        request.getHistoricalAvgResponse(),
                        request.getDayOfWeek()
                );

                // 2. Incrementa o contador de sucesso
                predictionCounter.increment();

                // 3. Salva a predição via porta de persistência para auditoria
                predictionPort.save(result, request);

                // 4. Mapeia para o DTO de resposta usando o Mapper
                return predictionMapper.toErrorResponse(result);
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
                // 1. Realiza a inferência através da porta de domínio
                PredictionResult result = mlServicePort.predictResponseTime(
                        request.getMethod().toUpperCase(),
                        request.getHour(),
                        request.getHistoricalAvgResponse(),
                        request.getDayOfWeek()
                );

                // 2. Incrementa o contador de sucesso
                predictionCounter.increment();

                // 3. Salva a predição via porta de persistência para auditoria
                predictionPort.save(result, request);

                // 4. Mapeia para o DTO de resposta usando o Mapper
                return predictionMapper.toResponseTimeResponse(result);
            } catch (Exception e) {
                predictionErrorCounter.increment();
                log.error("Response time prediction failed: {}", e.getMessage());
                throw new RuntimeException("ML service response time prediction failed: " + e.getMessage(), e);
            }
        });
    }
}
