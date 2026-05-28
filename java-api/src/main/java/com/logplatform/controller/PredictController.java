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

/**
 * Controller responsável pelas Predições de Machine Learning.
 * 
 * Teoria para aula:
 * - Este controller atua como um "Orquestrador" ou "Gateway".
 * - Ele não executa a inteligência artificial aqui (quem faz isso é o Python), 
 *   mas ele gerencia o fluxo: recebe o pedido -> consulta o cache -> chama o ML -> 
 *   salva para auditoria -> retorna ao cliente.
 * 
 * Algoritmos envolvidos (no lado Python):
 * 1. Classificação (Erro): Prediz se a requisição vai falhar (sim/não + probabilidade).
 * 2. Regressão (Tempo): Prediz um valor numérico (quantos ms vai demorar).
 */
@RestController
@RequestMapping("/predict")
@RequiredArgsConstructor
@Tag(name = "Predictions", description = "ML-based predictions for error probability and response time")
public class PredictController {

    private final PredictionService predictionService;

    @PostMapping("/error")
    @Operation(summary = "Predizer probabilidade de erro", description = "Prediz a chance de erro HTTP (4xx/5xx) baseado no método, hora e histórico")
    public ResponseEntity<ErrorPredictionResponse> predictError(
            @Valid @RequestBody ErrorPredictionRequest request) {
        ErrorPredictionResponse response = predictionService.predictError(request);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/response-time")
    @Operation(summary = "Predizer tempo de resposta", description = "Prediz o tempo de resposta esperado com intervalo de confiança de 95%")
    public ResponseEntity<ResponseTimePrediction> predictResponseTime(
            @Valid @RequestBody ErrorPredictionRequest request) {
        ResponseTimePrediction response = predictionService.predictResponseTime(request);
        return ResponseEntity.ok(response);
    }
}
