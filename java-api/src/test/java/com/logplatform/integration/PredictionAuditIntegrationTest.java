package com.logplatform.integration;

import com.logplatform.config.TestContainersConfig;
import com.logplatform.domain.model.PredictionResult;
import com.logplatform.domain.port.MlServicePort;
import com.logplatform.dto.ErrorPredictionRequest;
import com.logplatform.dto.ErrorPredictionResponse;
import com.logplatform.repository.PredictionRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.*;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;

/**
 * Teste de integração da Auditoria de Predições (Módulo M-10).
 *
 * Teoria para aula:
 * - Toda predição é registrada no banco para auditoria (comparar previsto x real no futuro).
 *   Aqui validamos que, ao chamar POST /predict/error, a predição é de fato salva no PostgreSQL.
 * - Decisão de arquitetura: mockamos a PORTA de domínio (MlServicePort), não o WebClient.
 *   Isso isola o teste do serviço Python de ML (que não sobe aqui), mas mantém REAL todo o
 *   restante do fluxo: controller -> PredictionService -> porta de persistência -> banco.
 *   É o ponto de mock correto numa arquitetura hexagonal (ports & adapters).
 */
class PredictionAuditIntegrationTest extends TestContainersConfig {

    // Substitui a chamada ao serviço de ML por uma resposta controlada
    @MockBean
    private MlServicePort mlServicePort;

    @Autowired
    private PredictionRepository predictionRepository;

    @BeforeEach
    void cleanDatabase() {
        // Isolamento entre testes: zera a tabela de predições
        predictionRepository.deleteAll();
    }

    @Test
    void predictError_savesAuditRecordInDatabase() {
        // Configura o ML "fake" para devolver uma predição conhecida
        when(mlServicePort.predictError(anyString(), anyInt(), anyDouble(), anyInt()))
                .thenReturn(PredictionResult.errorPrediction(0.15, "LOW", "RandomForestClassifier"));

        ErrorPredictionRequest request = new ErrorPredictionRequest();
        request.setMethod("GET");
        request.setHour(14);
        request.setHistoricalAvgResponse(250.0);
        request.setDayOfWeek(2);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        ResponseEntity<ErrorPredictionResponse> response = restTemplate.postForEntity(
                baseUrl + "/predict/error",
                new HttpEntity<>(request, headers),
                ErrorPredictionResponse.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        // O ponto central do teste: a predição foi gravada para auditoria
        assertThat(predictionRepository.count()).isEqualTo(1);
    }

    @Test
    void predictError_returnsCorrectPredictionValues() {
        // Verifica que os valores retornados pela API batem com o que o ML "previu"
        PredictionResult fakeResult = PredictionResult.errorPrediction(0.87, "CRITICAL", "XGBoostClassifier");
        fakeResult.setInferenceTimeMs(5.2);
        when(mlServicePort.predictError(anyString(), anyInt(), anyDouble(), anyInt()))
                .thenReturn(fakeResult);

        ErrorPredictionRequest request = new ErrorPredictionRequest();
        request.setMethod("POST");
        request.setHour(3);
        request.setHistoricalAvgResponse(5000.0);
        request.setDayOfWeek(0);

        ResponseEntity<ErrorPredictionResponse> response = restTemplate.postForEntity(
                baseUrl + "/predict/error",
                new HttpEntity<>(request, new HttpHeaders() {{ setContentType(MediaType.APPLICATION_JSON); }}),
                ErrorPredictionResponse.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        ErrorPredictionResponse body = response.getBody();
        assertThat(body).isNotNull();
        assertThat(body.getErrorProbability()).isEqualTo(0.87);
        assertThat(body.getRiskLevel()).isEqualTo("CRITICAL");
        assertThat(body.getModelUsed()).isEqualTo("XGBoostClassifier");
    }

    @Test
    void predictError_multipleCalls_savesAllAuditRecords() {
        // Três chamadas devem gerar três registros de auditoria distintos
        when(mlServicePort.predictError(anyString(), anyInt(), anyDouble(), anyInt()))
                .thenReturn(PredictionResult.errorPrediction(0.10, "LOW", "LogisticRegression"));

        ErrorPredictionRequest request = new ErrorPredictionRequest();
        request.setMethod("GET");
        request.setHour(10);
        request.setHistoricalAvgResponse(150.0);
        request.setDayOfWeek(1);

        HttpEntity<ErrorPredictionRequest> entity = new HttpEntity<>(request, new HttpHeaders() {{
            setContentType(MediaType.APPLICATION_JSON);
        }});

        for (int i = 0; i < 3; i++) {
            restTemplate.postForEntity(baseUrl + "/predict/error", entity, ErrorPredictionResponse.class);
        }

        assertThat(predictionRepository.count()).isEqualTo(3);
    }
}
