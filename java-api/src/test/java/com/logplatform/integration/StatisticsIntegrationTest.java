package com.logplatform.integration;

import com.logplatform.config.TestContainersConfig;
import com.logplatform.dto.StatsSummary;
import com.logplatform.fixture.WebLogFixture;
import com.logplatform.repository.WebLogRepository;
import com.logplatform.service.StatisticsService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.Cache;
import org.springframework.cache.CacheManager;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Teste de integração das Estatísticas Descritivas (Módulo M-10).
 *
 * Teoria para aula:
 * - Valida o fluxo: ingestão de logs no banco -> StatisticsService.computeSummary() -> agregações
 *   SQL reais no PostgreSQL. Também checa o endpoint HTTP GET /stats/summary de ponta a ponta.
 * - Cobre o caso de banco vazio (deve retornar zeros, sem quebrar) e o caso com dados.
 */
class StatisticsIntegrationTest extends TestContainersConfig {

    @Autowired
    private WebLogRepository webLogRepository;

    @Autowired
    private StatisticsService statisticsService;

    @Autowired
    private CacheManager cacheManager;

    @BeforeEach
    void cleanDatabase() {
        webLogRepository.deleteAll();
        // Atenção: o cache "statistics" usa chave fixa 'summary' (TTL 30s). Se não limparmos,
        // o resultado calculado num teste vazaria para o próximo, mascarando os dados reais.
        Cache statisticsCache = cacheManager.getCache("statistics");
        if (statisticsCache != null) {
            statisticsCache.clear();
        }
    }

    @Test
    void computeSummary_afterIngestion_returnsTotalRecordsGreaterThanZero() {
        // Critério M-10: após ingestão, computeSummary() retorna totalRecords > 0
        webLogRepository.saveAll(WebLogFixture.aListOf(50));

        StatsSummary summary = statisticsService.computeSummary();

        assertThat(summary.getTotalRecords()).isEqualTo(50);
    }

    @Test
    void computeSummary_afterIngestion_calculatesStatisticsCorrectly() {
        // Valida que as métricas estatísticas são calculadas de forma coerente
        webLogRepository.saveAll(WebLogFixture.aListOf(100));

        StatsSummary summary = statisticsService.computeSummary();

        assertThat(summary.getMeanResponseTime()).isGreaterThan(0);
        assertThat(summary.getMedianResponseTime()).isGreaterThan(0);
        // P95 (percentil 95) nunca pode ser menor que a média num conjunto crescente
        assertThat(summary.getPercentile95ResponseTime()).isGreaterThanOrEqualTo(summary.getMeanResponseTime());
        assertThat(summary.getErrorRate()).isGreaterThan(0);
        assertThat(summary.getMethodFrequency()).isNotEmpty();
        assertThat(summary.getStatusCodeFrequencyAbsolute()).isNotEmpty();
    }

    @Test
    void computeSummary_emptyDatabase_returnsZeroedStats() {
        // Sem dados, o resumo deve vir zerado (e não lançar exceção)
        StatsSummary summary = statisticsService.computeSummary();

        assertThat(summary.getTotalRecords()).isZero();
        assertThat(summary.getMeanResponseTime()).isZero();
        assertThat(summary.getErrorRate()).isZero();
    }

    @Test
    void statsEndpoint_afterIngestion_returns200WithData() {
        // Mesma validação, mas pela porta de entrada real: o endpoint HTTP
        webLogRepository.saveAll(WebLogFixture.aListOf(20));

        ResponseEntity<StatsSummary> response = restTemplate.getForEntity(
                baseUrl + "/stats/summary", StatsSummary.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull();
        assertThat(response.getBody().getTotalRecords()).isEqualTo(20);
    }

    @Test
    void statsEndpoint_emptyDatabase_returns200WithZeros() {
        // Banco vazio pelo endpoint: deve responder 200 com totais zerados
        ResponseEntity<StatsSummary> response = restTemplate.getForEntity(
                baseUrl + "/stats/summary", StatsSummary.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull();
        assertThat(response.getBody().getTotalRecords()).isZero();
    }
}
