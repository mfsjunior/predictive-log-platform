package com.logplatform;

import com.logplatform.entity.WebLog;
import com.logplatform.repository.WebLogRepository;
import com.logplatform.service.StatisticsService;
import com.logplatform.dto.StatsSummary;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.test.context.ActiveProfiles;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests using Spring Boot's embedded test context with H2.
 * For full Testcontainers integration, Docker must be available.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
class IntegrationTest {

    @LocalServerPort
    private int port;

    @Autowired
    private TestRestTemplate restTemplate;

    @Autowired
    private WebLogRepository webLogRepository;

    @Autowired
    private StatisticsService statisticsService;

    private String baseUrl;

    @BeforeEach
    void setUp() {
        baseUrl = "http://localhost:" + port;
        webLogRepository.deleteAll();
    }

    @Test
    void contextLoads() {
        // Verify application context starts successfully
    }

    @Test
    void statsSummary_withData_shouldReturnStatistics() {
        // Insert test data
        List<WebLog> logs = new ArrayList<>();
        LocalDateTime base = LocalDateTime.of(2025, 1, 15, 10, 0, 0);

        for (int i = 0; i < 100; i++) {
            logs.add(WebLog.builder()
                    .timestamp(base.plusMinutes(i))
                    .method(i % 3 == 0 ? "POST" : "GET")
                    .path("/api/test")
                    .statusCode(i % 10 == 0 ? 500 : (i % 7 == 0 ? 404 : 200))
                    .responseTimeMs(100.0 + (i * 5.0))
                    .ipAddress("192.168.1.1")
                    .bytesSent(1024)
                    .build());
        }
        webLogRepository.saveAll(logs);

        // Test stats endpoint
        ResponseEntity<StatsSummary> response = restTemplate.getForEntity(
                baseUrl + "/stats/summary", StatsSummary.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull();

        StatsSummary stats = response.getBody();
        assertThat(stats.getTotalRecords()).isEqualTo(100);
        assertThat(stats.getMeanResponseTime()).isGreaterThan(0);
        assertThat(stats.getMedianResponseTime()).isGreaterThan(0);
        assertThat(stats.getErrorRate()).isGreaterThan(0);
        assertThat(stats.getPercentile95ResponseTime()).isGreaterThan(stats.getMeanResponseTime());
    }

    @Test
    void statsSummary_emptyDatabase_shouldReturnZeros() {
        ResponseEntity<StatsSummary> response = restTemplate.getForEntity(
                baseUrl + "/stats/summary", StatsSummary.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull();
        assertThat(response.getBody().getTotalRecords()).isEqualTo(0);
    }

    @Test
    void webLogRepository_shouldPersistAndRetrieve() {
        WebLog log = WebLog.builder()
                .timestamp(LocalDateTime.now())
                .method("GET")
                .path("/api/test")
                .statusCode(200)
                .responseTimeMs(150.0)
                .ipAddress("10.0.0.1")
                .bytesSent(512)
                .build();

        WebLog saved = webLogRepository.save(log);

        assertThat(saved.getId()).isNotNull();
        assertThat(webLogRepository.findById(saved.getId())).isPresent();
    }

    @Test
    void swaggerUi_shouldBeAccessible() {
        ResponseEntity<String> response = restTemplate.getForEntity(
                baseUrl + "/swagger-ui/index.html", String.class);

        // Swagger UI should respond (may redirect)
        assertThat(response.getStatusCode().value()).isLessThan(400);
    }

    @Test
    void actuatorHealth_shouldBeAccessible() {
        ResponseEntity<String> response = restTemplate.getForEntity(
                baseUrl + "/actuator/health", String.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).contains("UP");
    }
}
