package com.logplatform.infrastructure.health;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

/**
 * Health Indicator para o serviço Python ML Service.
 * Faz ping no endpoint /health do Python a cada verificação.
 * Timeout: máximo 2 segundos para não travar o health check geral.
 */
@Component
@Slf4j
public class MlServiceHealthIndicator implements HealthIndicator {

    @Value("${ml-service.url:http://python-ml:8000}")
    private String mlServiceUrl;

    private final RestTemplate restTemplate;

    public MlServiceHealthIndicator(@Autowired(required = false) RestTemplate restTemplate) {
        // Se houver um bean específico para health checks, use-o; senão, crie um default
        this.restTemplate = restTemplate != null ? restTemplate : createDefaultRestTemplate();
    }

    @Override
    public Health health() {
        long start = System.nanoTime();
        String healthUrl = mlServiceUrl + "/health";

        try {
            // Tenta fazer um GET para /health do Python com timeout de 2 segundos
            restTemplate.getForObject(healthUrl, String.class);
            long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);

            return Health.up()
                    .withDetail("url", mlServiceUrl)
                    .withDetail("responseTimeMs", elapsed)
                    .withDetail("endpoint", healthUrl)
                    .build();

        } catch (Exception e) {
            long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
            log.warn("ML Service health check failed: {}", e.getMessage());

            return Health.down()
                    .withDetail("url", mlServiceUrl)
                    .withDetail("error", e.getMessage())
                    .withDetail("responseTimeMs", elapsed)
                    .build();
        }
    }

    private RestTemplate createDefaultRestTemplate() {
        return new RestTemplateBuilder()
                .setConnectTimeout(Duration.ofSeconds(2))
                .setReadTimeout(Duration.ofSeconds(2))
                .build();
    }
}
