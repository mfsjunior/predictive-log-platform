package com.logplatform.infrastructure.health;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.stereotype.Component;
import org.springframework.kafka.core.KafkaTemplate;

/**
 * Health Indicator para Apache Kafka.
 * Verifica se há conexão com o broker Kafka através do KafkaTemplate.
 */
@Component
@Slf4j
public class KafkaHealthIndicator implements HealthIndicator {

    private final KafkaTemplate<String, Object> kafkaTemplate;

    public KafkaHealthIndicator(KafkaTemplate<String, Object> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    @Override
    public Health health() {
        long start = System.currentTimeMillis();

        try {
            // Se conseguimos criar um produtor, Kafka está acessível
            kafkaTemplate.getProducerFactory();
            long elapsed = System.currentTimeMillis() - start;

            return Health.up()
                    .withDetail("broker", "Kafka (Apache)")
                    .withDetail("responseTimeMs", elapsed)
                    .withDetail("topics", new String[]{"log.events", "predictions", "anomalies"})
                    .build();

        } catch (Exception e) {
            long elapsed = System.currentTimeMillis() - start;
            log.warn("Kafka health check failed: {}", e.getMessage());

            return Health.down()
                    .withDetail("error", e.getMessage())
                    .withDetail("responseTimeMs", elapsed)
                    .build();
        }
    }
}
