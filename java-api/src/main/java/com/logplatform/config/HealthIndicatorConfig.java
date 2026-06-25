package com.logplatform.config;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.boot.actuate.health.Status;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import org.springframework.data.redis.connection.RedisConnection;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.kafka.core.KafkaAdmin;
import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.concurrent.TimeUnit;
import org.apache.kafka.clients.admin.AdminClient;
import org.apache.kafka.clients.admin.DescribeClusterResult;

import java.time.Duration;
import java.util.Map;

/**
 * Configura indicadores de saúde customizados para dependências externas.
 */
@Configuration
@RequiredArgsConstructor
public class HealthIndicatorConfig {

    private final WebClient.Builder webClientBuilder;

    @Value("${ml.service.url}")
    private String mlServiceUrl;

    @Value("${ml.service.health-path:/health}")
    private String mlHealthPath;

    @Value("${ml.service.timeout-ms:5000}")
    private int mlTimeoutMs;

    @Bean
    public HealthIndicator mlServiceHealthIndicator() {
        WebClient client = webClientBuilder
                .baseUrl(mlServiceUrl)
                .build();

        return () -> {
            try {
                Map response = client.get()
                        .uri(mlHealthPath)
                        .retrieve()
                        .bodyToMono(Map.class)
                        .timeout(Duration.ofMillis(mlTimeoutMs))
                        .block();

                if (response == null || !response.containsKey("status")) {
                    return Health.status(Status.UNKNOWN)
                            .withDetail("mlService", "invalid response")
                            .build();
                }

                String status = String.valueOf(response.get("status"));
                return "healthy".equalsIgnoreCase(status)
                        ? Health.up().withDetail("mlService", "healthy").build()
                        : Health.status(Status.DOWN)
                                .withDetail("mlService", status)
                                .build();
            } catch (WebClientResponseException ex) {
                return Health.down(ex)
                        .withDetail("statusCode", ex.getStatusCode().value())
                        .withDetail("error", ex.getMessage())
                        .build();
            } catch (WebClientRequestException ex) {
                return Health.down(ex)
                        .withDetail("error", ex.getMessage())
                        .build();
            } catch (Exception ex) {
                return Health.down(ex)
                        .withDetail("error", ex.getMessage())
                        .build();
            }
        };
    }

    @Bean
    public HealthIndicator redisHealthIndicator(RedisConnectionFactory factory) {
        return () -> {
            try (RedisConnection connection = factory.getConnection()) {
                String ping = connection.ping();
                if ("PONG".equalsIgnoreCase(ping)) {
                    return Health.up().withDetail("redis", "reachable").build();
                }
                return Health.down().withDetail("redis", "unreachable").build();
            } catch (Exception ex) {
                return Health.down(ex).withDetail("error", ex.getMessage()).build();
            }
        };
    }

    @Bean
    public HealthIndicator kafkaHealthIndicator(KafkaAdmin kafkaAdmin) {
        return () -> {
            try (AdminClient client = AdminClient.create(kafkaAdmin.getConfigurationProperties())) {
                DescribeClusterResult cluster = client.describeCluster();
                cluster.nodes().get(5, TimeUnit.SECONDS);
                return Health.up().withDetail("kafka", "reachable").build();
            } catch (Exception ex) {
                return Health.down(ex).withDetail("error", ex.getMessage()).build();
            }
        };
    }

    @Bean
    public HealthIndicator dbHealthIndicator(DataSource dataSource) {
        return () -> {
            try (Connection connection = dataSource.getConnection()) {
                if (connection.isValid(2)) {
                    return Health.up().withDetail("database", "reachable").build();
                }
                return Health.down().withDetail("database", "invalid connection").build();
            } catch (SQLException ex) {
                return Health.down(ex).withDetail("error", ex.getMessage()).build();
            }
        };
    }
}