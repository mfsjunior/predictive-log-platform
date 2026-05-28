package com.logplatform.infrastructure.health;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.stereotype.Component;

/**
 * Health Indicator para Redis.
 * Verifica a conexão com o servidor Redis através da RedisConnectionFactory do Spring.
 */
@Component
@Slf4j
public class RedisHealthIndicator implements HealthIndicator {

    private final RedisConnectionFactory redisConnectionFactory;

    public RedisHealthIndicator(RedisConnectionFactory redisConnectionFactory) {
        this.redisConnectionFactory = redisConnectionFactory;
    }

    @Override
    public Health health() {
        long start = System.currentTimeMillis();

        try {
            // Obtém uma conexão do factory e testa PING
            var connection = redisConnectionFactory.getConnection();
            if (connection != null) {
                connection.ping();
                connection.close();
                long elapsed = System.currentTimeMillis() - start;

                return Health.up()
                        .withDetail("connection", "Redis (Spring Data)")
                        .withDetail("responseTimeMs", elapsed)
                        .build();
            } else {
                return Health.down()
                        .withDetail("error", "Failed to get Redis connection")
                        .build();
            }

        } catch (Exception e) {
            long elapsed = System.currentTimeMillis() - start;
            log.warn("Redis health check failed: {}", e.getMessage());

            return Health.down()
                    .withDetail("error", e.getMessage())
                    .withDetail("responseTimeMs", elapsed)
                    .build();
        }
    }
}
