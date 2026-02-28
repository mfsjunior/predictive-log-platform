package com.logplatform.config;

import com.logplatform.repository.PredictionRepository;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.actuate.autoconfigure.metrics.MeterRegistryCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@Slf4j
public class MetricsConfig {

    @Bean
    public MeterRegistryCustomizer<MeterRegistry> metricsCustomizer(
            PredictionRepository predictionRepository) {

        return registry -> {
            // Total predictions gauge
            Gauge.builder("ml.predictions.volume",
                            predictionRepository, PredictionRepository::count)
                    .description("Total number of predictions stored")
                    .register(registry);

            // Error predictions gauge
            Gauge.builder("ml.predictions.error.count",
                            predictionRepository,
                            repo -> repo.countByPredictionType("error"))
                    .description("Total error predictions")
                    .register(registry);

            // Response time predictions gauge
            Gauge.builder("ml.predictions.response_time.count",
                            predictionRepository,
                            repo -> repo.countByPredictionType("response_time"))
                    .description("Total response time predictions")
                    .register(registry);

            log.info("Custom ML metrics registered");
        };
    }
}
