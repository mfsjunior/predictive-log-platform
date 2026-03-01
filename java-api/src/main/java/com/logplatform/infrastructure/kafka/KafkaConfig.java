package com.logplatform.infrastructure.kafka;

import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;

/**
 * Kafka topic and configuration.
 * Creates topics on startup if they don't exist.
 */
@Configuration
public class KafkaConfig {

    public static final String LOGS_RAW_TOPIC = "plip.logs.raw";
    public static final String LOGS_AGGREGATED_TOPIC = "plip.logs.aggregated";

    @Bean
    public NewTopic logsRawTopic() {
        return TopicBuilder.name(LOGS_RAW_TOPIC)
                .partitions(3)
                .replicas(1)
                .build();
    }

    @Bean
    public NewTopic logsAggregatedTopic() {
        return TopicBuilder.name(LOGS_AGGREGATED_TOPIC)
                .partitions(3)
                .replicas(1)
                .build();
    }
}
