package com.logplatform.infrastructure.kafka;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.logplatform.domain.model.WebLogDomain;
import com.logplatform.domain.port.LogRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Consumes raw log events from Kafka, persists to database via domain port.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class LogEventConsumer {

    private final LogRepository logRepository;
    private final ObjectMapper objectMapper;

    @KafkaListener(topics = KafkaConfig.LOGS_RAW_TOPIC, groupId = "plip-log-consumer", containerFactory = "kafkaListenerContainerFactory")
    public void consume(String message) {
        try {
            WebLogDomain logEntry = objectMapper.readValue(message, WebLogDomain.class);
            logRepository.saveAll(List.of(logEntry));

            if (logEntry.isError()) {
                log.warn("Error log ingested via Kafka: {} {} → {}",
                        logEntry.getMethod(), logEntry.getPath(), logEntry.getStatusCode());
            }
        } catch (Exception e) {
            log.error("Failed to process Kafka log event: {}", e.getMessage());
        }
    }
}
