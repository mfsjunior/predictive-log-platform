package com.logplatform.infrastructure.kafka;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.common.utils.Bytes;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.kstream.*;
import org.apache.kafka.streams.state.WindowStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.time.Duration;

/**
 * Kafka Streams processor for real-time log aggregation.
 *
 * Topology:
 * plip.logs.raw → 5-minute tumbling window → aggregate(count, avgResponseTime,
 * errorCount)
 * → plip.logs.aggregated
 *
 * Produces per-method stats every 5 minutes:
 * { "method": "GET", "count": 1234, "avg_response_time_ms": 45.3,
 * "error_count": 12 }
 */
@Component
@Slf4j
public class LogStreamProcessor {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Autowired
    public void buildTopology(StreamsBuilder streamsBuilder) {
        KStream<String, String> rawLogs = streamsBuilder.stream(
                KafkaConfig.LOGS_RAW_TOPIC,
                Consumed.with(Serdes.String(), Serdes.String()));

        // 5-minute tumbling window aggregation, keyed by HTTP method
        rawLogs
                .groupByKey(Grouped.with(Serdes.String(), Serdes.String()))
                .windowedBy(TimeWindows.ofSizeWithNoGrace(Duration.ofMinutes(5)))
                .aggregate(
                        // Initializer: { "count": 0, "total_response_time": 0.0, "error_count": 0 }
                        () -> "{\"count\":0,\"total_response_time\":0.0,\"error_count\":0}",
                        // Aggregator: accumulate count, response time sum, error count
                        (key, value, aggregate) -> {
                            try {
                                JsonNode logNode = objectMapper.readTree(value);
                                JsonNode aggNode = objectMapper.readTree(aggregate);

                                long count = aggNode.get("count").asLong() + 1;
                                double totalRt = aggNode.get("total_response_time").asDouble()
                                        + logNode.path("responseTimeMs").asDouble(0.0);
                                long errors = aggNode.get("error_count").asLong()
                                        + (logNode.path("statusCode").asInt(200) >= 400 ? 1 : 0);

                                return String.format(
                                        "{\"count\":%d,\"total_response_time\":%.2f,\"error_count\":%d}",
                                        count, totalRt, errors);
                            } catch (Exception e) {
                                log.error("Aggregation error: {}", e.getMessage());
                                return aggregate;
                            }
                        },
                        Materialized.<String, String, WindowStore<Bytes, byte[]>>as("log-stats-store")
                                .withKeySerde(Serdes.String())
                                .withValueSerde(Serdes.String()))
                .suppress(Suppressed.untilWindowCloses(Suppressed.BufferConfig.unbounded()))
                .toStream()
                .map((windowedKey, value) -> {
                    try {
                        JsonNode agg = objectMapper.readTree(value);
                        long count = agg.get("count").asLong();
                        double totalRt = agg.get("total_response_time").asDouble();
                        long errors = agg.get("error_count").asLong();
                        double avgRt = count > 0 ? totalRt / count : 0.0;

                        String result = String.format(
                                "{\"method\":\"%s\",\"window_start\":\"%s\",\"window_end\":\"%s\","
                                        + "\"count\":%d,\"avg_response_time_ms\":%.2f,\"error_count\":%d}",
                                windowedKey.key(),
                                windowedKey.window().startTime(),
                                windowedKey.window().endTime(),
                                count, avgRt, errors);

                        log.info("Window aggregation: method={}, count={}, avgRT={:.2f}ms, errors={}",
                                windowedKey.key(), count, avgRt, errors);

                        return KeyValue.pair(windowedKey.key(), result);
                    } catch (Exception e) {
                        log.error("Window result mapping error: {}", e.getMessage());
                        return KeyValue.pair(windowedKey.key(), value);
                    }
                })
                .to(KafkaConfig.LOGS_AGGREGATED_TOPIC, Produced.with(Serdes.String(), Serdes.String()));
    }
}
