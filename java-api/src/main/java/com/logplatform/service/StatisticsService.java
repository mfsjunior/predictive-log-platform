package com.logplatform.service;

import com.logplatform.dto.StatsSummary;
import com.logplatform.repository.WebLogRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class StatisticsService {

    private final WebLogRepository webLogRepository;

    public StatsSummary computeSummary() {
        long totalRecords = webLogRepository.count();

        if (totalRecords == 0) {
            return StatsSummary.builder()
                    .totalRecords(0)
                    .statusCodeFrequencyAbsolute(Collections.emptyMap())
                    .statusCodeFrequencyRelative(Collections.emptyMap())
                    .methodFrequency(Collections.emptyMap())
                    .meanResponseTime(0)
                    .medianResponseTime(0)
                    .stdDevResponseTime(0)
                    .percentile95ResponseTime(0)
                    .errorRate(0)
                    .peakHour(0)
                    .peakHourCount(0)
                    .build();
        }

        // Status code frequencies
        List<Object[]> statusCounts = webLogRepository.countByStatusCode();
        Map<Integer, Long> absoluteFreq = new LinkedHashMap<>();
        Map<Integer, Double> relativeFreq = new LinkedHashMap<>();
        for (Object[] row : statusCounts) {
            int code = (Integer) row[0];
            long count = (Long) row[1];
            absoluteFreq.put(code, count);
            relativeFreq.put(code, Math.round((double) count / totalRecords * 10000.0) / 10000.0);
        }

        // Method frequencies
        List<Object[]> methodCounts = webLogRepository.countByMethod();
        Map<String, Long> methodFreq = new LinkedHashMap<>();
        for (Object[] row : methodCounts) {
            methodFreq.put((String) row[0], (Long) row[1]);
        }

        // Response time statistics
        List<Double> responseTimes = webLogRepository.findAllResponseTimesOrdered();
        double mean = responseTimes.stream().mapToDouble(d -> d).average().orElse(0);
        double median = computeMedian(responseTimes);
        double stdDev = computeStdDev(responseTimes, mean);
        double p95 = computePercentile(responseTimes, 95);

        // Error rate
        long errorCount = webLogRepository.countByStatusCodeGreaterThanEqual(400);
        double errorRate = (double) errorCount / totalRecords;

        // Peak hour
        List<Object[]> hourCounts = webLogRepository.countByHour();
        int peakHour = 0;
        long peakCount = 0;
        if (!hourCounts.isEmpty()) {
            Object hourVal = hourCounts.get(0)[0];
            peakHour = hourVal instanceof Number ? ((Number) hourVal).intValue() : 0;
            peakCount = (Long) hourCounts.get(0)[1];
        }

        return StatsSummary.builder()
                .totalRecords(totalRecords)
                .statusCodeFrequencyAbsolute(absoluteFreq)
                .statusCodeFrequencyRelative(relativeFreq)
                .methodFrequency(methodFreq)
                .meanResponseTime(Math.round(mean * 100.0) / 100.0)
                .medianResponseTime(Math.round(median * 100.0) / 100.0)
                .stdDevResponseTime(Math.round(stdDev * 100.0) / 100.0)
                .percentile95ResponseTime(Math.round(p95 * 100.0) / 100.0)
                .errorRate(Math.round(errorRate * 10000.0) / 10000.0)
                .peakHour(peakHour)
                .peakHourCount(peakCount)
                .build();
    }

    private double computeMedian(List<Double> sorted) {
        if (sorted.isEmpty()) return 0;
        int n = sorted.size();
        if (n % 2 == 0) {
            return (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
        }
        return sorted.get(n / 2);
    }

    private double computeStdDev(List<Double> values, double mean) {
        if (values.size() < 2) return 0;
        double sumSquaredDiff = values.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .sum();
        return Math.sqrt(sumSquaredDiff / (values.size() - 1));
    }

    private double computePercentile(List<Double> sorted, int percentile) {
        if (sorted.isEmpty()) return 0;
        int index = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        index = Math.max(0, Math.min(index, sorted.size() - 1));
        return sorted.get(index);
    }
}
