package com.logplatform.dto;

import lombok.*;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class StatsSummary {
    private long totalRecords;
    private Map<Integer, Long> statusCodeFrequencyAbsolute;
    private Map<Integer, Double> statusCodeFrequencyRelative;
    private Map<String, Long> methodFrequency;
    private double meanResponseTime;
    private double medianResponseTime;
    private double stdDevResponseTime;
    private double percentile95ResponseTime;
    private double errorRate;
    private Integer peakHour;
    private long peakHourCount;
}
