package com.logplatform.controller;

import com.logplatform.dto.StatsSummary;
import com.logplatform.service.StatisticsService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.bean.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.test.web.servlet.MockMvc;
import com.logplatform.config.SecurityConfig;

import java.util.Map;

import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(StatsController.class)
@Import(SecurityConfig.class)
class StatsControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private StatisticsService statisticsService;

    @Test
    void getSummary_shouldReturnStatistics() throws Exception {
        StatsSummary summary = StatsSummary.builder()
                .totalRecords(5000)
                .statusCodeFrequencyAbsolute(Map.of(200, 3500L, 404, 500L, 500, 200L))
                .statusCodeFrequencyRelative(Map.of(200, 0.70, 404, 0.10, 500, 0.04))
                .methodFrequency(Map.of("GET", 2750L, "POST", 1250L))
                .meanResponseTime(285.50)
                .medianResponseTime(220.00)
                .stdDevResponseTime(180.30)
                .percentile95ResponseTime(750.00)
                .errorRate(0.14)
                .peakHour(10)
                .peakHourCount(350)
                .build();

        when(statisticsService.computeSummary()).thenReturn(summary);

        mockMvc.perform(get("/stats/summary"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.totalRecords").value(5000))
                .andExpect(jsonPath("$.meanResponseTime").value(285.50))
                .andExpect(jsonPath("$.medianResponseTime").value(220.00))
                .andExpect(jsonPath("$.stdDevResponseTime").value(180.30))
                .andExpect(jsonPath("$.percentile95ResponseTime").value(750.00))
                .andExpect(jsonPath("$.errorRate").value(0.14))
                .andExpect(jsonPath("$.peakHour").value(10));
    }

    @Test
    void getSummary_shouldReturnEmptyStats_whenNoData() throws Exception {
        StatsSummary emptySummary = StatsSummary.builder()
                .totalRecords(0)
                .statusCodeFrequencyAbsolute(Map.of())
                .statusCodeFrequencyRelative(Map.of())
                .methodFrequency(Map.of())
                .meanResponseTime(0)
                .medianResponseTime(0)
                .stdDevResponseTime(0)
                .percentile95ResponseTime(0)
                .errorRate(0)
                .peakHour(0)
                .peakHourCount(0)
                .build();

        when(statisticsService.computeSummary()).thenReturn(emptySummary);

        mockMvc.perform(get("/stats/summary"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.totalRecords").value(0));
    }
}
