package com.logplatform.controller;

import com.logplatform.dto.StatsSummary;
import com.logplatform.service.StatisticsService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/stats")
@RequiredArgsConstructor
@Tag(name = "Statistics", description = "Descriptive statistics for ingested logs")
public class StatsController {

    private final StatisticsService statisticsService;

    @GetMapping("/summary")
    @Operation(
            summary = "Get descriptive statistics",
            description = "Returns frequency distributions, mean, median, std dev, P95, peak hour, and error rate"
    )
    public ResponseEntity<StatsSummary> getSummary() {
        StatsSummary summary = statisticsService.computeSummary();
        return ResponseEntity.ok(summary);
    }
}
