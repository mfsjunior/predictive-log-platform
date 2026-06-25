package com.logplatform.controller;

import com.logplatform.dto.PagedResponse;
import com.logplatform.dto.WebLogResponse;
import com.logplatform.entity.WebLog;
import com.logplatform.repository.WebLogRepository;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Controller para listagem paginada de logs.
 */
@RestController
@RequestMapping("/logs")
@RequiredArgsConstructor
@Tag(name = "Log Query", description = "List and filter log records with pagination")
public class LogQueryController {

    private final WebLogRepository webLogRepository;

    @GetMapping
    @Operation(summary = "List logs", description = "Retrieve a paginated list of logs excluding soft-deleted entries")
    public ResponseEntity<PagedResponse<WebLogResponse>> queryLogs(
            @Parameter(description = "Zero-based page number") @RequestParam(defaultValue = "0") int page,
            @Parameter(description = "Page size (maximum 100)") @RequestParam(defaultValue = "20") int size,
            @Parameter(description = "Sort order, e.g. timestamp,desc") @RequestParam(defaultValue = "timestamp,desc") String sort,
            @Parameter(description = "Filter by HTTP method") @RequestParam(required = false) String method,
            @Parameter(description = "Filter by HTTP status code") @RequestParam(required = false) Integer statusCode) {

        if (size <= 0) {
            throw new IllegalArgumentException("size must be greater than 0");
        }
        if (size > 100) {
            throw new IllegalArgumentException("size must be <= 100");
        }

        String[] sortParts = sort.split(",", 2);
        String sortField = sortParts[0];
        Sort.Direction direction = Sort.Direction.DESC;
        if (sortParts.length > 1) {
            direction = Sort.Direction.fromString(sortParts[1]);
        }

        Pageable pageable = PageRequest.of(page, size, Sort.by(direction, sortField));
        Page<WebLog> logs = webLogRepository.findAllByMethodAndStatusCode(method, statusCode, pageable);

        List<WebLogResponse> content = logs.stream()
                .map(this::toWebLogResponse)
                .collect(Collectors.toList());

        PagedResponse<WebLogResponse> response = PagedResponse.<WebLogResponse>builder()
                .content(content)
                .page(logs.getNumber())
                .size(logs.getSize())
                .totalElements(logs.getTotalElements())
                .totalPages(logs.getTotalPages())
                .last(logs.isLast())
                .build();

        return ResponseEntity.ok(response);
    }

    private WebLogResponse toWebLogResponse(WebLog log) {
        return WebLogResponse.builder()
                .id(log.getId())
                .timestamp(log.getTimestamp())
                .method(log.getMethod())
                .path(log.getPath())
                .statusCode(log.getStatusCode())
                .responseTimeMs(log.getResponseTimeMs())
                .userAgent(log.getUserAgent())
                .ipAddress(log.getIpAddress())
                .bytesSent(log.getBytesSent())
                .createdAt(log.getCreatedAt())
                .build();
    }
}
