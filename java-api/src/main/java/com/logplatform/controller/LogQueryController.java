package com.logplatform.controller;

import com.logplatform.dto.PagedResponse;
import com.logplatform.dto.WebLogResponse;
import com.logplatform.entity.WebLog;
import com.logplatform.exception.BusinessException;
import com.logplatform.repository.WebLogRepository;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/logs")
@RequiredArgsConstructor
@Validated
@Tag(name = "Log Queries", description = "Paginated log retrieval")
public class LogQueryController {

    private final WebLogRepository webLogRepository;

    @GetMapping
    @Operation(summary = "List web logs paginated", description = "List web logs with pagination, ordering and optional filters")
    public ResponseEntity<PagedResponse<WebLogResponse>> listLogs(
            @PageableDefault(size = 20, sort = "timestamp", direction = Sort.Direction.DESC) Pageable pageable,
            @RequestParam(required = false) String method,
            @RequestParam(required = false) Integer statusCode) {

        if (pageable.getPageSize() > 100) {
            throw new BusinessException("page size must be less than or equal to 100", 400);
        }

        String normalizedMethod = (method != null && method.isBlank()) ? null : method;
        Integer normalizedStatusCode = statusCode != null && statusCode <= 0 ? null : statusCode;

        Page<WebLog> page = webLogRepository.findAllByFilters(normalizedMethod, normalizedStatusCode, pageable);
        return ResponseEntity.ok(PagedResponse.from(page.map(WebLogResponse::from)));
    }
}
