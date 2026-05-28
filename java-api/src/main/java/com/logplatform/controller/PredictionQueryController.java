package com.logplatform.controller;

import com.logplatform.dto.PagedResponse;
import com.logplatform.dto.PredictionResponse;
import com.logplatform.entity.Prediction;
import com.logplatform.exception.BusinessException;
import com.logplatform.repository.PredictionRepository;
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
@RequestMapping("/predictions")
@RequiredArgsConstructor
@Validated
@Tag(name = "Prediction Queries", description = "Paginated prediction history retrieval")
public class PredictionQueryController {

    private final PredictionRepository predictionRepository;

    @GetMapping
    @Operation(summary = "List predictions paginated", description = "List prediction history with pagination and optional type filtering")
    public ResponseEntity<PagedResponse<PredictionResponse>> listPredictions(
            @PageableDefault(size = 20, sort = "createdAt", direction = Sort.Direction.DESC) Pageable pageable,
            @RequestParam(required = false) String type) {

        if (pageable.getPageSize() > 100) {
            throw new BusinessException("page size must be less than or equal to 100", 400);
        }

        String normalizedType = (type != null && type.isBlank()) ? null : type;

        Page<Prediction> page = predictionRepository.findAllByFilters(normalizedType, pageable);
        return ResponseEntity.ok(PagedResponse.from(page.map(PredictionResponse::from)));
    }
}
