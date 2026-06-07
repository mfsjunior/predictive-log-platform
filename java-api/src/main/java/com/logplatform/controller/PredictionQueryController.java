package com.logplatform.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.logplatform.dto.PagedResponse;
import com.logplatform.dto.PredictionResponse;
import com.logplatform.entity.Prediction;
import com.logplatform.repository.PredictionRepository;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
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
 * Controller para consulta paginada do histórico de predições.
 */
@RestController
@RequestMapping("/predictions")
@RequiredArgsConstructor
@Tag(name = "Prediction History", description = "Browse ML prediction history with pagination and optional filtering")
@SecurityRequirement(name = "bearerAuth")
public class PredictionQueryController {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final PredictionRepository predictionRepository;

    @GetMapping
    @Operation(summary = "List predictions", description = "Retrieve a paginated list of prediction history records")
    public ResponseEntity<PagedResponse<PredictionResponse>> queryPredictions(
            @Parameter(description = "Zero-based page number") @RequestParam(defaultValue = "0") int page,
            @Parameter(description = "Page size (maximum 100)") @RequestParam(defaultValue = "20") int size,
            @Parameter(description = "Sort order, e.g. createdAt,desc") @RequestParam(defaultValue = "createdAt,desc") String sort,
            @Parameter(description = "Filter by prediction type") @RequestParam(required = false) String type) {

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
        Page<Prediction> predictions = predictionRepository.findByPredictionType(type, pageable);

        List<PredictionResponse> content = predictions.stream()
                .map(this::toPredictionResponse)
                .collect(Collectors.toList());

        PagedResponse<PredictionResponse> response = PagedResponse.<PredictionResponse>builder()
                .content(content)
                .page(predictions.getNumber())
                .size(predictions.getSize())
                .totalElements(predictions.getTotalElements())
                .totalPages(predictions.getTotalPages())
                .last(predictions.isLast())
                .build();

        return ResponseEntity.ok(response);
    }

    private PredictionResponse toPredictionResponse(Prediction prediction) {
        return PredictionResponse.builder()
                .id(prediction.getId())
                .predictionType(prediction.getPredictionType())
                .inputData(parseJson(prediction.getInputData()))
                .result(parseJson(prediction.getResult()))
                .modelVersion(prediction.getModelVersion())
                .latencyMs(prediction.getLatencyMs())
                .createdAt(prediction.getCreatedAt())
                .build();
    }

    private Object parseJson(String json) {
        if (json == null) {
            return null;
        }
        try {
            return MAPPER.readValue(json, Object.class);
        } catch (JsonProcessingException e) {
            return json;
        }
    }
}
