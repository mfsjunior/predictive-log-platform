package com.logplatform.dto;

import java.time.LocalDateTime;
import java.util.List;

public record ApiErrorResponse(
    LocalDateTime timestamp,
    Integer status,
    String error,
    String message,
    String path,
    List<FieldError> validationErrors
) {
    public record FieldError(String field, String message) {}
}