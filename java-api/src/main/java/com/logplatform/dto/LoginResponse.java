package com.logplatform.dto;

public record LoginResponse(
    String token,
    String type,
    String username
) {}