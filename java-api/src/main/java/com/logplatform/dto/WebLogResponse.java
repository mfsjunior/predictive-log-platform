package com.logplatform.dto;

import com.logplatform.entity.WebLog;

import java.time.LocalDateTime;

public record WebLogResponse(
        Long id,
        LocalDateTime timestamp,
        String method,
        String path,
        Integer statusCode,
        Double responseTimeMs,
        String userAgent,
        String ipAddress,
        Integer bytesSent
) {
    public static WebLogResponse from(WebLog webLog) {
        return new WebLogResponse(
                webLog.getId(),
                webLog.getTimestamp(),
                webLog.getMethod(),
                webLog.getPath(),
                webLog.getStatusCode(),
                webLog.getResponseTimeMs(),
                webLog.getUserAgent(),
                webLog.getIpAddress(),
                webLog.getBytesSent()
        );
    }
}
