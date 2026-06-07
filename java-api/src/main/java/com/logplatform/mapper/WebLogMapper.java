package com.logplatform.mapper;

import com.logplatform.domain.model.WebLogDomain;
import com.logplatform.dto.WebLogResponse;
import com.logplatform.entity.WebLog;
import org.springframework.stereotype.Component;

/**
 * Mapper manual para converter entre Entidade JPA WebLog, Domínio WebLogDomain e DTO WebLogResponse.
 */
@Component
public class WebLogMapper {

    public WebLogDomain toDomain(WebLog entity) {
        if (entity == null) {
            return null;
        }

        WebLogDomain domain = new WebLogDomain();
        domain.setId(entity.getId());
        domain.setTimestamp(entity.getTimestamp());
        domain.setMethod(entity.getMethod());
        domain.setPath(entity.getPath());
        domain.setStatusCode(entity.getStatusCode() != null ? entity.getStatusCode() : 0);
        domain.setResponseTimeMs(entity.getResponseTimeMs() != null ? entity.getResponseTimeMs() : 0.0);
        domain.setUserAgent(entity.getUserAgent());
        domain.setIpAddress(entity.getIpAddress());
        domain.setBytesSent(entity.getBytesSent() != null ? entity.getBytesSent() : 0);
        domain.setCreatedAt(entity.getCreatedAt());
        return domain;
    }

    public WebLog toEntity(WebLogDomain domain) {
        if (domain == null) {
            return null;
        }

        return WebLog.builder()
                .id(domain.getId())
                .timestamp(domain.getTimestamp())
                .method(domain.getMethod())
                .path(domain.getPath())
                .statusCode(domain.getStatusCode())
                .responseTimeMs(domain.getResponseTimeMs())
                .userAgent(domain.getUserAgent())
                .ipAddress(domain.getIpAddress())
                .bytesSent(domain.getBytesSent())
                .createdAt(domain.getCreatedAt())
                .build();
    }

    public WebLogResponse toResponse(WebLogDomain domain) {
        if (domain == null) {
            return null;
        }

        return WebLogResponse.builder()
                .id(domain.getId())
                .timestamp(domain.getTimestamp())
                .method(domain.getMethod())
                .path(domain.getPath())
                .statusCode(domain.getStatusCode())
                .responseTimeMs(domain.getResponseTimeMs())
                .userAgent(domain.getUserAgent())
                .ipAddress(domain.getIpAddress())
                .bytesSent(domain.getBytesSent())
                .build();
    }
}
