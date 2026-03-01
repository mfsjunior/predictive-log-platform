package com.logplatform.infrastructure.adapter;

import com.logplatform.domain.model.WebLogDomain;
import com.logplatform.domain.port.LogRepository;
import com.logplatform.entity.WebLog;
import com.logplatform.repository.WebLogRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Infrastructure adapter: implements domain's LogRepository using JPA.
 * Translates between domain model (WebLogDomain) and JPA entity (WebLog).
 */
@Component
@RequiredArgsConstructor
public class JpaLogRepositoryAdapter implements LogRepository {

    private final WebLogRepository jpaRepository;

    @Override
    public void saveAll(List<WebLogDomain> logs) {
        List<WebLog> entities = logs.stream()
                .map(this::toEntity)
                .collect(Collectors.toList());
        jpaRepository.saveAll(entities);
    }

    @Override
    public List<WebLogDomain> findAll() {
        return jpaRepository.findAll().stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }

    @Override
    public long count() {
        return jpaRepository.count();
    }

    private WebLog toEntity(WebLogDomain domain) {
        return WebLog.builder()
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

    private WebLogDomain toDomain(WebLog entity) {
        WebLogDomain domain = new WebLogDomain(
                entity.getTimestamp(),
                entity.getMethod(),
                entity.getPath(),
                entity.getStatusCode(),
                entity.getResponseTimeMs(),
                entity.getUserAgent(),
                entity.getIpAddress(),
                entity.getBytesSent() != null ? entity.getBytesSent() : 0);
        domain.setId(entity.getId());
        domain.setCreatedAt(entity.getCreatedAt());
        return domain;
    }
}
