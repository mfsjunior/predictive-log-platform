package com.logplatform.mapper;

import com.logplatform.domain.model.WebLogDomain;
import com.logplatform.dto.WebLogResponse;
import com.logplatform.entity.WebLog;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;

import static org.junit.jupiter.api.Assertions.*;

class WebLogMapperTest {

    private final WebLogMapper mapper = new WebLogMapper();

    @Test
    void toDomain_shouldMapCorrectly() {
        LocalDateTime now = LocalDateTime.now();
        WebLog entity = WebLog.builder()
                .id(1L)
                .timestamp(now)
                .method("GET")
                .path("/test")
                .statusCode(200)
                .responseTimeMs(120.5)
                .userAgent("Mozilla")
                .ipAddress("127.0.0.1")
                .bytesSent(500)
                .createdAt(now)
                .build();

        WebLogDomain domain = mapper.toDomain(entity);

        assertNotNull(domain);
        assertEquals(entity.getId(), domain.getId());
        assertEquals(entity.getTimestamp(), domain.getTimestamp());
        assertEquals(entity.getMethod(), domain.getMethod());
        assertEquals(entity.getPath(), domain.getPath());
        assertEquals(entity.getStatusCode(), domain.getStatusCode());
        assertEquals(entity.getResponseTimeMs(), domain.getResponseTimeMs());
        assertEquals(entity.getUserAgent(), domain.getUserAgent());
        assertEquals(entity.getIpAddress(), domain.getIpAddress());
        assertEquals(entity.getBytesSent(), domain.getBytesSent());
        assertEquals(entity.getCreatedAt(), domain.getCreatedAt());
    }

    @Test
    void toDomain_shouldHandleNulls() {
        assertNull(mapper.toDomain(null));

        WebLog entity = WebLog.builder()
                .id(1L)
                .timestamp(null)
                .method(null)
                .path(null)
                .statusCode(null)
                .responseTimeMs(null)
                .userAgent(null)
                .ipAddress(null)
                .bytesSent(null)
                .createdAt(null)
                .build();

        WebLogDomain domain = mapper.toDomain(entity);

        assertNotNull(domain);
        assertEquals(0, domain.getStatusCode());
        assertEquals(0.0, domain.getResponseTimeMs());
        assertEquals(0, domain.getBytesSent());
        assertNull(domain.getTimestamp());
        assertNull(domain.getMethod());
    }

    @Test
    void toEntity_shouldMapCorrectly() {
        LocalDateTime now = LocalDateTime.now();
        WebLogDomain domain = new WebLogDomain(
                now, "POST", "/api/save", 201, 80.0, "curl", "192.168.0.1", 200
        );
        domain.setId(2L);
        domain.setCreatedAt(now);

        WebLog entity = mapper.toEntity(domain);

        assertNotNull(entity);
        assertEquals(domain.getId(), entity.getId());
        assertEquals(domain.getTimestamp(), entity.getTimestamp());
        assertEquals(domain.getMethod(), entity.getMethod());
        assertEquals(domain.getPath(), entity.getPath());
        assertEquals(domain.getStatusCode(), entity.getStatusCode());
        assertEquals(domain.getResponseTimeMs(), entity.getResponseTimeMs());
        assertEquals(domain.getUserAgent(), entity.getUserAgent());
        assertEquals(domain.getIpAddress(), entity.getIpAddress());
        assertEquals(domain.getBytesSent(), entity.getBytesSent());
        assertEquals(domain.getCreatedAt(), entity.getCreatedAt());
    }

    @Test
    void toEntity_shouldHandleNull() {
        assertNull(mapper.toEntity(null));
    }

    @Test
    void toResponse_shouldMapCorrectly() {
        LocalDateTime now = LocalDateTime.now();
        WebLogDomain domain = new WebLogDomain(
                now, "GET", "/api/data", 200, 15.2, "browser", "10.0.0.2", 150
        );
        domain.setId(3L);

        WebLogResponse response = mapper.toResponse(domain);

        assertNotNull(response);
        assertEquals(domain.getId(), response.getId());
        assertEquals(domain.getTimestamp(), response.getTimestamp());
        assertEquals(domain.getMethod(), response.getMethod());
        assertEquals(domain.getPath(), response.getPath());
        assertEquals(domain.getStatusCode(), response.getStatusCode());
        assertEquals(domain.getResponseTimeMs(), response.getResponseTimeMs());
        assertEquals(domain.getUserAgent(), response.getUserAgent());
        assertEquals(domain.getIpAddress(), response.getIpAddress());
        assertEquals(domain.getBytesSent(), response.getBytesSent());
    }

    @Test
    void toResponse_shouldHandleNull() {
        assertNull(mapper.toResponse(null));
    }
}
