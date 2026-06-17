package com.logplatform.infrastructure.adapter;

import com.logplatform.domain.model.WebLogDomain;
import com.logplatform.domain.port.LogRepository;
import com.logplatform.entity.WebLog;
import com.logplatform.mapper.WebLogMapper;
import com.logplatform.repository.WebLogRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Adaptador de Infraestrutura: Implementa a porta 'LogRepository' via JPA usando WebLogMapper.
 */
@Component
@RequiredArgsConstructor
public class JpaLogRepositoryAdapter implements LogRepository {

    private final WebLogRepository jpaRepository;
    private final WebLogMapper webLogMapper;

    @Override
    public void saveAll(List<WebLogDomain> logs) {
        if (logs == null) return;
        List<WebLog> entities = logs.stream()
                .map(webLogMapper::toEntity)
                .collect(Collectors.toList());
        jpaRepository.saveAll(entities);
    }

    @Override
    public List<WebLogDomain> findAll() {
        return jpaRepository.findAll().stream()
                .map(webLogMapper::toDomain)
                .collect(Collectors.toList());
    }

    @Override
    public long count() {
        return jpaRepository.count();
    }
}
