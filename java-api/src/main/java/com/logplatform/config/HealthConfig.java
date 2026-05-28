package com.logplatform.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;

/**
 * Configuração de Health Checks e infraestrutura para o sistema.
 * 
 * - RestTemplate: Com timeout customizado para evitar travamentos em health checks
 */
@Configuration
@Slf4j
public class HealthConfig {

    /**
     * RestTemplate com timeout de 2 segundos para health checks do ML Service.
     * Evita que um serviço lento/offline trave o health check geral.
     */
    @Bean(name = "healthCheckRestTemplate")
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        return builder
                .setConnectTimeout(Duration.ofSeconds(2))
                .setReadTimeout(Duration.ofSeconds(2))
                .build();
    }
}
