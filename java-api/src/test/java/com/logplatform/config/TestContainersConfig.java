package com.logplatform.config;

import org.junit.jupiter.api.BeforeEach;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

/**
 * Classe-base para os testes de integração (Módulo M-10).
 *
 * Teoria para aula:
 * - Teste de Integração: diferente do teste unitário (que isola uma classe), aqui validamos
 *   o sistema funcionando de ponta a ponta — controller, service, repositório e banco REAL.
 * - Testcontainers: biblioteca que sobe contêineres Docker descartáveis (PostgreSQL e Redis)
 *   só durante o teste. Assim não dependemos de um banco instalado na máquina; o ambiente é
 *   idêntico ao de produção e é destruído ao final, evitando "sujeira" entre execuções.
 * - Por que herdar desta classe? Centralizamos aqui o setup dos contêineres e da aplicação,
 *   para que cada teste concreto (filho) só precise focar no cenário que quer validar.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT) // Sobe a app numa porta aleatória livre
@Testcontainers // Habilita o ciclo de vida automático dos contêineres anotados com @Container
public abstract class TestContainersConfig {

    // Contêiner PostgreSQL real: substitui o banco de produção durante os testes
    @Container
    static final PostgreSQLContainer<?> POSTGRES = new PostgreSQLContainer<>("postgres:16-alpine")
            .withDatabaseName("testdb")
            .withUsername("testuser")
            .withPassword("testpass");

    // Contêiner Redis real: necessário porque o StatisticsService usa cache (@Cacheable)
    @Container
    static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7-alpine")
            .withExposedPorts(6379);

    // Kafka é "mockado" (simulado): o fluxo de upload/estatísticas não publica eventos,
    // então não precisamos subir um broker real só para os testes desta camada.
    @MockBean
    @SuppressWarnings("rawtypes")
    KafkaTemplate kafkaTemplate;

    // Cliente HTTP de teste: faz chamadas reais aos endpoints da aplicação no ar
    @Autowired
    protected TestRestTemplate restTemplate;

    // Porta real onde a aplicação subiu (preenchida pelo Spring em tempo de execução)
    @LocalServerPort
    protected int port;

    protected String baseUrl;

    @BeforeEach
    void setUpBaseUrl() {
        // Monta a URL base antes de cada teste, já com a porta sorteada
        baseUrl = "http://localhost:" + port;
    }

    /**
     * Injeta dinamicamente as credenciais dos contêineres nas propriedades do Spring.
     *
     * Teoria para aula: como as portas dos contêineres são aleatórias (sorteadas pelo Docker),
     * não dá para fixá-las num application.yml. O @DynamicPropertySource resolve isso lendo
     * a URL/porta real do contêiner em tempo de execução e sobrescrevendo a configuração.
     */
    @DynamicPropertySource
    static void overrideProperties(DynamicPropertyRegistry registry) {
        // Aponta o datasource da aplicação para o PostgreSQL do contêiner
        registry.add("spring.datasource.url", POSTGRES::getJdbcUrl);
        registry.add("spring.datasource.username", POSTGRES::getUsername);
        registry.add("spring.datasource.password", POSTGRES::getPassword);
        registry.add("spring.datasource.driver-class-name", () -> "org.postgresql.Driver");
        // create-drop: cria o schema ao iniciar e apaga ao final, garantindo banco limpo
        registry.add("spring.jpa.hibernate.ddl-auto", () -> "create-drop");
        registry.add("spring.jpa.database-platform", () -> "org.hibernate.dialect.PostgreSQLDialect");
        // Aponta o cache da aplicação para o Redis do contêiner
        registry.add("spring.data.redis.host", REDIS::getHost);
        registry.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        // Endereço de Kafka inválido de propósito: o bean está mockado, ninguém vai conectar
        registry.add("spring.kafka.bootstrap-servers", () -> "localhost:9999");
    }
}
