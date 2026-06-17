package com.logplatform.fixture;

import com.logplatform.entity.WebLog;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Fábrica de entidades WebLog para os testes de integração.
 *
 * Teoria para aula:
 * - Factory de teste: concentra a criação de objetos de domínio usados nos cenários.
 *   Quando o teste precisa de "um log qualquer", chama aqui em vez de repetir o builder.
 * - Isso deixa os testes mais legíveis: o foco fica no QUE está sendo validado, não em
 *   COMO montar o objeto.
 */
public final class WebLogFixture {

    // Construtor privado: classe utilitária, não instanciável
    private WebLogFixture() {}

    // Um log de sucesso "padrão" (HTTP 200)
    public static WebLog aWebLog() {
        return WebLog.builder()
                .timestamp(LocalDateTime.of(2025, 6, 1, 10, 0))
                .method("GET")
                .path("/api/test")
                .statusCode(200)
                .responseTimeMs(150.0)
                .userAgent("Mozilla/5.0")
                .ipAddress("192.168.1.1")
                .bytesSent(1024)
                .build();
    }

    // Um log de erro (HTTP 500) com tempo de resposta alto — útil para validar taxa de erro
    public static WebLog anErrorLog() {
        return WebLog.builder()
                .timestamp(LocalDateTime.of(2025, 6, 1, 14, 30))
                .method("POST")
                .path("/api/resource")
                .statusCode(500)
                .responseTimeMs(3200.0)
                .userAgent("curl/8.4.0")
                .ipAddress("10.0.0.1")
                .bytesSent(0)
                .build();
    }

    /**
     * Gera uma lista de logs variados (métodos, status e tempos diferentes).
     * A cada 10 registros insere um erro 500, para que as estatísticas tenham dados realistas.
     */
    public static List<WebLog> aListOf(int count) {
        List<WebLog> logs = new ArrayList<>();
        LocalDateTime base = LocalDateTime.of(2025, 6, 1, 8, 0);
        for (int i = 0; i < count; i++) {
            logs.add(WebLog.builder()
                    .timestamp(base.plusMinutes(i))
                    .method(i % 3 == 0 ? "POST" : "GET")
                    .path("/api/items/" + i)
                    .statusCode(i % 10 == 0 ? 500 : 200)
                    .responseTimeMs(100.0 + i * 2.5)
                    .userAgent("TestAgent/1.0")
                    .ipAddress("172.16.0." + (i % 254 + 1))
                    .bytesSent(512 + i * 10)
                    .build());
        }
        return logs;
    }
}
