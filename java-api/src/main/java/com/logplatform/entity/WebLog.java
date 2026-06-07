package com.logplatform.entity;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;
import org.hibernate.annotations.SQLDelete;
import org.hibernate.annotations.Where;

/**
 * Entidade JPA que representa um registro de log web.
 * 
 * Teoria para aula:
 * - Granularidade: Cada linha do arquivo CSV enviado pelo usuário se torna um objeto 
 *   desta classe no banco de dados.
 * - @Table(name = "web_logs"): Define o nome da tabela física no PostgreSQL onde os 
 *   logs brutos serão armazenados para posterior análise estatística e de ML.
 */
@Entity // Define que esta classe é uma entidade gerenciada pelo JPA e mapeada para uma tabela
@Table(name = "web_logs") // Especifica o nome da tabela no banco de dados
@SQLDelete(sql = "UPDATE web_logs SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?")
@Where(clause = "deleted_at IS NULL")
@Getter // Lombok: Gera automaticamente todos os métodos Getters para os campos
@Setter // Lombok: Gera automaticamente todos os métodos Setters para os campos
@NoArgsConstructor // Lombok: Gera um construtor vazio (exigido pelo JPA)
@AllArgsConstructor // Lombok: Gera um construtor com todos os campos (útil para testes)
@Builder // Lombok: Implementa o padrão de projeto Builder para criação fluenta de objetos
public class WebLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private LocalDateTime timestamp;

    @Column(nullable = false, length = 10)
    private String method;

    @Column(nullable = false, length = 500)
    private String path;

    @Column(name = "status_code", nullable = false)
    private Integer statusCode;

    @Column(name = "response_time_ms", nullable = false)
    private Double responseTimeMs;

    @Column(name = "user_agent", length = 500)
    private String userAgent;

    @Column(name = "ip_address", length = 45)
    private String ipAddress;

    @Column(name = "bytes_sent")
    private Integer bytesSent;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @Column(name = "deleted_at")
    private LocalDateTime deletedAt;

    @PrePersist
    protected void onCreate() {
        // Garante que o registro no banco tenha o horário exato da inserção
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
    }
}
