package com.logplatform.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.SQLRestriction;
import java.time.LocalDateTime;

/**
 * Entidade JPA que representa uma predição feita pelo modelo de ML.
 * 
 * Teoria para aula:
 * - ORM (Object-Relational Mapping): Técnica usada para mapear objetos Java
 * para tabelas SQL.
 * - @Entity: Diz ao Hibernate que esta classe é uma tabela no banco de dados.
 * - Auditoria: Salvar cada predição permite que, no futuro, possamos comparar o
 * que a
 * IA previu com o que realmente aconteceu, ajudando a retreinar o modelo.
 * - Soft Delete: deleted_at != null significa predição logicamente deletada (preserva histórico).
 */
@Entity // Define que esta classe é uma entidade gerenciada pelo JPA e mapeada para uma
        // tabela
@Table(name = "predictions") // Especifica o nome da tabela no banco de dados para auditoria de ML
@SQLRestriction("deleted_at IS NULL") // Hibernate 6: Filtra automaticamente registros deletados
@Getter // Lombok: Gera automaticamente todos os métodos Getters para os campos
@Setter // Lombok: Gera automaticamente todos os métodos Setters para os campos
@NoArgsConstructor // Lombok: Gera um construtor vazio (exigido pelo JPA)
@AllArgsConstructor // Lombok: Gera um construtor com todos os campos
@Builder // Lombok: Implementa o padrão de projeto Builder para criação fluenta de
         // objetos
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "prediction_type", nullable = false, length = 50)
    private String predictionType;

    @Column(name = "input_data", nullable = false, columnDefinition = "TEXT")
    private String inputData;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String result;

    @Column(name = "model_version", length = 100)
    private String modelVersion;

    @Column(name = "latency_ms")
    private Double latencyMs;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @Column(name = "deleted_at")
    private LocalDateTime deletedAt;

    @PrePersist
    protected void onCreate() {
        // Recurso do JPA para preencher a data de criação automaticamente antes de
        // salvar
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
    }
}
