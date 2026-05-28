package com.logplatform.repository;

import com.logplatform.entity.WebLog;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Repositório JPA para a entidade WebLog com Consultas Customizadas.
 * 
 * Teoria para aula:
 * - @Query (JPQL): Linguagem de consulta do Java que parece SQL, mas opera sobre 
 *   as classes (Objetos) e não sobre as tabelas diretamente.
 * - Agregação: Usamos GROUP BY e funções como AVG e COUNT para transformar logs 
 *   individuais em métricas de alto nível (ex: média de tempo de resposta).
 */
@Repository
public interface WebLogRepository extends JpaRepository<WebLog, Long> {

    @Query("SELECT w.statusCode, COUNT(w) FROM WebLog w GROUP BY w.statusCode ORDER BY COUNT(w) DESC")
    List<Object[]> countByStatusCode();

    @Query("SELECT w.method, COUNT(w) FROM WebLog w GROUP BY w.method ORDER BY COUNT(w) DESC")
    List<Object[]> countByMethod();

    @Query("SELECT AVG(w.responseTimeMs) FROM WebLog w")
    Double findAverageResponseTime();

    @Query("SELECT HOUR(w.timestamp), COUNT(w) " +
            "FROM WebLog w GROUP BY HOUR(w.timestamp) " +
            "ORDER BY COUNT(w) DESC")
    List<Object[]> countByHour();

    @Query("SELECT w.responseTimeMs FROM WebLog w ORDER BY w.responseTimeMs")
    List<Double> findAllResponseTimesOrdered();

    long countByStatusCodeGreaterThanEqual(int statusCode);

    /**
     * Soft delete: marca um log como deletado sem remover fisicamente do banco.
     * Necessário para auditoria e conformidade.
     */
    @Modifying
    @Transactional
    @Query("UPDATE WebLog w SET w.deletedAt = :now WHERE w.id = :id AND w.deletedAt IS NULL")
    void softDeleteById(Long id, LocalDateTime now);

    /**
     * Soft delete em lote: marca múltiplos logs como deletados.
     */
    @Modifying
    @Transactional
    @Query("UPDATE WebLog w SET w.deletedAt = :now WHERE w.id IN :ids AND w.deletedAt IS NULL")
    void softDeleteByIdIn(List<Long> ids, LocalDateTime now);

    @Query("SELECT w FROM WebLog w WHERE (:method IS NULL OR w.method = :method) AND (:statusCode IS NULL OR w.statusCode = :statusCode)")
    Page<WebLog> findAllByFilters(@Param("method") String method, @Param("statusCode") Integer statusCode, Pageable pageable);
}

