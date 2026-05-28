package com.logplatform.repository;

import com.logplatform.entity.Prediction;
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
 * Repositório JPA para a entidade Prediction.
 * 
 * Teoria para aula:
 * - JpaRepository: Interface que o Spring "mágicamente" implementa para nós, 
 *   fornecendo Save, Delete, Find, etc., sem precisarmos escrever SQL manual.
 */
@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {
    // Derived Query: O Spring cria o SQL "SELECT count(*) FROM predictions WHERE prediction_type = ?" 
    // baseado apenas no nome deste método.
    long countByPredictionType(String predictionType);

    /**
     * Soft delete: marca uma predição como deletada sem remover fisicamente do banco.
     * Necessário para manter histórico de predições para auditoria.
     */
    @Modifying
    @Transactional
    @Query("UPDATE Prediction p SET p.deletedAt = :now WHERE p.id = :id AND p.deletedAt IS NULL")
    void softDeleteById(Long id, LocalDateTime now);

    /**
     * Soft delete em lote: marca múltiplas predições como deletadas.
     */
    @Modifying
    @Transactional
    @Query("UPDATE Prediction p SET p.deletedAt = :now WHERE p.id IN :ids AND p.deletedAt IS NULL")
    void softDeleteByIdIn(List<Long> ids, LocalDateTime now);

    @Query("SELECT p FROM Prediction p WHERE (:predictionType IS NULL OR p.predictionType = :predictionType)")
    Page<Prediction> findAllByFilters(@Param("predictionType") String predictionType, Pageable pageable);
}
