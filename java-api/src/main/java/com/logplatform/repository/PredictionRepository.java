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
    @Query("SELECT p FROM Prediction p WHERE (:predictionType IS NULL OR p.predictionType = :predictionType)")
    Page<Prediction> findByPredictionType(@Param("predictionType") String predictionType, Pageable pageable);

    long countByPredictionType(String predictionType);

    @Modifying
    @Transactional
    @Query("UPDATE Prediction p SET p.deletedAt = CURRENT_TIMESTAMP WHERE p.id = :id")
    void softDeleteById(@Param("id") Long id);
}
