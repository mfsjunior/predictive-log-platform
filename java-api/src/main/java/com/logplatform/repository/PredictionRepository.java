package com.logplatform.repository;

import com.logplatform.entity.Prediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {
    long countByPredictionType(String predictionType);
}
