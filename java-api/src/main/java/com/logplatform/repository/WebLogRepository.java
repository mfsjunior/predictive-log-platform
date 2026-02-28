package com.logplatform.repository;

import com.logplatform.entity.WebLog;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

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
}
