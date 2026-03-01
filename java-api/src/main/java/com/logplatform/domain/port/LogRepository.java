package com.logplatform.domain.port;

import com.logplatform.domain.model.WebLogDomain;
import java.util.List;

/**
 * Port (interface) for log persistence.
 * Domain defines the contract — infrastructure implements it.
 */
public interface LogRepository {

    void saveAll(List<WebLogDomain> logs);

    List<WebLogDomain> findAll();

    long count();
}
