package com.logplatform.domain.port;

import com.logplatform.domain.model.PredictionResult;

/**
 * Port (interface) para persistência de auditoria de predições.
 */
public interface PredictionPort {

    void save(PredictionResult domain, Object input);
}
