package com.logplatform.domain.port;

import com.logplatform.domain.model.PredictionResult;

/**
 * Port (interface) for persisting prediction audit records.
 */
public interface PredictionPort {

    void save(String type, String inputJson, String resultJson);
}
