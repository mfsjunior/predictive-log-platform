-- ============================================================
-- Predictive Log Intelligence Platform - Database Schema
-- ============================================================

-- Web Logs Table
CREATE TABLE IF NOT EXISTS web_logs (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    method          VARCHAR(10) NOT NULL,
    path            VARCHAR(500) NOT NULL,
    status_code     INTEGER NOT NULL,
    response_time_ms DOUBLE PRECISION NOT NULL,
    user_agent      VARCHAR(500),
    ip_address      VARCHAR(45),
    bytes_sent      INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions History Table
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    prediction_type VARCHAR(50) NOT NULL,
    input_data      JSONB NOT NULL,
    result          JSONB NOT NULL,
    model_version   VARCHAR(100),
    latency_ms      DOUBLE PRECISION,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Metadata Table
CREATE TABLE IF NOT EXISTS model_metadata (
    id              BIGSERIAL PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,
    model_type      VARCHAR(50) NOT NULL,
    version         VARCHAR(50) NOT NULL,
    metrics         JSONB,
    file_path       VARCHAR(500),
    is_active       BOOLEAN DEFAULT FALSE,
    trained_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training Runs Table
CREATE TABLE IF NOT EXISTS training_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_id          VARCHAR(100),
    status          VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    num_samples     INTEGER,
    best_classifier VARCHAR(100),
    best_regressor  VARCHAR(100),
    classifier_metrics JSONB,
    regressor_metrics  JSONB,
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at    TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_web_logs_timestamp ON web_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_web_logs_status_code ON web_logs(status_code);
CREATE INDEX IF NOT EXISTS idx_web_logs_method ON web_logs(method);
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_model_metadata_active ON model_metadata(is_active);
CREATE INDEX idx_training_status ON training_runs(status);
CREATE INDEX idx_training_started ON training_runs(started_at);

-- =============================================
-- Table: alerts (real-time anomaly alerts)
-- =============================================
CREATE TABLE IF NOT EXISTS alerts (
    id              BIGSERIAL PRIMARY KEY,
    alert_type      VARCHAR(50)     NOT NULL DEFAULT 'anomaly_detected',
    severity        VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    log_data        JSONB,
    anomaly_details JSONB,
    acknowledged    BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_created ON alerts(created_at);
CREATE INDEX idx_alerts_acknowledged ON alerts(acknowledged);
