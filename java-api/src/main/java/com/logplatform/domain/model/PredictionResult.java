package com.logplatform.domain.model;

/**
 * Pure domain model for ML prediction results.
 * No framework dependencies.
 */
public class PredictionResult {

    private String predictionType;
    private double value;
    private String riskLevel;
    private String modelUsed;
    private double inferenceTimeMs;
    private double lowerBound;
    private double upperBound;
    private double confidenceLevel;

    public PredictionResult() {
    }

    public static PredictionResult errorPrediction(double probability, String riskLevel, String modelUsed) {
        PredictionResult r = new PredictionResult();
        r.predictionType = "error";
        r.value = probability;
        r.riskLevel = riskLevel;
        r.modelUsed = modelUsed;
        return r;
    }

    public static PredictionResult responseTimePrediction(double predictedMs,
            double lower, double upper,
            double confidence) {
        PredictionResult r = new PredictionResult();
        r.predictionType = "response_time";
        r.value = predictedMs;
        r.lowerBound = lower;
        r.upperBound = upper;
        r.confidenceLevel = confidence;
        return r;
    }

    // --- Getters & Setters ---

    public String getPredictionType() {
        return predictionType;
    }

    public void setPredictionType(String predictionType) {
        this.predictionType = predictionType;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public String getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(String riskLevel) {
        this.riskLevel = riskLevel;
    }

    public String getModelUsed() {
        return modelUsed;
    }

    public void setModelUsed(String modelUsed) {
        this.modelUsed = modelUsed;
    }

    public double getInferenceTimeMs() {
        return inferenceTimeMs;
    }

    public void setInferenceTimeMs(double inferenceTimeMs) {
        this.inferenceTimeMs = inferenceTimeMs;
    }

    public double getLowerBound() {
        return lowerBound;
    }

    public void setLowerBound(double lowerBound) {
        this.lowerBound = lowerBound;
    }

    public double getUpperBound() {
        return upperBound;
    }

    public void setUpperBound(double upperBound) {
        this.upperBound = upperBound;
    }

    public double getConfidenceLevel() {
        return confidenceLevel;
    }

    public void setConfidenceLevel(double confidenceLevel) {
        this.confidenceLevel = confidenceLevel;
    }
}
