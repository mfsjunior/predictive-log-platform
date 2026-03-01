package com.logplatform.domain.service;

import com.logplatform.domain.model.WebLogDomain;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;

/**
 * Pure domain service for log ingestion.
 * No Spring, no JPA, no framework — only Java SE.
 * Parses CSV rows into domain objects with validation.
 */
public class LogIngestor {

    private static final DateTimeFormatter[] FORMATTERS = {
            DateTimeFormatter.ISO_LOCAL_DATE_TIME,
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss"),
            DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm:ss"),
    };

    /**
     * Parse a CSV row into a domain WebLog.
     *
     * @param row       CSV fields
     * @param columnMap index mapping: 0=timestamp, 1=method, 2=path, 3=status,
     *                  4=responseTime, 5=userAgent, 6=ip, 7=bytes
     */
    public WebLogDomain parseRow(String[] row, int[] columnMap) {
        LocalDateTime timestamp = parseTimestamp(row[columnMap[0]].trim());
        String method = row[columnMap[1]].trim().toUpperCase();
        String path = columnMap[2] >= 0 ? row[columnMap[2]].trim() : "/unknown";
        int statusCode = Integer.parseInt(row[columnMap[3]].trim());
        double responseTimeMs = Double.parseDouble(row[columnMap[4]].trim());
        String userAgent = safeGet(row, columnMap[5]);
        String ipAddress = safeGet(row, columnMap[6]);
        int bytesSent = columnMap[7] >= 0 && columnMap[7] < row.length
                ? Integer.parseInt(row[columnMap[7]].trim())
                : 0;

        return new WebLogDomain(timestamp, method, path, statusCode,
                responseTimeMs, userAgent, ipAddress, bytesSent);
    }

    /**
     * Resolve column header names to index positions.
     *
     * @throws IllegalArgumentException if required columns are missing
     */
    public int[] resolveColumnMap(String[] header) {
        int[] map = new int[] { -1, -1, -1, -1, -1, -1, -1, -1 };

        for (int i = 0; i < header.length; i++) {
            String col = header[i].trim().toLowerCase().replace(" ", "_");
            switch (col) {
                case "timestamp" -> map[0] = i;
                case "method" -> map[1] = i;
                case "path" -> map[2] = i;
                case "status_code" -> map[3] = i;
                case "response_time_ms" -> map[4] = i;
                case "user_agent" -> map[5] = i;
                case "ip_address" -> map[6] = i;
                case "bytes_sent" -> map[7] = i;
            }
        }

        if (map[0] == -1 || map[1] == -1 || map[3] == -1 || map[4] == -1) {
            throw new IllegalArgumentException(
                    "CSV must contain at least: timestamp, method, status_code, response_time_ms");
        }
        return map;
    }

    private LocalDateTime parseTimestamp(String value) {
        for (DateTimeFormatter formatter : FORMATTERS) {
            try {
                return LocalDateTime.parse(value, formatter);
            } catch (DateTimeParseException ignore) {
            }
        }
        throw new DateTimeParseException("Cannot parse timestamp", value, 0);
    }

    private String safeGet(String[] row, int index) {
        return index >= 0 && index < row.length ? row[index].trim() : null;
    }
}
