package com.logplatform.service;

import com.logplatform.entity.WebLog;
import com.logplatform.repository.WebLogRepository;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStreamReader;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class LogIngestionService {

    private final WebLogRepository webLogRepository;

    private static final DateTimeFormatter[] FORMATTERS = {
            DateTimeFormatter.ISO_LOCAL_DATE_TIME,
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss"),
            DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm:ss"),
    };

    @Transactional
    public int[] uploadCsv(MultipartFile file) throws IOException {
        if (file.isEmpty()) {
            throw new IllegalArgumentException("Uploaded file is empty");
        }

        String filename = file.getOriginalFilename();
        if (filename == null || !filename.toLowerCase().endsWith(".csv")) {
            throw new IllegalArgumentException("File must be a CSV file");
        }

        int processed = 0;
        int failed = 0;
        List<WebLog> batch = new ArrayList<>();

        try (CSVReader reader = new CSVReader(new InputStreamReader(file.getInputStream()))) {
            List<String[]> rows = reader.readAll();

            if (rows.isEmpty()) {
                throw new IllegalArgumentException("CSV file has no data");
            }

            // Skip header row
            String[] header = rows.get(0);
            int[] columnMap = resolveColumnMap(header);

            for (int i = 1; i < rows.size(); i++) {
                try {
                    String[] row = rows.get(i);
                    WebLog webLog = parseRow(row, columnMap);
                    batch.add(webLog);
                    processed++;

                    if (batch.size() >= 500) {
                        webLogRepository.saveAll(batch);
                        batch.clear();
                    }
                } catch (Exception e) {
                    failed++;
                    log.warn("Failed to parse row {}: {}", i, e.getMessage());
                }
            }

            if (!batch.isEmpty()) {
                webLogRepository.saveAll(batch);
            }

        } catch (CsvException e) {
            throw new IOException("Failed to parse CSV: " + e.getMessage(), e);
        }

        log.info("CSV upload complete: {} processed, {} failed", processed, failed);
        return new int[]{processed, failed};
    }

    private int[] resolveColumnMap(String[] header) {
        int[] map = new int[]{-1, -1, -1, -1, -1, -1, -1, -1};
        // 0=timestamp, 1=method, 2=path, 3=status_code, 4=response_time_ms,
        // 5=user_agent, 6=ip_address, 7=bytes_sent

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
                    "CSV must contain at least: timestamp, method, status_code, response_time_ms columns"
            );
        }

        return map;
    }

    private WebLog parseRow(String[] row, int[] columnMap) {
        return WebLog.builder()
                .timestamp(parseTimestamp(row[columnMap[0]].trim()))
                .method(row[columnMap[1]].trim().toUpperCase())
                .path(columnMap[2] >= 0 ? row[columnMap[2]].trim() : "/unknown")
                .statusCode(Integer.parseInt(row[columnMap[3]].trim()))
                .responseTimeMs(Double.parseDouble(row[columnMap[4]].trim()))
                .userAgent(columnMap[5] >= 0 && columnMap[5] < row.length ? row[columnMap[5]].trim() : null)
                .ipAddress(columnMap[6] >= 0 && columnMap[6] < row.length ? row[columnMap[6]].trim() : null)
                .bytesSent(columnMap[7] >= 0 && columnMap[7] < row.length ?
                        Integer.parseInt(row[columnMap[7]].trim()) : 0)
                .build();
    }

    private LocalDateTime parseTimestamp(String value) {
        for (DateTimeFormatter formatter : FORMATTERS) {
            try {
                return LocalDateTime.parse(value, formatter);
            } catch (DateTimeParseException ignore) {}
        }
        throw new DateTimeParseException("Cannot parse timestamp", value, 0);
    }
}
