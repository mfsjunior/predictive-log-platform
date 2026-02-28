package com.logplatform.controller;

import com.logplatform.dto.LogUploadResponse;
import com.logplatform.service.LogIngestionService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/logs")
@RequiredArgsConstructor
@Slf4j
@Tag(name = "Log Ingestion", description = "Upload and manage web logs")
public class LogController {

    private final LogIngestionService logIngestionService;

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "Upload CSV log file", description = "Upload a CSV file containing web log entries for ingestion")
    public ResponseEntity<LogUploadResponse> uploadCsv(@RequestParam("file") MultipartFile file) {
        try {
            int[] result = logIngestionService.uploadCsv(file);
            return ResponseEntity.ok(LogUploadResponse.builder()
                    .status("success")
                    .recordsProcessed(result[0])
                    .recordsFailed(result[1])
                    .message(String.format("Successfully processed %d records (%d failed)", result[0], result[1]))
                    .build());
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(LogUploadResponse.builder()
                    .status("error")
                    .recordsProcessed(0)
                    .recordsFailed(0)
                    .message(e.getMessage())
                    .build());
        } catch (Exception e) {
            log.error("CSV upload failed", e);
            return ResponseEntity.internalServerError().body(LogUploadResponse.builder()
                    .status("error")
                    .recordsProcessed(0)
                    .recordsFailed(0)
                    .message("Upload failed: " + e.getMessage())
                    .build());
        }
    }
}
