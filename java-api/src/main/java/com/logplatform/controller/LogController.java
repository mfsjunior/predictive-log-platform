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

/**
 * Controller responsável pela Ingestão de Logs.
 * 
 * Teoria para aula:
 * - Este endpoint aceita arquivos CSV via 'multipart/form-data', que é o padrão para upload de arquivos.
 * - Ingestão de Dados: É o processo de obter dados de fontes externas e trazê-los para o sistema.
 * - Aqui, transformamos um arquivo estático (CSV) em eventos vivos no banco de dados e no Kafka.
 */
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
        // 1. Delega o processamento pesado do CSV para o Service (Padrão de Camadas)
        // Exceções são tratadas automaticamente por GlobalExceptionHandler
        int[] result = logIngestionService.uploadCsv(file);
        
        // 2. Monta a resposta de sucesso com o resumo do processamento
        return ResponseEntity.ok(LogUploadResponse.builder()
                .status("success")
                .recordsProcessed(result[0])
                .recordsFailed(result[1])
                .message(String.format("Processado com sucesso: %d registros (%d falhas)", result[0], result[1]))
                .build());
    }
}
