package com.logplatform.service;

import com.logplatform.domain.model.WebLogDomain;
import com.logplatform.domain.port.LogRepository;
import com.logplatform.domain.service.LogIngestor;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStreamReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Serviço de Ingestão de Logs da camada de Aplicação.
 * 
 * Desacoplado de entidades de banco e adaptadores de persistência.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class LogIngestionService {

    private final LogRepository logRepository;
    private final LogIngestor logIngestor = new LogIngestor();

    @Transactional
    public int[] uploadCsv(MultipartFile file) throws IOException {
        // 1. Validação inicial: verifica se o arquivo não está vazio
        if (file.isEmpty()) {
            throw new IllegalArgumentException("Uploaded file is empty");
        }

        // 2. Validação de extensão: garante que é um arquivo CSV
        String filename = file.getOriginalFilename();
        if (filename == null || !filename.toLowerCase().endsWith(".csv")) {
            throw new IllegalArgumentException("File must be a CSV file");
        }

        int processed = 0;
        int failed = 0;
        List<WebLogDomain> batch = new ArrayList<>();

        // 3. Abre o leitor de CSV com Try-with-resources para garantir o fechamento do stream
        try (CSVReader reader = new CSVReader(new InputStreamReader(file.getInputStream()))) {
            List<String[]> rows = reader.readAll();

            if (rows.isEmpty()) {
                throw new IllegalArgumentException("CSV file has no data");
            }

            // 4. Resolve o mapeamento dinâmico de colunas baseado no cabeçalho
            String[] header = rows.get(0);
            int[] columnMap = logIngestor.resolveColumnMap(header);

            // 5. Itera sobre as linhas de dados (pulando o cabeçalho no índice 0)
            for (int i = 1; i < rows.size(); i++) {
                try {
                    String[] row = rows.get(i);
                    // Converte a linha de texto para o objeto de domínio usando o Domain Service
                    WebLogDomain webLogDomain = logIngestor.parseRow(row, columnMap);
                    batch.add(webLogDomain);
                    processed++;

                    // 6. Estratégia de Batch (Lote): Salva de 500 em 500 para performance
                    if (batch.size() >= 500) {
                        logRepository.saveAll(batch);
                        batch.clear();
                    }
                } catch (Exception e) {
                    // Se uma linha falhar, incrementamos o erro mas continuamos o processo
                    failed++;
                    log.warn("Failed to parse row {}: {}", i, e.getMessage());
                }
            }

            // 7. Salva o último lote remanescente
            if (!batch.isEmpty()) {
                logRepository.saveAll(batch);
            }

        } catch (CsvException e) {
            throw new IOException("Failed to parse CSV: " + e.getMessage(), e);
        }

        log.info("CSV upload complete: {} processed, {} failed", processed, failed);
        return new int[] { processed, failed };
    }
}
