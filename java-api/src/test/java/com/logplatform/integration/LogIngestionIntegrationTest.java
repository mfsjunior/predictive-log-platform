package com.logplatform.integration;

import com.logplatform.config.TestContainersConfig;
import com.logplatform.fixture.CsvFixture;
import com.logplatform.repository.WebLogRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;

import static org.assertj.core.api.Assertions.assertThat;

import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

/**
 * Teste de integração do fluxo de Ingestão de Logs (Módulo M-10).
 *
 * Teoria para aula:
 * - Valida o caminho completo: requisição HTTP de upload -> LogController -> LogIngestionService
 *   -> parsing do CSV -> persistência no PostgreSQL REAL (contêiner).
 * - Diferente do teste de controller (que mocka o service), aqui nada é simulado nesse fluxo:
 *   se o CSV é gravado de verdade no banco, o teste passa.
 */
class LogIngestionIntegrationTest extends TestContainersConfig {

    @Autowired
    private WebLogRepository webLogRepository;

    @BeforeEach
    void cleanDatabase() {
        // Garante isolamento: cada teste começa com a tabela vazia
        webLogRepository.deleteAll();
    }

    @Test
    void uploadCsv_validFile_persistsRecordsInDatabase() {
        // Cenário principal: 10 linhas válidas devem virar 10 registros no banco
        String csv = CsvFixture.validCsv(10);
        ResponseEntity<String> response = uploadCsv("web_logs.csv", "text/csv", csv.getBytes());

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(webLogRepository.count()).isEqualTo(10); // critério M-10: count() > 0 após upload
    }

    @Test
    void uploadCsv_validFile_returnsSuccessBody() {
        // Verifica o corpo da resposta de sucesso (status e contagem de processados)
        String csv = CsvFixture.SINGLE_ROW_CSV;
        ResponseEntity<String> response = uploadCsv("logs.csv", "text/csv", csv.getBytes());

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).contains("success");
        assertThat(response.getBody()).contains("recordsProcessed");
    }

    @Test
    void uploadCsv_nonCsvFile_returns400() {
        // Critério M-10: arquivo que não é CSV deve ser rejeitado com 400, sem gravar nada
        ResponseEntity<String> response = uploadCsv(
                "data.txt", "text/plain", CsvFixture.NON_CSV_CONTENT.getBytes());

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(webLogRepository.count()).isZero();
    }

    @Test
    void uploadCsv_emptyFile_returns400() {
        // Arquivo vazio também é entrada inválida -> 400
        ResponseEntity<String> response = uploadCsv("empty.csv", "text/csv", new byte[0]);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(webLogRepository.count()).isZero();
    }

    @Test
    void uploadCsv_csvWithInvalidRows_persistsOnlyValidRows() {
        // Resiliência: das 3 linhas, 1 é quebrada -> só as 2 válidas são salvas
        ResponseEntity<String> response = uploadCsv(
                "mixed.csv", "text/csv", CsvFixture.CSV_WITH_INVALID_ROWS.getBytes());

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(webLogRepository.count()).isEqualTo(2);
    }

    @Test
    void uploadCsv_largeBatch_persistsAllRecords() {
        // Exercita a estratégia de batch do service (salva de 500 em 500):
        // 550 linhas forçam dois lotes (500 + 50)
        String csv = CsvFixture.validCsv(550);
        ResponseEntity<String> response = uploadCsv("big_logs.csv", "text/csv", csv.getBytes());

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(webLogRepository.count()).isEqualTo(550);
    }

    /**
     * Método auxiliar: monta uma requisição multipart/form-data com o arquivo enviado.
     *
     * Teoria para aula: upload de arquivo no HTTP usa o formato 'multipart'. Aqui simulamos
     * exatamente o que o navegador/Postman faria — um campo "file" com o conteúdo binário.
     */
    private ResponseEntity<String> uploadCsv(String filename, String contentType, byte[] content) {
        // ByteArrayResource com getFilename() sobrescrito: o nome do arquivo importa,
        // pois o service valida se a extensão termina em .csv
        ByteArrayResource file = new ByteArrayResource(content) {
            @Override
            public String getFilename() {
                return filename;
            }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", file);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        return restTemplate.postForEntity(
                baseUrl + "/logs/upload",
                new HttpEntity<>(body, headers),
                String.class);
    }
}
