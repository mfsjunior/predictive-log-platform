package com.logplatform.fixture;

/**
 * Fábrica de conteúdos CSV reutilizáveis para os testes de integração.
 *
 * Teoria para aula:
 * - Fixture: é um "molde" de dados de teste pré-definido. Em vez de cada teste montar seu
 *   próprio CSV na mão (o que gera repetição e erros), centralizamos aqui os exemplos.
 * - Observação importante: o arquivo real 'data/web_logs.csv' NÃO é versionado no repositório
 *   (a pasta só tem um .gitkeep, pois o CSV é gerado em runtime). Por isso geramos CSVs
 *   sintéticos aqui, com o mesmo formato de colunas que o parser real (LogIngestor) espera.
 */
public final class CsvFixture {

    // Construtor privado: esta é uma classe utilitária, não deve ser instanciada
    private CsvFixture() {}

    // Cabeçalho com as 8 colunas reconhecidas pelo LogIngestor.resolveColumnMap()
    public static final String HEADER =
            "timestamp,method,path,status_code,response_time_ms,user_agent,ip_address,bytes_sent";

    /**
     * Gera um CSV válido com a quantidade de linhas pedida.
     * Varia método, status e tempo de resposta para simular tráfego realista.
     */
    public static String validCsv(int rows) {
        StringBuilder sb = new StringBuilder(HEADER).append('\n');
        for (int i = 0; i < rows; i++) {
            sb.append("2025-01-15T10:").append(String.format("%02d", i % 60)).append(":00,")
              .append(i % 2 == 0 ? "GET" : "POST").append(',')
              .append("/api/resource/").append(i).append(',')
              .append(i % 5 == 0 ? 500 : 200).append(',') // a cada 5 linhas, um erro 500
              .append(100 + i * 3).append(".0,")
              .append("Mozilla/5.0,")
              .append("192.168.1.").append(i % 255 + 1).append(',')
              .append(1024)
              .append('\n');
        }
        return sb.toString();
    }

    // CSV mínimo com uma única linha válida — usado para checar a resposta de sucesso
    public static final String SINGLE_ROW_CSV =
            HEADER + "\n" +
            "2025-06-01T08:30:00,GET,/api/health,200,45.0,curl/8.4.0,10.0.0.1,512\n";

    // Conteúdo que NÃO é CSV — usado para validar a rejeição com HTTP 400
    public static final String NON_CSV_CONTENT = "this is not a csv file at all";

    // Arquivo vazio — outro caso que deve ser rejeitado com HTTP 400
    public static final String EMPTY_CSV = "";

    // CSV com uma linha quebrada no meio: testa que as válidas são salvas e a inválida é
    // contabilizada como falha, sem abortar o processamento (comportamento do LogIngestionService)
    public static final String CSV_WITH_INVALID_ROWS =
            HEADER + "\n" +
            "2025-06-01T08:30:00,GET,/api/health,200,45.0,curl/8.4.0,10.0.0.1,512\n" +
            "bad_row_missing_fields\n" +
            "2025-06-01T08:31:00,POST,/api/data,201,120.0,Mozilla,10.0.0.2,2048\n";
}
