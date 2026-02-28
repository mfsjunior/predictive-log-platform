package com.logplatform.controller;

import com.logplatform.service.LogIngestionService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.bean.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import com.logplatform.config.SecurityConfig;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(LogController.class)
@Import(SecurityConfig.class)
class LogControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private LogIngestionService logIngestionService;

    @Test
    void uploadCsv_shouldReturnSuccess() throws Exception {
        String csvContent = "timestamp,method,path,status_code,response_time_ms,user_agent,ip_address,bytes_sent\n"
                + "2025-01-15T10:30:00,GET,/api/users,200,150.5,Mozilla/5.0,192.168.1.1,1024\n"
                + "2025-01-15T10:31:00,POST,/api/orders,201,250.3,curl/8.4.0,10.0.0.1,2048\n";

        MockMultipartFile file = new MockMultipartFile(
                "file", "web_logs.csv", "text/csv", csvContent.getBytes());

        when(logIngestionService.uploadCsv(any())).thenReturn(new int[]{2, 0});

        mockMvc.perform(multipart("/logs/upload").file(file))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("success"))
                .andExpect(jsonPath("$.recordsProcessed").value(2))
                .andExpect(jsonPath("$.recordsFailed").value(0));
    }

    @Test
    void uploadCsv_shouldReturnBadRequest_whenFileEmpty() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file", "empty.csv", "text/csv", new byte[0]);

        when(logIngestionService.uploadCsv(any()))
                .thenThrow(new IllegalArgumentException("Uploaded file is empty"));

        mockMvc.perform(multipart("/logs/upload").file(file))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value("error"));
    }

    @Test
    void uploadCsv_shouldReturnBadRequest_whenNotCsv() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file", "data.txt", "text/plain", "not csv".getBytes());

        when(logIngestionService.uploadCsv(any()))
                .thenThrow(new IllegalArgumentException("File must be a CSV file"));

        mockMvc.perform(multipart("/logs/upload").file(file))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value("error"));
    }

    @Test
    void uploadCsv_shouldHandlePartialFailure() throws Exception {
        String csvContent = "timestamp,method,path,status_code,response_time_ms\n"
                + "2025-01-15T10:30:00,GET,/api,200,150.5\n"
                + "invalid_row\n";

        MockMultipartFile file = new MockMultipartFile(
                "file", "logs.csv", "text/csv", csvContent.getBytes());

        when(logIngestionService.uploadCsv(any())).thenReturn(new int[]{1, 1});

        mockMvc.perform(multipart("/logs/upload").file(file))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.recordsProcessed").value(1))
                .andExpect(jsonPath("$.recordsFailed").value(1));
    }
}
