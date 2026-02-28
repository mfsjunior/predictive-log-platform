package com.logplatform.dto;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class LogUploadResponse {
    private String status;
    private int recordsProcessed;
    private int recordsFailed;
    private String message;
}
