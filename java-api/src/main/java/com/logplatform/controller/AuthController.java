package com.logplatform.controller;

import com.logplatform.security.JwtTokenProvider;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * Authentication controller — issues JWT tokens.
 */
@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
@Tag(name = "Authentication", description = "Login and JWT token management")
public class AuthController {

    private final JwtTokenProvider jwtTokenProvider;

    @Value("${jwt.admin.username:admin}")
    private String adminUsername;

    @Value("${jwt.admin.password:admin123}")
    private String adminPassword;

    @PostMapping("/login")
    @Operation(summary = "Login", description = "Authenticate with username/password and receive a JWT token")
    public ResponseEntity<?> login(@RequestBody Map<String, String> credentials) {
        String username = credentials.getOrDefault("username", "");
        String password = credentials.getOrDefault("password", "");

        if (adminUsername.equals(username) && adminPassword.equals(password)) {
            String token = jwtTokenProvider.generateToken(username);
            return ResponseEntity.ok(Map.of(
                    "token", token,
                    "type", "Bearer",
                    "username", username
            ));
        }

        return ResponseEntity.status(401).body(Map.of(
                "error", "Invalid credentials"
        ));
    }
}
