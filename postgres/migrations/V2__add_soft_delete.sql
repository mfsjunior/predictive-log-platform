-- Migration: Add deleted_at columns for soft delete support
ALTER TABLE web_logs ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP DEFAULT NULL;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP DEFAULT NULL;
