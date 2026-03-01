"""
WebSocket endpoint for real-time anomaly alerts.
Clients connect to ws://host:8000/ws/alerts and receive push notifications
whenever an anomaly is detected in ingested logs.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# Connected WebSocket clients
_active_connections: Set[WebSocket] = set()


async def broadcast_alert(alert: dict):
    """Send alert to all connected WebSocket clients."""
    if not _active_connections:
        return

    message = json.dumps(alert, default=str)
    disconnected = set()

    for connection in _active_connections:
        try:
            await connection.send_text(message)
        except Exception:
            disconnected.add(connection)

    _active_connections.difference_update(disconnected)


async def check_and_alert(log_data: dict):
    """
    Check a log entry for anomalies and broadcast alert if detected.
    Called after each log ingestion.
    """
    from app.routers.train import anomaly_detector
    from app.feature_engineering import prepare_single_prediction

    if anomaly_detector is None:
        return

    try:
        response_time = float(log_data.get("response_time_ms", 0))
        method = log_data.get("method", "GET")
        hour = int(log_data.get("hour", datetime.now().hour))

        features_df = prepare_single_prediction(method, hour, response_time)
        result = anomaly_detector.detect(
            response_time_ms=response_time,
            hour=hour,
            method=method,
            features_df=features_df,
        )

        if result.get("is_anomaly"):
            alert = {
                "type": "anomaly_detected",
                "timestamp": datetime.now().isoformat(),
                "severity": "HIGH" if result.get("score", 0) > 0.8 else "MEDIUM",
                "log_data": log_data,
                "anomaly_details": result,
            }
            logger.warning(f"Anomaly alert: {log_data.get('path', 'N/A')} — score={result.get('score', 0):.2f}")
            await broadcast_alert(alert)

    except Exception as e:
        logger.error(f"Alert check failed: {e}")


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time anomaly alerts.
    Connect with: wscat -c ws://localhost:8000/ws/alerts
    """
    await websocket.accept()
    _active_connections.add(websocket)
    logger.info(f"WebSocket client connected. Total: {len(_active_connections)}")

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to PLIP real-time alerts",
            "timestamp": datetime.now().isoformat(),
        })

        # Keep connection alive — listen for client messages (ping/pong)
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})

    except WebSocketDisconnect:
        _active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(_active_connections)}")
    except Exception as e:
        _active_connections.discard(websocket)
        logger.error(f"WebSocket error: {e}")
