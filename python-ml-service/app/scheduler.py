"""
Scheduler for periodic drift checking and automatic retraining.
Runs every 6 hours — if drift > 50%, triggers model retraining.
"""
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_scheduler_task = None


async def _check_drift_and_retrain():
    """Periodic task that checks drift and retrains if needed."""
    from app.monitoring.drift import get_reference_data, generate_drift_report
    from app.feature_engineering import engineer_features

    while True:
        try:
            await asyncio.sleep(6 * 3600)  # Wait 6 hours

            reference = get_reference_data()
            if reference is None:
                logger.info("[Scheduler] No reference data yet — skipping drift check.")
                continue

            # Fetch current data
            from app.routers.monitor import _fetch_current_data
            current_df = _fetch_current_data()

            if current_df is None or len(current_df) == 0:
                logger.info("[Scheduler] No current data — skipping drift check.")
                continue

            current_features = engineer_features(current_df)
            result = generate_drift_report(current_data=current_features)

            dataset_drift = result.get("dataset_drift", False)
            drift_share = result.get("drift_share", 0.0)

            logger.info(
                f"[Scheduler] Drift check: drift_detected={dataset_drift}, "
                f"drift_share={drift_share:.2%}"
            )

            if dataset_drift:
                logger.warning("[Scheduler] Drift detected! Triggering automatic retraining...")
                await _trigger_retrain()
            else:
                logger.info("[Scheduler] No significant drift — models are up to date.")

        except Exception as e:
            logger.error(f"[Scheduler] Drift check failed: {e}", exc_info=True)


async def _trigger_retrain():
    """Trigger model retraining."""
    try:
        from app.routers.train import train_models
        result = await train_models()
        logger.info(
            f"[Scheduler] Retraining complete — "
            f"classifier: {result.classifier_results.get('best_model', 'N/A')}, "
            f"regressor: {result.regressor_results.get('best_model', 'N/A')}"
        )
    except Exception as e:
        logger.error(f"[Scheduler] Retraining failed: {e}", exc_info=True)


def start_scheduler():
    """Start the periodic drift-check scheduler."""
    global _scheduler_task
    loop = asyncio.get_event_loop()
    _scheduler_task = loop.create_task(_check_drift_and_retrain())
    logger.info("[Scheduler] Periodic drift monitor started (interval: 6h)")


def stop_scheduler():
    """Stop the scheduler."""
    global _scheduler_task
    if _scheduler_task:
        _scheduler_task.cancel()
        _scheduler_task = None
        logger.info("[Scheduler] Drift monitor stopped.")
