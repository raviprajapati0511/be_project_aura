-- AURA TimescaleDB schema (data contract).
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS vehicle_telemetry (
    time          TIMESTAMPTZ     NOT NULL,
    vehicle_id    VARCHAR(64),
    pack_voltage  DOUBLE PRECISION,
    pack_current  DOUBLE PRECISION,
    soc_reported  DOUBLE PRECISION,
    cell_temp_max DOUBLE PRECISION,
    speed_kmh     DOUBLE PRECISION,
    odometer_km   DOUBLE PRECISION,
    latitude      DOUBLE PRECISION,
    longitude     DOUBLE PRECISION
);

SELECT create_hypertable('vehicle_telemetry', 'time', if_not_exists => TRUE);

-- Speed up per-vehicle queries on recent windows.
CREATE INDEX IF NOT EXISTS idx_telemetry_vehicle_time
    ON vehicle_telemetry (vehicle_id, time DESC);

-- Retain raw data ~30 days to keep the PoC tiny.
SELECT add_retention_policy('vehicle_telemetry', INTERVAL '30 days',
       if_not_exists => TRUE);
