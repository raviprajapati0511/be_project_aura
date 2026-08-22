-- AURA MySQL schema (data contract).

CREATE TABLE IF NOT EXISTS vehicle_telemetry (
    time          TIMESTAMP       NOT NULL,
    vehicle_id    VARCHAR(64),
    pack_voltage  DOUBLE,
    pack_current  DOUBLE,
    soc_reported  DOUBLE,
    cell_temp_max DOUBLE,
    speed_kmh     DOUBLE,
    odometer_km   DOUBLE,
    latitude      DOUBLE,
    longitude     DOUBLE,
    INDEX idx_telemetry_vehicle_time (vehicle_id, time DESC)
);
