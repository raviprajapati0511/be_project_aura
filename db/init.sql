-- AURA MySQL schema (data contract).

CREATE TABLE IF NOT EXISTS vehicle_telemetry (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    time          TIMESTAMP(3)    NOT NULL,
    vehicle_id    VARCHAR(64),
    pack_voltage  DOUBLE,
    pack_current  DOUBLE,
    soc_reported  DOUBLE,
    cell_temp_max DOUBLE,
    speed_kmh     DOUBLE,
    odometer_km   DOUBLE,
    latitude      DOUBLE,
    longitude     DOUBLE,
    battery_soh   DOUBLE,
    motor_temp    DOUBLE,
    motor_vibration DOUBLE,
    power_consumption DOUBLE,
    component_health DOUBLE,
    failure_probability DOUBLE,
    rul_score     DOUBLE,
    maintenance_type VARCHAR(32),
    acceleration_mps2 DOUBLE,
    can_anomaly_score DOUBLE NULL,
    INDEX idx_telemetry_vehicle_time (vehicle_id, time DESC)
);
