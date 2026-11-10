SET @has_density = (
    SELECT COUNT(*)
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = 'traffic_records'
      AND column_name = 'density'
);

SET @ddl = IF(
    @has_density = 0,
    'ALTER TABLE traffic_records ADD COLUMN density DECIMAL(10,3) NULL',
    'SELECT ''traffic_records.density already exists'' AS migration_status'
);

PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;