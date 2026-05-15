-- Long-term storage for operator recommendations aged out of the active table.
-- Run periodically via: spuncast-ml archive-recommendations --days 60

CREATE TABLE IF NOT EXISTS heat_recommendations_archive (
    archive_id           BIGSERIAL    PRIMARY KEY,
    id                   BIGINT,
    heat_number          TEXT         NOT NULL,
    decision_code        TEXT,
    recommendation_text  TEXT,
    primary_driver       TEXT,
    scrap_probability    FLOAT,
    feature_set          TEXT,
    created_at           TIMESTAMPTZ,
    resolved_at          TIMESTAMPTZ,
    resolved_by          TEXT,
    archived_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hra_heat
    ON heat_recommendations_archive (heat_number);

CREATE INDEX IF NOT EXISTS idx_hra_archived_at
    ON heat_recommendations_archive (archived_at DESC);
