-- Persistent scoring table written by the live scoring daemon.
-- Deployed in the Spuncast-Operations database alongside pour_logs.

CREATE TABLE IF NOT EXISTS ml_heat_scores (
    id                BIGSERIAL    PRIMARY KEY,
    heat_number       TEXT         NOT NULL UNIQUE,
    scrap_probability FLOAT        NOT NULL,
    predicted_flag    INTEGER      NOT NULL,
    recommended_action TEXT,
    feature_set       TEXT         NOT NULL DEFAULT 'early_remelt_decision',
    model_version     TEXT,
    scored_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    operator_action   TEXT,        -- 'remelt' | 'proceed' | NULL
    actual_scrap_flag INTEGER      -- filled in after outcome known
);

CREATE INDEX IF NOT EXISTS idx_mhs_heat
    ON ml_heat_scores (heat_number);

CREATE INDEX IF NOT EXISTS idx_mhs_scored_at
    ON ml_heat_scores (scored_at DESC);
