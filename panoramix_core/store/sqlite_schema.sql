-- Tab1 habit lifecycle state machine — SQLite schema
-- 双表：scene_cards (状态机) + habits (批次产物)

CREATE TABLE IF NOT EXISTS scene_cards (
    card_id TEXT PRIMARY KEY,
    username TEXT NOT NULL,

    status TEXT NOT NULL CHECK (status IN (
        'pending', 'accepted', 'recommendation', 'retired'
    )),

    structural_key TEXT NOT NULL,
    display_name TEXT NOT NULL,

    frozen_content_snapshot_json TEXT,
    content_snapshot_json TEXT,
    supersede_candidate_for_json TEXT,

    first_seen_batch_id INTEGER NOT NULL,
    last_reinforced_batch_id INTEGER NOT NULL,

    created_at TEXT NOT NULL,
    accepted_at TEXT,
    retired_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_scene_cards_status_user
    ON scene_cards(username, status);

CREATE INDEX IF NOT EXISTS idx_scene_cards_structural_key
    ON scene_cards(username, status, structural_key);

CREATE INDEX IF NOT EXISTS idx_scene_cards_last_reinforced
    ON scene_cards(last_reinforced_batch_id);


CREATE TABLE IF NOT EXISTS habits (
    habit_id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    batch_id INTEGER NOT NULL,

    text TEXT NOT NULL,

    signal_category TEXT NOT NULL,
    signal_name TEXT NOT NULL,
    structural_key TEXT NOT NULL,

    context_time_bucket TEXT NOT NULL,
    context_vehicle_state TEXT NOT NULL,
    context_geofence TEXT,
    context_weekday INTEGER,
    context_poi_type TEXT,
    context_wiper_state TEXT,
    context_temp_bucket TEXT,
    context_window_state TEXT,
    context_door_lock TEXT,
    context_approach_unlock TEXT,

    raw_value_stats_json TEXT NOT NULL,
    member_fact_ids_json TEXT NOT NULL,

    scene_card_id TEXT,

    created_at TEXT NOT NULL,

    FOREIGN KEY (scene_card_id) REFERENCES scene_cards(card_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_habits_batch_id ON habits(batch_id);
CREATE INDEX IF NOT EXISTS idx_habits_username_batch ON habits(username, batch_id);
CREATE INDEX IF NOT EXISTS idx_habits_structural_key
    ON habits(username, batch_id, structural_key);
