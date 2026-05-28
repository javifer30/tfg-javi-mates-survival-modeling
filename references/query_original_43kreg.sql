-- ===========================================================
-- STEP 1: BASE ICU + ADMISSIONS + PATIENTS (adult ICU stays)
-- ===========================================================

WITH icu_base AS (
  SELECT
    icu.subject_id,
    icu.hadm_id,
    icu.stay_id,
    icu.intime,
    icu.outtime,
    adm.admittime,
    adm.dischtime,
    adm.deathtime,
    adm.hospital_expire_flag,

    -- Raw demographics
    pat.gender AS sex,
    pat.anchor_age AS age,

    -- ICU unit type (first_careunit)
    icu.first_careunit AS unit_type,

    -- Admission location
    adm.admission_location AS admission_location

  FROM `physionet-data.mimiciv_3_1_icu.icustays` AS icu
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` AS adm
      USING (subject_id, hadm_id)
  JOIN `physionet-data.mimiciv_3_1_hosp.patients` AS pat
      USING (subject_id)

  WHERE pat.anchor_age >= 18
),


-- ===========================================================
-- STEP 2: HEIGHT & WEIGHT (first measurements in ICU window)
-- ===========================================================

height_weight AS (
  SELECT
    stay_id,
    -- FIRST recorded height during ICU stay
    ANY_VALUE(CASE WHEN item = 'Height' THEN val END) AS height,
    -- FIRST recorded weight during ICU stay
    ANY_VALUE(CASE WHEN item = 'Weight' THEN val END) AS weight
  FROM (
    SELECT
      ce.stay_id,
      CASE 
         WHEN ce.itemid IN (226707, 226730) THEN 'Height'
         WHEN ce.itemid IN (226512, 226531) THEN 'Weight'
      END AS item,
      ce.valuenum AS val,
      ce.charttime
    FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
    WHERE ce.itemid IN (
        226707, 226730,   -- Height
        226512, 226531    -- Weight
    )
  )
  GROUP BY stay_id
),


-- ===========================================================
-- STEP 3: ETHNICITY (from admissions table)
-- ===========================================================

ethnicity AS (
  SELECT 
    hadm_id,
    race
  FROM `physionet-data.mimiciv_3_1_hosp.admissions`
),


-- ===========================================================
-- STEP 4: GCS MOTOR & VERBAL (first 24h after ICU admission)
-- ===========================================================

gcs AS (
  SELECT
    stay_id,
    ANY_VALUE(CASE WHEN item = 'GCS_MOTOR'  THEN val END) AS gcs_motor,
    ANY_VALUE(CASE WHEN item = 'GCS_VERBAL' THEN val END) AS gcs_verbal
  FROM (
    SELECT
      ce.stay_id,
      CASE
        WHEN ce.itemid IN (454, 223901) THEN 'GCS_MOTOR'
        WHEN ce.itemid IN (723, 223900) THEN 'GCS_VERBAL'
      END AS item,
      ce.valuenum AS val,
      ce.charttime
    FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
    JOIN icu_base ib USING(stay_id)
    WHERE ce.itemid IN (
        454, 223901,     -- GCS motor
        723, 223900      -- GCS verbal
    )
    AND ce.charttime BETWEEN ib.intime AND ib.intime + INTERVAL 24 HOUR
  )
  GROUP BY stay_id
),


-- ===========================================================
-- STEP 5: MERGE ALL STATIC VARIABLES
-- ===========================================================

static_all AS (
  SELECT
    ib.*,
    hw.height,
    hw.weight,
    et.race,
    g.gcs_motor,
    g.gcs_verbal,

    -- Hour of admission (0-23)
    EXTRACT(HOUR FROM ib.intime) AS hour_of_admission,

    -- Time since hospital admission (days)
    TIMESTAMP_DIFF(ib.intime, ib.admittime, HOUR) / 24.0 AS time_since_admission

  FROM icu_base ib
  LEFT JOIN height_weight hw USING(stay_id)
  LEFT JOIN ethnicity et USING(hadm_id)
  LEFT JOIN gcs g USING(stay_id)
),


-- ===========================================================
-- STEP 6: SURVIVAL TIME + EVENT (10-day cap)
-- ===========================================================

survival AS (
  SELECT
    *,
    TIMESTAMP_DIFF(dischtime, intime, HOUR) / 24.0 AS time_to_discharge,
    CASE 
      WHEN deathtime IS NOT NULL THEN TIMESTAMP_DIFF(deathtime, intime, HOUR) / 24.0
      ELSE NULL
    END AS time_to_death
  FROM static_all
),

surv_final AS (
  SELECT
    *,
    LEAST(
      10.0,
      time_to_discharge,
      IF(time_to_death IS NULL, 1e9, time_to_death)
    ) AS time,

    CASE
      WHEN time_to_death IS NOT NULL AND time_to_death <= 10.0 THEN 1
      ELSE 0
    END AS event
  FROM survival
),


-- ===========================================================
-- STEP 7: FINAL FILTERS (DySurv-style)
-- ===========================================================

filtered AS (
  SELECT *
  FROM surv_final
  WHERE 
    -- remove corrupted stays
    intime < outtime

    -- height & weight must be present
    AND height IS NOT NULL
    AND weight IS NOT NULL

    -- essential GCS components
    AND gcs_motor IS NOT NULL
    AND gcs_verbal IS NOT NULL

    -- categorical fields
    AND race IS NOT NULL
    AND admission_location IS NOT NULL
    AND unit_type IS NOT NULL

    -- time since admission must be computable (rare)
    AND time_since_admission IS NOT NULL
)

-- ===========================================================
-- OUTPUT
-- ===========================================================

SELECT * FROM filtered;
