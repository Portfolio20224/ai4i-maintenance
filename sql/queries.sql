-- Requêtes d'analyse sur la table `machines` (SQLite ≥ 3.25, fonctions de fenêtrage).
-- Format : « -- name: <nom> » avant chaque requête, « -- setup » pour les vues.

-- setup
DROP VIEW IF EXISTS v_physics;
CREATE VIEW v_physics AS
SELECT *,
       process_temp_k - air_temp_k                        AS delta_t_k,
       torque_nm * rpm * 2 * 3.141592653589793 / 60.0     AS power_w,
       tool_wear_min * torque_nm                          AS wear_torque
FROM machines;

-- name: failure_by_type
SELECT type,
       COUNT(*)                                AS n_cycles,
       SUM(machine_failure)                    AS n_failures,
       ROUND(100.0 * AVG(machine_failure), 2)  AS failure_rate_pct
FROM machines
GROUP BY type
ORDER BY type;

-- name: failure_by_wear_bucket
SELECT (tool_wear_min / 40) * 40              AS wear_from_min,
       COUNT(*)                                AS n_cycles,
       SUM(machine_failure)                    AS n_failures,
       ROUND(100.0 * AVG(machine_failure), 2)  AS failure_rate_pct,
       SUM(osf)                                AS osf,
       SUM(twf)                                AS twf,
       SUM(hdf)                                AS hdf,
       SUM(pwf)                                AS pwf
FROM machines
GROUP BY wear_from_min
ORDER BY wear_from_min;

-- name: modes_by_type
SELECT type,
       SUM(twf) AS twf, SUM(hdf) AS hdf, SUM(pwf) AS pwf,
       SUM(osf) AS osf, SUM(rnf) AS rnf
FROM machines
GROUP BY type
ORDER BY type;

-- name: label_consistency
-- Cohérence entre « Machine failure » et les 5 modes.
SELECT CASE
         WHEN machine_failure = 1 AND (twf + hdf + pwf + osf + rnf) = 0 THEN 'panne sans mode renseigné'
         WHEN machine_failure = 0 AND (twf + hdf + pwf + osf + rnf) > 0 THEN 'mode renseigné sans panne'
         WHEN machine_failure = 1 THEN 'cohérent (panne)'
         ELSE 'cohérent (pas de panne)'
       END AS statut,
       COUNT(*) AS n_lignes
FROM machines
GROUP BY statut
ORDER BY n_lignes DESC;

-- name: failure_rate_per_block
-- Taux de panne par bloc de 1 000 cycles + cumul (fonction de fenêtrage).
WITH blocks AS (
  SELECT (udi - 1) / 1000 AS block,
         COUNT(*) AS n_cycles,
         SUM(machine_failure) AS n_failures
  FROM machines
  GROUP BY block
)
SELECT block,
       n_cycles,
       n_failures,
       ROUND(100.0 * n_failures / n_cycles, 2)             AS failure_rate_pct,
       SUM(n_failures) OVER (ORDER BY block)                AS cumulative_failures
FROM blocks
ORDER BY block;

-- name: tool_lives
-- Un outil est remplacé quand l'usure diminue d'une ligne à la suivante (LAG).
-- On récupère l'usure de la dernière ligne de chaque outil et si elle porte un TWF.
WITH seq AS (
  SELECT udi, tool_wear_min, twf,
         LAG(tool_wear_min) OVER (ORDER BY udi) AS prev_wear,
         LAG(twf)           OVER (ORDER BY udi) AS prev_twf
  FROM machines
)
SELECT udi - 1                    AS last_row_udi,
       prev_wear                  AS end_of_life_wear_min,
       prev_twf                   AS failed_by_tool_wear
FROM seq
WHERE tool_wear_min < prev_wear
ORDER BY udi;

-- name: tool_life_summary
WITH seq AS (
  SELECT tool_wear_min, twf,
         LAG(tool_wear_min) OVER (ORDER BY udi) AS prev_wear,
         LAG(twf)           OVER (ORDER BY udi) AS prev_twf
  FROM machines
), lives AS (
  SELECT prev_wear AS eol_wear, prev_twf AS failed FROM seq WHERE tool_wear_min < prev_wear
)
SELECT COUNT(*)                              AS n_complete_tool_lives,
       ROUND(AVG(eol_wear), 1)               AS mean_end_of_life_wear_min,
       MIN(eol_wear)                         AS min_wear,
       MAX(eol_wear)                         AS max_wear,
       SUM(failed)                           AS n_tool_wear_failures,
       ROUND(1.0 * SUM(failed) / COUNT(*), 3) AS share_ended_by_failure
FROM lives;

-- name: high_risk_cycles
-- Cycles à risque cumulant plusieurs signaux faibles (exemple de requête métier).
SELECT udi, type, tool_wear_min, torque_nm, rpm,
       ROUND(delta_t_k, 1) AS delta_t_k,
       ROUND(power_w, 0)   AS power_w,
       machine_failure
FROM v_physics
WHERE (delta_t_k < 9.0 AND rpm < 1500)
   OR power_w < 4000 OR power_w > 8500
   OR wear_torque > 0.9 * CASE type WHEN 'L' THEN 11000 WHEN 'M' THEN 12000 ELSE 13000 END
ORDER BY machine_failure DESC, udi
LIMIT 15;
