# External Irrigation Data Audit

This report summarizes the three supplied time series without training or evaluating the project's Random Forest. Source files are read-only; only aggregate reports are written.

## Recorded data

- Coverage: 2022-07-12 00:00:00 through 2022-09-16 23:57:37.
- Relay records: 17427; state changes between adjacent records: 118, of which 118 occurred across gaps of no more than 10 minutes; recorded states: {0.0: 14038, 1.0: 3389}.
- Relay records with a preceding moisture reading no more than 30 minutes old: 17179/17427 (98.6%).
- Flow readings are monotonic in the supplied file: yes. This does not establish whether the field is a cumulative counter or its unit.

## Limits on interpretation

The folder does not provide verified provenance, site/crop context, a data dictionary, sensor units, relay-state meaning, or confirmation of the flow-meter semantics. The streams are asynchronous and contain long gaps; see `stream_quality.csv`. The 20-minute moisture and 5-minute flow/relay reference intervals used to count long gaps are inferred from the observed cadence, not specified by source metadata. Moisture summaries by relay state are descriptive co-observations only, not causal effects, recommended thresholds, or evidence that logged actions were agronomically safe.

The records do not contain the current model's required water-table depth, crop stage, and other features. Therefore they cannot be passed to the existing classifier or treated as independent gate-action labels. No model was retrained or changed by this audit.

## Next evidence needed

Verify the dataset source/citation, site and crop, channel definitions and units, whether relay `1` means energized/open, whether `litres` is a cumulative counter, and whether the data are from a real farm. Then have an agronomist or system owner confirm which analyses and safety conclusions are appropriate.

Detailed aggregate tables: `stream_quality.csv` and `relay_moisture_summary.csv`.
