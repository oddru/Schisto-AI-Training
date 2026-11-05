# CHAPTER 3

# RESULTS AND DISCUSSION

This chapter presents the results in relation to SOP 1–SOP 6: dataset and model performance, controlled sensor-fault stress tests, projected AWD boundary behavior, controller software checks, and a staged quality-assurance plan. The study's engineering contribution is a reproducible Python prototype that keeps the Random Forest advisory separate from an explicit AWD policy, routes faults according to the policy's input requirements, and compares projected boundary outcomes under documented assumptions.

The evidence is organized by what each evaluation can establish. The generated benchmark measures model behavior against its rule-derived target; controlled perturbations compare responses to specified fault profiles; and one-hour projections compare controller paths under simulated actuator rates. These results provide a repeatable basis for software testing and identify practical next steps for verification. Field effectiveness, ecological or epidemiological outcomes, and physical ESP32 performance require separate measurements and are not outcomes of the present experiments.

## 3.1 Baseline Dataset Profile and Predictive Performance (SOP 1 and SOP 2)

### 3.1.1 Dataset profile and label characteristics

The active dataset contains 4,320 hourly records covering a generated 180-day period from January 1, 2024, at 00:00 through June 28, 2024, at 23:00. Each record contains 12 predictors, the separate `Snail_Habitat_Risk` label, and the `AWD_Gate_Action` prediction target. The predictors describe temporal context (`Month`, `Day_of_Week`, and `Hour`), crop stage, hydrology and water quality, soil, and vegetation. The dataset has no missing values.

The records were generated using a fixed-seed simulation rather than collected from field sensors. The generator assigns crop stages on a repeating 120-day schedule and simulates water depth using drawdown cycles, crop stage, daily variation, a simplified seasonal rainfall factor, and random variation. Soil moisture is coupled to water depth and rainfall; soil temperature follows a daily solar pattern; and flow, pH, and vegetation measurements are generated from configured relationships and random variation. This shared generator provides a stable benchmark for controlled comparisons and repeatable software tests.

`AWD_Gate_Action` encodes the simulated gate policy: `1` means open/reflood and `0` means close/maintain. During reproductive/flowering, the generator assigns an open action when water depth is below 2 cm. Outside flowering, it assigns an open action when water depth is at or below -15 cm or soil moisture is at or below 25%. The separate `Snail_Habitat_Risk` label uses configured environmental rules and is not included as a predictor in the gate-action model.

**Table 3.1. Selected descriptive statistics for the generated dataset**

| Variable | Unit | Minimum | Maximum | Mean |
|---|---:|---:|---:|---:|
| Water-table depth | cm | -16.811 | 5.374 | -4.136 |
| Inundation duration | days | 0.000 | 15.583 | 1.116 |
| Water-flow velocity | m/s | 0.000 | 0.350 | 0.148 |
| Water pH | pH | 6.028 | 7.405 | 6.750 |
| Soil moisture | % VWC | 20.684 | 80.959 | 53.732 |
| Soil temperature | °C | 19.819 | 32.997 | 25.427 |
| Vegetation coverage | % | 10.000 | 95.000 | 52.608 |
| Vegetation height | cm | 5.000 | 120.000 | 62.765 |

The generated gate-action target is imbalanced: 107 of 4,320 rows (2.48%) are positive open/reflood actions. The remaining 4,213 rows (97.52%) are close/maintain actions. The generated habitat-risk categories comprise 3,810 Low (88.19%), 478 Moderate (11.06%), and 32 High (0.74%) rows.

**Table 3.2. Generated label distribution**

| Label | Category | Records | Percentage |
|---|---|---:|---:|
| `AWD_Gate_Action` | 0 — close/maintain | 4,213 | 97.52% |
| `AWD_Gate_Action` | 1 — open/reflood | 107 | 2.48% |
| `Snail_Habitat_Risk` | Low | 3,810 | 88.19% |
| `Snail_Habitat_Risk` | Moderate | 478 | 11.06% |
| `Snail_Habitat_Risk` | High | 32 | 0.74% |

![Figure 3.1. Generated gate-action and habitat-risk label distributions](reports/chapter3_experiments/Figure_3_1_label_distributions.png)

Because positive actions account for only 2.48% of generated records, accuracy alone is not informative about the rarer action. Positive-class recall measures how many generated reflood requests are identified; precision measures how many positive predictions correspond to generated requests. Both are reported alongside accuracy and the confusion matrix.

The configured habitat-risk rules designate High risk when inundation exceeds 14 days, soil temperature is 15–30°C, pH is 5.5–7.9, and soil moisture is 20–80%. When that full rule is not met, Moderate risk is assigned for inundation exceeding 3 days, temperature 15–32°C, pH 5.3–8.1, and moisture 20–80%; other rows are Low. These rules make the study's environmental motivation explicit and reproducible. Their output is a simulated habitat indicator, not an observation of *Oncomelania hupensis* or disease transmission.

### 3.1.2 Chronological baseline

The initial baseline used a chronological 80/20 split. The first 3,456 records were used for fitting and the final 864 later records were held out. The held-out segment contained 20 positive and 844 negative actions. The pipeline one-hot encoded `Month`, `Day_of_Week`, and `Crop_Growth_Stage`, passed numeric predictors through, and fitted a 400-tree Random Forest with balanced class weights and random seed 42.

**Table 3.3. Chronological held-out baseline performance**

| Metric | Result |
|---|---:|
| Accuracy | 98.50% |
| Precision | 88.89% |
| Recall | 40.00% |
| F1-score | 55.17% |

**Table 3.4. Confusion matrix for the chronological baseline**

| Actual class | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 — close/maintain | 843 | 1 |
| 1 — open/reflood | 12 | 8 |

The model identified 8 of the 20 positive actions in this chronological test. This establishes a useful initial reference: on this split, the balanced default threshold was conservative for the rare positive class. The result motivated a separate experiment in which class weighting and decision threshold were selected on generated validation data, without using the locked test for tuning.

### 3.1.3 Validation-based model and threshold selection

A separate experiment tested validation-based operating-point selection. Four generated realizations (seeds 42–45; 17,280 records combined) were used for fitting. Four class-weight settings—balanced, positive-class weight 8, 16, and 32—were evaluated on validation seed 46 (4,320 records, including 104 positive actions). Thresholds from 0.05 to 0.95, in 0.01 increments, were compared using positive-class F2-score, which weights recall more heavily than precision. Ties were resolved by higher precision and then the higher threshold.

The selected validation configuration was positive-class weight 8 and probability threshold 0.51. It achieved 100.00% recall, 99.05% precision, and 99.81% F2-score on seed 46. The validation comparison is shown in Figure 3.2. This is a validation result under the synthetic generator, not a field-performance estimate.

![Figure 3.2. Validation operating-point comparison for class-weight and threshold candidates](reports/chapter3_tuning/Figure_3_2_validation_thresholds.png)

The selected and default models produced identical predictions on locked seed 47. Both achieved 99.88% accuracy, 100.00% precision, and 93.42% recall (71 true positives and 5 false negatives among 76 positive records), as shown in Table 3.5 and Figure 3.3. Thus, the selection procedure improved the validation operating point, but did not demonstrate an improvement over the balanced default on this particular locked test. Reporting this comparison avoids overstating the effect of tuning.

**Table 3.5. Validation and locked-test model comparison**

| Evaluation set | Configuration | Accuracy | Precision | Recall | F1 | F2 | TP / FN | TN / FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Validation, seed 46 | Balanced, threshold 0.50 | 99.93% | 98.10% | 99.04% | 98.56% | 98.85% | 103 / 1 | 4,214 / 2 |
| Validation, seed 46 | Positive weight 8, threshold 0.51 | 99.98% | 99.05% | 100.00% | 99.52% | 99.81% | 104 / 0 | 4,215 / 1 |
| Locked test, seed 47 | Balanced, threshold 0.50 | 99.88% | 100.00% | 93.42% | 96.60% | 94.67% | 71 / 5 | 4,244 / 0 |
| Locked test, seed 47 | Positive weight 8, threshold 0.51 | 99.88% | 100.00% | 93.42% | 96.60% | 94.67% | 71 / 5 | 4,244 / 0 |

On validation, selection raised recall from 99.04% to 100.00% while reducing false positives from two to one; F1 and F2 also increased. On the locked test, both configurations made the same predictions, detecting 71 of 76 positive actions with no false positives. This comparison shows that the validation-based procedure selected a strong operating point on its selection data and that the observed gain did not carry over as a difference from the default on this test. It is a transparent test of the selection process under the generator, not evidence of performance on independently collected farm data.

![Figure 3.3. Balanced default and validation-selected models on the locked synthetic test](reports/chapter3_tuning/Figure_3_3_locked_test_comparison.png)

The locked test used seed 47 and contained 4,320 records, including 76 positive actions. Model and threshold selection were completed before this test was evaluated. The agreement of the two models is evidence of reproducibility within this generator setup, not evidence that the classifier generalizes to a farm or to a different data source.

## 3.2 TFCST Sensor-Fault Robustness Analysis (SOP 3)

The validation-selected model (positive-class weight 8, probability threshold 0.51) was tested under four controlled sensor-fault profiles: Gaussian noise, impulse spikes, sensor drift, and sensor freezing. Each profile was applied at 5%, 10%, 15%, 20%, 25%, and 30% intensity. Fault injection was limited to the direct numeric sensor channels; calendar `Hour`, categorical context, and generator-derived `Inundation_Duration_Days` were left unchanged. Ten seeded corruption trials were performed for each fault-intensity condition on the fixed seed-47 test realization. Table 3.6 reports the mean and standard deviation of positive-class precision, recall, and F1-score across trials. The model and threshold remained fixed; variation therefore reflects the fault-injection realizations.

**Table 3.6. Locked-test classification under simulated sensor corruption (mean ± SD, %)**

| Fault profile | Intensity | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Gaussian noise | 5% | 78.7 ± 3.0 | 82.1 ± 4.5 | 80.3 ± 3.2 |
| Gaussian noise | 10% | 58.6 ± 3.3 | 75.7 ± 3.8 | 66.0 ± 3.3 |
| Gaussian noise | 15% | 43.2 ± 2.8 | 70.9 ± 4.1 | 53.6 ± 3.0 |
| Gaussian noise | 20% | 32.8 ± 1.7 | 70.3 ± 5.9 | 44.7 ± 2.6 |
| Gaussian noise | 25% | 24.4 ± 1.8 | 65.0 ± 3.6 | 35.4 ± 2.4 |
| Gaussian noise | 30% | 20.9 ± 1.2 | 64.7 ± 3.1 | 31.5 ± 1.5 |
| Impulse spikes | 5% | 41.5 ± 2.0 | 91.2 ± 1.6 | 57.1 ± 2.1 |
| Impulse spikes | 10% | 25.2 ± 1.0 | 88.9 ± 2.1 | 39.2 ± 1.3 |
| Impulse spikes | 15% | 18.7 ± 0.8 | 88.2 ± 2.6 | 30.8 ± 1.2 |
| Impulse spikes | 20% | 14.5 ± 0.7 | 86.3 ± 2.5 | 24.8 ± 1.1 |
| Impulse spikes | 25% | 11.1 ± 0.8 | 82.0 ± 4.9 | 19.6 ± 1.4 |
| Impulse spikes | 30% | 9.5 ± 0.4 | 81.1 ± 3.0 | 16.9 ± 0.8 |
| Sensor drift | 5% | 100.0 ± 0.0 | 27.6 ± 0.0 | 43.3 ± 0.0 |
| Sensor drift | 10% | 100.0 ± 0.0 | 9.2 ± 0.0 | 16.9 ± 0.0 |
| Sensor drift | 15% | 100.0 ± 0.0 | 7.9 ± 0.0 | 14.6 ± 0.0 |
| Sensor drift | 20% | 100.0 ± 0.0 | 5.3 ± 0.0 | 10.0 ± 0.0 |
| Sensor drift | 25% | 100.0 ± 0.0 | 3.9 ± 0.0 | 7.6 ± 0.0 |
| Sensor drift | 30% | 100.0 ± 0.0 | 1.3 ± 0.0 | 2.6 ± 0.0 |
| Sensor freezing | 5% | 100.0 ± 0.0 | 90.8 ± 5.6 | 95.1 ± 3.2 |
| Sensor freezing | 10% | 97.9 ± 6.8 | 84.3 ± 7.3 | 90.5 ± 6.3 |
| Sensor freezing | 15% | 100.0 ± 0.0 | 80.1 ± 8.8 | 88.7 ± 5.8 |
| Sensor freezing | 20% | 50.7 ± 42.8 | 82.9 ± 10.1 | 51.4 ± 30.4 |
| Sensor freezing | 25% | 79.2 ± 34.2 | 74.5 ± 5.4 | 72.2 ± 22.3 |
| Sensor freezing | 30% | 59.3 ± 42.9 | 77.5 ± 10.1 | 56.2 ± 27.0 |

![Figure 3.4. Positive-class F1-score under simulated sensor faults; shading indicates one standard deviation across trials](reports/chapter3_tuning/Figure_3_4_robustness_curves.png)

The results separate into distinct response patterns. With Gaussian noise, mean F1 declined from 80.33% at 5% intensity to 31.55% at 30%, alongside lower precision. Impulse spikes retained recall above 81% across the tested range, while precision fell from 41.54% to 9.46%, indicating an increasing false-positive burden in this setup. Drift preserved 100% precision but recall decreased from 27.63% at 5% to 1.32% at 30%; the model therefore missed a larger share of positive actions as the simulated drift increased. Freezing retained a mean recall between 74.47% and 90.79%, but results varied substantially at 20–30% intensity; for example, F1 was 51.37% ± 30.42 percentage points at 20%.

Figure 3.4 and Table 3.6 complement one another: the figure makes the F1 trends visible, while the table reports precision and recall alongside the mean and standard deviation for every tested intensity. Together, they characterize how this trained model responds to these specific perturbations and suggest distinct bench-test questions for drift, transient spikes, and repeated readings. The percentages are controlled injection settings, not measured sensor-error tolerances or recommended field alert thresholds.

## 3.3 Projected AWD Boundary Outcomes (SOP 4)

The boundary experiment applies the study's configured AWD limits: outside flowering, water depth is compared with -15 cm and +5 cm boundaries; during flowering, it is compared with 0 cm and +5 cm boundaries. In the one-hour projection, an open gate increases depth by 1.0 cm and a closed gate decreases it by 0.5 cm. These rates are simulation assumptions, not measured actuator response. The direct projection uses Random Forest actions; the policy-controlled projection uses the configured rule and one-hour interlock.

The fault-free locked test produced 59 projected boundary violations across 4,320 records (1.37%) for direct Random Forest actions and 9 (0.21%) for the policy-controlled path. The ten-trial fault tests show that this difference depends on the injected fault profile: under impulse spikes and freezing, the policy-controlled path had fewer projected violations on average at the tested levels; under Gaussian noise and drift, critical-input safe mode produced 120 violations (2.78%) in each trial. The complete results in Table 3.7 and the trend lines in Figure 3.5 make both the clean comparison and the fault-dependent behavior visible.

**Table 3.7. Projected AWD boundary violations across tested fault intensities**

| Fault profile | Intensity | Direct model, mean violations ± SD | Policy-controlled, mean violations ± SD |
|---|---:|---:|---:|
| Clean baseline | 0% | 59.0 | 9.0 |
| Gaussian noise | 5% | 57.5 ± 4.1 | 120.0 ± 0.0 |
| Gaussian noise | 10% | 57.0 ± 2.9 | 120.0 ± 0.0 |
| Gaussian noise | 15% | 57.2 ± 5.2 | 120.0 ± 0.0 |
| Gaussian noise | 20% | 56.2 ± 4.8 | 120.0 ± 0.0 |
| Gaussian noise | 25% | 63.2 ± 5.5 | 120.0 ± 0.0 |
| Gaussian noise | 30% | 68.8 ± 6.6 | 120.0 ± 0.0 |
| Impulse spikes | 5% | 64.2 ± 3.8 | 10.3 ± 0.8 |
| Impulse spikes | 10% | 67.2 ± 2.9 | 12.3 ± 2.5 |
| Impulse spikes | 15% | 73.1 ± 5.1 | 13.6 ± 2.5 |
| Impulse spikes | 20% | 79.4 ± 7.5 | 13.6 ± 2.7 |
| Impulse spikes | 25% | 84.2 ± 5.1 | 16.4 ± 2.7 |
| Impulse spikes | 30% | 88.5 ± 5.8 | 16.4 ± 3.3 |
| Sensor drift | 5% | 101.0 ± 0.0 | 120.0 ± 0.0 |
| Sensor drift | 10% | 113.0 ± 0.0 | 120.0 ± 0.0 |
| Sensor drift | 15% | 114.0 ± 0.0 | 120.0 ± 0.0 |
| Sensor drift | 20% | 116.0 ± 0.0 | 120.0 ± 0.0 |
| Sensor drift | 25% | 117.0 ± 0.0 | 120.0 ± 0.0 |
| Sensor drift | 30% | 119.0 ± 0.0 | 120.0 ± 0.0 |
| Sensor freezing | 5% | 60.9 ± 4.0 | 12.1 ± 6.5 |
| Sensor freezing | 10% | 65.4 ± 5.3 | 21.5 ± 5.7 |
| Sensor freezing | 15% | 67.7 ± 5.6 | 23.6 ± 7.0 |
| Sensor freezing | 20% | 99.6 ± 44.0 | 31.2 ± 7.4 |
| Sensor freezing | 25% | 90.1 ± 32.9 | 41.8 ± 7.7 |
| Sensor freezing | 30% | 120.4 ± 47.5 | 40.5 ± 9.5 |

*Each injected-fault result is the mean across ten trials, each evaluated over 4,320 records. The clean baseline is a single locked-test run. Counts are projected outcomes under the configured one-hour model, not observed field events.*

![Figure 3.5. Direct model and policy-controlled projected AWD boundary violations under simulated corruption](reports/chapter3_tuning/Figure_3_5_awd_safety.png)

![Figure 3.6. Mean positive-class precision and recall across sensor-fault intensity levels; shaded bands show one standard deviation across ten corruption trials. Dotted lines indicate clean-test reference levels.](reports/chapter3_tuning/Figure_3_6_fault_precision_recall.png)

Figure 3.6 makes the fault-specific precision-recall trade-offs visible. Under drift, the model preserves precision in these runs while recall declines; under impulse spikes, recall remains comparatively higher while precision declines. Reading this figure alongside the projection results is useful because classification metrics and projected water-boundary outcomes answer different questions: one measures agreement with the generated action labels, and the other measures the simulated depth effect of selected actions under the assumed one-hour response.

On clean locked-test inputs, the Random Forest and rule policy differed on 5 of 4,311 records compared by the controller. Under the one-hour simulation, the controlled projection had 9 boundary violations (0.21%), compared with 59 (1.37%) for direct model actions. The rule policy matches the generated labels because it encodes the same rule; this establishes implementation consistency, not independent agronomic validation.

On the locked test, the channel-level oracle scenarios demonstrate that routing can distinguish inputs used by the policy from channels that are not part of its rule. A depth-only flag routed all 4,320 records to safe mode and yielded 120 projected violations (2.78%); a soil-moisture-only flag yielded 112 (2.59%), with 3,873 safe-mode and 447 degraded-mode records. Flags limited to flow, pH, soil temperature, or vegetation allowed degraded policy operation on 4,311 records; nine records entered safe mode through the repeated-reading check, and the projection had nine violations (0.21%). These tests measure controller routing when a fault flag is supplied, not the ability of a detector to discover a real fault.

This comparison also identifies the next controller design question: how to select a site-appropriate response when a critical input is unavailable. In the tested combined critical-input scenarios, holding the initial closed command resulted in 120 projected violations (2.78%). This outcome is specific to the starting states, limits, and assumed actuator rates in the simulation. The interlock can compare candidate projections, but cannot ensure that a feasible action exists when neither candidate returns the projected depth within limits. These findings support continued implementation and bench testing of the routing and projection logic, together with a separately justified fallback procedure.

The habitat-risk criteria represent configured environmental conditions relevant to the study motivation. The experiments report generated gate-action classification and projected water-depth outcomes; they do not measure greenhouse-gas emissions, snail abundance, habitat change, or epidemiological response. The boundary table should therefore be interpreted as a comparison within the stated simulation, not as measured ecological impact or field safety.

## 3.4 Controller Logic and Embedded Demonstrator Status (SOP 5)

The controller separates three concerns: input validity, the rule used to request an action, and the interlock that evaluates a one-hour projected depth. Water depth is treated as critical at every crop stage; soil moisture is critical outside flowering because the configured rule uses it there. A fault in either critical input enters safe mode and holds the previous gate command. If only a channel not used by the policy is flagged, the rule can continue in degraded mode. The Random Forest supplies an advisory comparison and does not command the simulated gate. For valid rule inputs, the interlock rejects actions projected to cross a configured limit and selects the smaller projected violation if neither option remains within limits. Figure 3.7 summarizes this evaluation design.

![Figure 3.7. Evaluation and controller architecture, showing the advisory model, policy/interlock path, and safe-mode route](reports/chapter3_tuning/Figure_3_7_evaluation_architecture.png)

The clean and channel-isolated comparisons demonstrate how the implemented components behave under their test assumptions. On the locked synthetic test, policy-controlled projection reduced violations from 59 to 9 on clean inputs. Fault-routing results show continued degraded-mode operation for faults in noncritical channels, while critical-channel faults enter safe mode. In the combined critical-input simulation, holding the previous command produced 120 projected violations. This provides a concrete next engineering task: evaluate a site-specific fallback using verified sensor and hydraulic behavior.

Twenty controller/model tests passed, covering deterministic and channel-specific corruption, generated-seed consistency, policy/label consistency, threshold selection, predictor threshold handling, stale and invalid readings, critical and noncritical fault routing, repeated-value detection, advisory model behavior, and the one-hour interlock.

The `wokwi/` PlatformIO project was compiled successfully and run in the Wokwi ESP32 simulator. At startup, the firmware executed 12 labeled checks: reset behavior; low-depth, upper-limit, and flowering interlock cases; stale-reading and fault-flag safe mode; and the three-reading frozen-depth check, with resets between independent cases. All 12 reported `DEMO_CHECK PASS`, and the serial output showed the expected gate and safe-mode states. The diagram also includes a virtual servo configured to represent the gate flap, with closed and open states mapped to 0° and 90°, respectively; the servo is a visual indicator, not a calibrated model of gate position or physical actuator performance. The requested gate action and sensor reading remain serial-injected test inputs; the Random Forest is not running on the ESP32. Physical sensors, actuator response, and field operation were not tested. Bench testing with physical components remains a subsequent validation step.

The prototype explores sensor-informed gate support. Operator workflow and unattended operation are part of later evaluation stages that depend on hardware integration, reliable communications, actuator safeguards, manual override, and maintenance procedures.

## 3.5 CRISP-ML(Q) Quality Assurance and Staged Validation Plan (SOP 6)

The controlled fault levels provide repeatable inputs for the next quality-assurance stage. Table 3.8 translates the observed response patterns into candidate bench checks and safeguards. The injection percentages support reproducible comparisons in this simulator; defining field tolerances or a recalibration schedule requires measured sensor error against a calibrated reference.

**Table 3.8. TFCST evidence and candidate QA actions for further testing**

| Condition / measured evidence | QA test focus | Candidate engineering response |
|---|---|---|
| Sensor drift at 5%: recall 27.63%; F1 43.30% | Reproduce this sensitivity in controlled acceptance tests | Compare against a calibrated reference and measure error in engineering units before defining a field threshold. |
| Sensor drift at 10%: recall 9.21%; F1 16.87% | Verify fault flagging and the separately approved fallback under severe injected drift | Do not treat this simulated percentage as a device tolerance; test a site-reviewed response. |
| Gaussian noise at 30%: precision 20.87%; recall 64.74%; F1 31.55% | Assess noise sensitivity and signal-quality checks | Test filtering or wiring changes against a predeclared evaluation set and both precision and recall. |
| Impulse spikes at 30%: precision 9.46%; recall 81.05%; F1 16.94% | Assess transient-spike handling and repeated actuation safeguards | Test plausibility checks and command-rate limits before any hardware integration. |
| Sensor freezing at 20%: F1 51.37% ± 30.42 percentage points | Repeat temporal-freeze detection tests across sensor channels | Check whether the implemented repeated-reading check detects the freeze patterns of interest. |
| Clean policy-controlled projection: 9 violations versus 59 for direct model actions | Preserve as a regression comparison under current synthetic assumptions | Keep the policy/interlock comparison, and separately verify policy thresholds and actuator response with justified references. |
| Stale, missing, invalid, or repeated readings | Exercise checks for readings older than 3,600 seconds, invalid values, external fault flags, and three identical consecutive numeric readings | Confirm the diagnostic route and operator procedure; separately determine whether the held command is appropriate for the site. |

The percentages in Table 3.6 describe the synthetic fault injectors; they are not recommended field alert thresholds. Implemented controller checks include input validity, externally supplied channel-fault flags, and repeated-value detection. Drift and noise detection, calibration intervals, and fallback actions are candidates for evaluation using measured sensor residuals, reference instrumentation, and local operational review.

The following protocols translate the measured behavior into an engineering QA workflow:

1. **Repeat the measured stress conditions.** Use the 5% and 10% drift settings as reproducible synthetic test cases, then measure the corresponding error in engineering units with calibrated references before defining any field alarm thresholds.
2. **Use fault-specific measures.** Report precision and recall together; the simulations show that drift and impulse faults affect them differently. Test range checks, transient handling, and temporal consistency as separate controller functions.
3. **Exercise invalid and stale input paths.** Test readings older than 3,600 seconds, missing or invalid values, external fault flags, and three consecutive identical numeric readings. Confirm that diagnostics are clear and validate the site-specific operator response for each condition, including whether holding the last command is appropriate.
4. **Maintain the advisory/policy separation.** Continue to treat the Random Forest as advisory in the controller prototype. Assess policy thresholds and interlock behavior separately using independently justified agronomic criteria and measured actuator dynamics.
5. **Evaluate positive-class performance and projected outcomes.** For every model or sensor configuration, report support, recall, precision, F1/F2, false positives, false negatives, and projected boundary outcomes. Record the split and number of positive cases to make comparisons interpretable.
6. **Use measured history for maintenance planning.** Estimate sensor residuals against reference instruments over time before proposing recalibration intervals. The present synthetic stress percentages cannot establish a service calendar.
7. **Progress in controlled stages.** Retain the successful Wokwi build and 12-check run as a regression baseline, and verify the virtual servo's open/closed indication after diagram or firmware changes. Then test sensors and actuator response on a bench with communications loss and manual override; consider a supervised pilot only after predefining acceptance criteria and stop conditions.

The next engineering stages are to verify the virtual servo's visible response to the open and closed states, bench-test calibrated sensors and actuator response (including communications loss and manual override), and only then consider supervised field evaluation with an independent time period. A subsequent agronomic study could pair water-management records with defined habitat and snail observations; any greenhouse-gas claim would require its own measurement protocol. Each stage should have predeclared sensor, actuator, false-positive/negative, boundary, and stop criteria. Retaining configurations, timestamps, reference measurements, commanded and observed gate states, and interventions will make later decisions reproducible.

Within the configured habitat rules, High risk is assigned using prolonged inundation and specified moisture, temperature, and pH ranges relevant to the study's motivation. A possible future hypothesis is that reliable sensor-informed AWD operation could change water conditions represented by those rules. Testing that hypothesis requires actual water-management records and observed habitat and snail outcomes; the present results do not measure those effects.

## Chapter Summary

For SOP 1 and SOP 2, the chapter profiles the generated dataset and establishes model benchmarks, then documents validation-only selection of a positive-class weight and threshold. The selected operating point improved precision, recall, F1, and F2 on the validation set. On the locked test, it matched the balanced default, with 71 of 76 positive actions detected, 100% precision, and 93.42% recall. This is a clear, reproducible comparison of model selection under the project's generated data.

For SOP 3, the ten-trial fault tests provide a response profile across four corruption types and six intensities. Drift primarily reduced recall, impulse spikes primarily reduced precision, and freezing produced the greatest variability at higher tested intensities. For SOP 4, the one-hour simulation recorded 59 direct-model versus 9 policy-controlled boundary violations on clean inputs. The full fault table shows that this comparison varies by fault type: policy control reduced projected violations under spike and freezing tests, while critical-input safe mode yielded 120 projected violations under the tested Gaussian-noise and drift conditions. These results identify a specific next design and test task: evaluate a site-appropriate response to unavailable critical readings using measured sensor and hydraulic behavior.

For SOP 5, Python software tests and a completed Wokwi run document the advisory-model/policy separation, channel-aware routing, interlock behavior, and corresponding firmware checks. The Wokwi startup suite reported 12 of 12 checks passing for the tested virtual-ESP32 cases. A virtual servo is included in the circuit as a gate-flap visual indicator and follows the firmware's open/closed state; this does not model real gate mechanics. Calibrated bench testing with physical sensors and an actuator remains the next validation stage.

For SOP 6, the measured patterns are translated into candidate QA checks and a staged path from software verification to calibrated bench testing and supervised evaluation. The focused controller/model suite passed 20 tests. Together, the chapter provides reproducible results, comparisons, figures, and concrete follow-on tests. Its findings support the prototype's software and simulation contributions while keeping field, agronomic, ecological, and public-health claims tied to evidence that remains to be collected.

## Reproducibility

The chronological experiment can be rerun using `python src/chapter3_experiments.py --repeats 10`; its outputs are under `reports/chapter3_experiments/`. The validation-tuned experiment can be rerun using `python -m src.tune_gate_model --repeats 10`; its model, tables, and figures are under `reports/chapter3_tuning/`. The tuning procedure uses generated seeds 42–45 for training, seed 46 for validation, and seed 47 for the locked test. Sensor corruption is tested at 5–30% in 5-point increments with ten repeats per condition. The supplied-log audit described in Chapter 2 is regenerated with `python -m src.evaluate_external_irrigation_data`. The separate soil-moisture benchmark is summarized in Appendix B and can be rerun using `python -m experiments.irrigation_forecasting.download_data` followed by `python -m experiments.irrigation_forecasting.evaluate` in its dedicated environment; see `experiments/irrigation_forecasting/README.md`. Supplementary figures, including the Chapter 2 audit figure, are generated with `python -m src.generate_chapter3_figures` after the audit outputs exist. One-hour actuator response and AWD boundaries are the configured assumptions recorded with the experiment outputs. The ESP32 firmware is in `wokwi/`; it builds with the PlatformIO `esp32dev` environment and runs through the Wokwi extension. Its startup checks print individual `DEMO_CHECK` results to the serial output. Full Python scripts, automated tests, notebook code cells, and Wokwi firmware listings are compiled in `appendices.md`. These virtual results do not replace physical bench testing.
