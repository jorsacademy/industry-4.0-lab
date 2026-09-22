# Final Comparative Campaign Results

## Status

Phase 4 is complete. The predeclared final-test campaign finished successfully without controller retuning after final-test access.

- GitHub Actions run: `33243737489`
- run head SHA: `eb90da3818f3aec17e0be5b4940a7561b97700d5`
- final artifact ID: `9712700007`
- artifact name: `final-comparative-eb90da3818f3aec17e0be5b4940a7561b97700d5`
- artifact digest: `sha256:89e6749deee96fcaa665e17e4c292973848f09fed69d336cef4624635f2c880b`
- final-test status: `final_test_complete`
- no-final-test-tuning flag: `true`

The run head and the later merged scientific-code tree are content-identical; the provenance difference is commit history, not experiment code.

The full raw `final_runs.csv` remains in the retained GitHub Actions artifact. Repository snapshots contain the compact manifest, result hashes, primary WTT table, and PPO training-seed dispersion.

## Frozen design

The final campaign used the controller settings frozen before final-test access:

- PPO training seeds: `101, 202, 303, 404, 505`;
- PPO training budget: 150,000 timesteps per member;
- rolling-horizon CP-SAT: horizon `8`, solve budget `100 ms`;
- eight deterministic dispatching rules;
- nominal final seeds: `20000–20099`;
- stress final seeds: `30000–30099`;
- nine predeclared stress scenarios plus nominal;
- 5,000 bootstrap resamples;
- 10,000 paired randomization permutations;
- Holm correction across the twenty primary PPO-vs-baseline weighted-tardiness tests.

For the primary PPO analysis, the five frozen PPO realizations are first averaged within each environment seed. Paired inference is then performed across environment seeds. The primary weighted-tardiness interval additionally uses a hierarchical bootstrap that resamples both PPO training seeds and environment seeds. The `5 x N` PPO observations are not treated as independent.

## Primary result

The primary research question does **not** support a claim that PPO improves scheduling performance over the strongest fixed dispatching rule in this environment.

`WEIGHTED_COMPOSITE` has lower mean priority-weighted tardiness than the five-training-seed PPO mean in all ten final scenarios.

| Scenario | PPO seed-mean WTT | Weighted Composite WTT | PPO penalty vs WC | CP-SAT WTT | PPO improvement vs CP-SAT |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal | 70.7847 | 63.2547 | 11.90% | 69.1594 | -2.35% |
| demand_120 | 211.7529 | 192.0211 | 10.28% | 214.6148 | 1.33% |
| demand_140 | 383.0956 | 345.8389 | 10.77% | 390.3137 | 1.85% |
| demand_160 | 547.4742 | 489.9622 | 11.74% | 572.7199 | 4.41% |
| breakdown_2x | 105.3940 | 95.9204 | 9.88% | 104.9812 | -0.39% |
| breakdown_4x | 166.9652 | 152.2761 | 9.65% | 166.1493 | -0.49% |
| tight_due_085 | 106.4317 | 98.9091 | 7.61% | 104.1211 | -2.22% |
| slow_machines_090 | 172.4179 | 153.4774 | 12.34% | 167.1815 | -3.13% |
| setup_2x | 195.6319 | 186.9471 | 4.65% | 205.5845 | 4.84% |
| compound_stress | 820.9690 | 757.3032 | 8.41% | 896.0073 | 8.37% |

A positive value in the last column means PPO has lower WTT than CP-SAT.

### PPO versus Weighted Composite

The PPO training-seed mean is worse in all ten scenarios. Hierarchical bootstrap intervals for the improvement statistic (positive favors PPO) exclude zero on the negative side in nine scenarios. `setup_2x` is borderline under the hierarchical interval even though its environment-seed paired Holm-adjusted test is nominally below 0.05.

This means the correct portfolio claim is not “RL beats heuristics.” The evidence says the strong fixed weighted-composite rule is the best WTT controller in the current dynamic parallel-machine formulation.

### PPO versus rolling-horizon CP-SAT

Once PPO training-seed uncertainty is retained, only `compound_stress` provides robust evidence that the PPO training-seed mean beats CP-SAT on weighted tardiness:

- mean WTT improvement: **75.0382**;
- percent improvement: **8.37%**;
- hierarchical 95% interval: **[1.1970, 142.0060]**;
- Holm-adjusted paired p-value: **0.0020**.

For `demand_160`, the environment-seed paired analysis favors PPO, but the hierarchical interval spans zero (`[-38.7201, 86.5731]`). It therefore is not treated as a robust PPO win after training-seed uncertainty is included.

No scenario provides a robust hierarchical conclusion that CP-SAT beats the PPO seed mean on primary WTT. However, CP-SAT also does not beat the strongest fixed rule on WTT.

## Compute and operational trade-offs

The controllers differ materially in online compute:

- Weighted Composite mean decision latency is approximately `0.0004–0.0005 ms`;
- PPO mean decision latency is approximately `0.26–0.27 ms`;
- CP-SAT mean decision latency ranges from approximately `21–69 ms` across final scenarios.

CP-SAT often improves on-time rate relative to Weighted Composite despite higher weighted tardiness. For example, nominal on-time rate is about `84.70%` for CP-SAT versus `82.47%` for Weighted Composite. This is an operational trade-off rather than a contradiction: the primary objective is priority-weighted tardiness, not unweighted on-time percentage.

## PPO training-seed instability

Training-seed variation is operationally important and grows under heavy demand/disruption.

Across the five frozen PPO members, WTT standard deviation across training seeds increases from **7.3307** in nominal conditions to **86.0259** under compound stress.

Three diagnostic patterns are especially important:

- PPO seed `303` has an episode-level KPI vector exactly identical to `WEIGHTED_COMPOSITE` for all `1,000` final scenario-seed episodes.
- PPO seed `202` has an episode-level KPI vector exactly identical to `MINIMUM_SETUP` / `SAME_FAMILY_FIRST` for all `1,000` final scenario-seed episodes.
- PPO seed `404` exactly matches the Weighted Composite episode outcome in `825 / 1,000` final episodes.

These are outcome-equivalence diagnostics, not direct action-trace proofs. They are consistent with some PPO training runs collapsing toward existing deterministic heuristics instead of learning a materially distinct adaptive switching policy. Future work should record action/rule-selection traces and policy entropy to test this directly.

## Scientific conclusion

For the current dynamic heterogeneous parallel-machine model:

1. **Weighted Composite is the primary WTT winner across the tested nominal and OOD regimes.**
2. **PPO does not provide robust incremental value over the strongest fixed rule when all five training seeds are retained.**
3. **PPO can outperform the frozen rolling-horizon CP-SAT under compound stress, but that does not establish an RL advantage because the fixed Weighted Composite rule remains better.**
4. **PPO training-seed instability is large enough that best-seed reporting would materially overstate RL performance.**
5. **The negative result is informative:** the present state/action/reward formulation does not justify the complexity of PPO for this scheduling model.

The next manufacturing research step should therefore not be PPO hyperparameter tuning on final seeds. It should change the problem structure so adaptivity has a credible source of value: true flexible job-shop routing/precedence, richer sequence-dependent setup matrices, explicitly pre-generated exogenous disruptions, urgent-order bursts, and action/rule trace diagnostics. Final seeds remain locked.

## Evidence files

Compact frozen evidence is stored under `results/final_comparative/`:

- `final_campaign_manifest.json`
- `final_result_hashes.json`
- `primary_wtt_results.csv`
- `ppo_training_seed_dispersion.csv`

The complete seed-level raw file and full per-model comparison table remain in GitHub Actions artifact `9712700007`, whose SHA-256 artifact digest is recorded above.
