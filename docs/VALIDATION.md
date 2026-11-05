# Validation record

This records implementation verification, not evidence that adaptive selection
outperforms the baselines.

## Automated tests

`python -m pytest -q`: **136 passed**, local Python 3.12, CPU.

The suite covers input validation and duplicates, UUID splits, protected tasks,
scenario isolation, immutable fantasies, hidden-label perturbations, context
resets, proper choice scores, balanced paired-UUID bootstrap, MNL feature/rank
handling, neural history learning and checkpoint round trips, command-line
training/evaluation, optimizer-update accounting, preserved/fresh cohort
construction, multi-seed aggregation, saved-history diagnostics, and output
overwrite protection.

The neural learning test uses an artificial persistent-latent problem and
unseen UUIDs; it is a functional test of history use, not the project benchmark.

## Real-file compatibility

The separately supplied mapped profiles, task JSON, and 300-respondent generated
response sample were read outside the repository:

- 54 task definitions; 18 per context.
- 16,200 response rows; 5,400 per context.
- 300 retained respondents; 54 responses per respondent.
- Extra mapped profiles without responses are reported separately and are not
  included in the retained cohort.
- The package's 34-column MNL design exactly matched the external generator's
  feature matrix for all 16,200 profile/task pairs and an additional 108 generic
  demographic/task cases, including categories absent from the mapped sample.

## End-to-end executions

1. Generated artificial development data and ran the documented split/train/
   evaluate path with all four question policies.
2. Split the real generated sample into 180 train, 60 validation, and 60 test
   UUIDs, retaining each UUID's three contexts together.
3. Trained five Non-graph ensemble members for up to 12 epochs each, 512
   training episodes per epoch. The final fixed validation bank contained all
   60 validation UUIDs and all 180 UUID/context pairs.
4. Evaluated all 60 test UUIDs in three independent context episodes with
   Fixed, Random, entropy, and weighted JS at budgets 3/5/7/9. Recorded paired
   UUID bootstrap and timing/provenance outputs.
5. Completed selected/full pooled-MNL analysis for Random and weighted JS at
   budgets 3 and 9, using the reference's common 30-dimensional identifiable
   basis within the full 34-column specification. All four fits converged.
6. Fitted a PanelMAP readout to development data and ran Fixed/Random/entropy
   through the same evaluation CLI.
7. Loaded the actual external Generator handoff and generated a new three-person,
   162-row panel. Additional checks verified one MMNL taste per UUID across
   contexts and input-order-independent results under this package's RNG keys.

The small real-sample run produced essentially the same NLL for the four neural
policies. It verifies execution and comparison conditions; it does not establish
an adaptive performance gain. Full experiment tuning, response-seed repetitions,
and graph/Bayesian comparisons remain separate research work.

## Longer-training recheck

The initial 12 × 512 / 64 configuration allowed only 96 optimizer updates per
member. The recheck preserves the original training 180 and validation 60
respondents and draws 300 new test respondents after excluding all original
300 UUIDs. The supplied Generator produces the new test panel; original
training/validation labels are unchanged. Completeness, preserved rows, profile
mapping, exclusions and source/generator/model hashes were checked independently.

Three training-procedure seeds each fit a five-member ensemble. The cap is
120 × 1,024 / 64 = 1,920 updates, with a 720-episode validation bank and patience
30. Validation selected 304–528 updates across the 15 members. Each validation
bank covered all 60 UUIDs and 180 UUID/context pairs. Test scores did not choose
the checkpoint, architecture, loss or training budget.

Each frozen ensemble ran all four policies on all new 300 UUIDs × 3 contexts,
with identical protected tasks, two initial questions and rollout seed. The
primary budget-average JS − Random NLL difference was −0.002275, with paired
UUID bootstrap 95% interval [−0.005480, +0.000660]. This does not establish a
policy advantage. [RECHECK.md](RECHECK.md) records the complete protocol and
results. The old and new test cohorts are not directly compared as evidence of
a training effect.

The separate same-checkpoint H0/H9 diagnostic found NLL 1.171341 → 1.120757,
with paired UUID 95% interval for H9−H0 [−0.078295, −0.026942]. The actual
third-question choice was reproduced before a controlled change to both
revealed initial answers. The next question changed in 98/180, 93/180 and
91/180 cases across seeds, using the same hash-selected 180 episodes. This
confirms computational answer sensitivity, not true information gain.

A first-pass seed-2027 trace was observed with its last 12 rows missing, despite
complete scores and predictions. That output was preserved. Evaluation was
repeated with the identical checkpoint, data, protocol, rollout seed and thread
settings. The complete 32,400-row replay matched all previous query histories,
probabilities, metric values and existing acquisition scores exactly; the
32,388-row original matched its prefix in selection, response, candidate and
score fields, excluding timing measurements. No learner or policy configuration
changed.

The orchestration helper was also exercised with two independent processes on
artificial development data, including training, all four rollouts, and seed
aggregation. Regression fixtures check analytic seed means/SD, UUID clustering,
missing or mismatched evaluation cells, changed worlds/checkpoints/holdouts,
saved H9 use, answer interventions, and refusal to overwrite evaluator inputs.

## Scope

- The 1.27 GB complete response CSV was not downloaded or audited end to end.
- Existing CSV responses were not claimed to match this package's new RNG-key
  contract byte for byte.
- Bayesian/GNN model implementations and on-policy history augmentation are
  not included. Their integration boundary is defined in DESIGN.md.
- No supplied respondent records, generator parameters, trained checkpoints,
  signed download references, or private result files are committed.
