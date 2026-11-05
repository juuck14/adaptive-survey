# Validation record

This records implementation verification, not evidence that adaptive selection
outperforms the baselines.

## Automated tests

`python -m pytest -q`: **117 passed**, local Python 3.12, CPU.

The suite covers input validation and duplicates, UUID splits, protected tasks,
scenario isolation, immutable fantasies, hidden-label perturbations, context
resets, proper choice scores, balanced paired-UUID bootstrap, MNL feature/rank
handling, neural history learning and checkpoint round trips, command-line
training/evaluation, and output overwrite protection.

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

## Scope

- The 1.27 GB complete response CSV was not downloaded or audited end to end.
- Existing CSV responses were not claimed to match this package's new RNG-key
  contract byte for byte.
- Bayesian/GNN model implementations and on-policy history augmentation are
  not included. Their integration boundary is defined in DESIGN.md.
- No supplied respondent records, generator parameters, trained checkpoints,
  signed download references, or private result files are committed.
