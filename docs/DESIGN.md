# Implementation contracts

## Public information and hidden answers

`Task`, `Profile`, `Observation`, and `SurveyState` are immutable public inputs. Policies receive a predictor, current public state, remaining candidates, and a seeded RNG. They never receive a `DataBundle` or `ResponseWorld`.

`ResponseWorld.reveal` checks candidate membership and returns one choice from one generator scenario. Only `score_checkpoint` reads held-out answers. Native/common predictions are saved before evaluation, and scores are never fed back to the policy.

Each UUID belongs to one split across every context/scenario. Context state resets between episodes. The runner rejects unknown/repeated tasks and inconsistent observation histories. Counterfactuals append one hypothetical choice to a separate immutable state.

## Adapter interface

```python
reset(profile, context) -> SurveyState
predict(state, tasks) -> numpy.ndarray  # [n_tasks, 4], A/B/C/NONE
observe(state, task, choice) -> SurveyState
fantasize(state, task, choice) -> SurveyState
```

`predict_many([(state, tasks), ...])` is an optional acceleration API. All methods use the same probabilities and decoder semantics. `Policy.select(predictor, state, candidates, rng)` returns a `Selection` containing the task ID and finite scores.

Use `runner.run_episode` for adapters or `run_survey` for a collection of episodes. `common_predictor` is an optional, frozen history readout; it receives exactly the revealed choices supplied to the native learner.

## GNN requirements

Training episodes must mask **all** focal respondent response edges outside the public history, not only the target edge. Cached node representations derived from those labels are also invalid. Free respondent-ID embeddings cannot retain hidden answers.

Freeze global weights and permanent training edges at test time. Compute required message passing inside an episode-specific temporary graph view; do not commit its node state to other test respondents. If support embeddings are cached/frozen, document which message paths are approximated. Do not accidentally eliminate the multi-hop hypothesis while claiming to test it.

Keep the same public features, history bank, choice loss, decoder where practical, ensemble resamples, and acquisition rule as Non-graph. Compare both identical histories and separately selected adaptive trajectories.

## Policies and estimands

The fixed policy filters a global training-prior greedy Fisher ranking. It is answer independent, but neither the original efficient block nor the optimal design conditional on each masked pool.

Weighted JS measures dependence in the model-induced fantasy-answer/future-choice distribution. It excludes q itself and hidden evaluation tasks. Nonnegativity does not establish useful information acquisition; compare validation rollouts with Random and entropy. A no-history predictor produces zero JS and seeded ties.

Primary adaptive evidence is a within-learner policy comparison. Native four-system comparisons include learner differences. An independent common checkpoint gives a readout-specific policy diagnostic, not a model-free quantity.

Population MNL analysis uses the same UUID/context population and fixes the reference feature basis for the subset fit. Under MMNL it reports reference agreement only. Rank and optimization failures are explicit; an unidentified parameter is not an estimated zero.

## Reproducibility

Split manifests bind the UUID partition, protocol, and input file hashes. Trained checkpoints bind their scenario and source split. History protocol checks include seed, holdout count, and initial count; diagnostic budget overrides preserve these fields.

Ensemble members are not independent experiment replicates. Paired confidence intervals resample UUIDs and preserve their context/budget cells. Incomplete grids are rejected. Training seed variability and new response-world variability must be reported separately.

The initial implementation uses uniform-cardinality training histories with the protocol's initial prefix and a frozen, seeded validation bank. Automatic on-policy history augmentation and posterior-latent information gain are not implemented.
