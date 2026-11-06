# Implementation contracts

## Current comparison and question budgets

The implemented comparison is Fixed efficient design versus Non-graph adaptive
selection. Bayesian and GNN remain later adapters in the broader research design.
The default CLI policies are `fixed` and `weighted_js`.

Every budget includes four common initial questions, two from each block, within
one respondent/context episode. Contexts reset personal state. The requested
default totals are **6, 9, 12, 15, 18**, so the post-initial counts are 2, 5, 8,
11, 14. UUIDs, task attributes, contexts, scenario and choice realization are
paired across methods within each evaluation track.

| Track | Candidate tasks | Protected tasks | Default budgets | Purpose |
|---|---:|---:|---|---|
| Prediction | 14 | 4 | 6, 9, 12 | Predict the same unasked tasks |
| Preference | 18 | 0 | 6, 9, 12, 15, 18 | Collect responses for common population MNL |

These are separate rollouts. With eighteen available tasks, collecting fifteen
or eighteen while preserving four unasked targets is infeasible. The prediction
track reports these budgets as unavailable, with no synthetic or in-sample NLL.
The preference track records collections, not held-out predictions. `estimate`
uses this full-pool track by default. `--tracks prediction` or `--tracks preference`
runs only the selected track; `--track` on `estimate` selects a saved track.

The project documents call for a fixed order or fixed subsets, caution that a
nine-question efficient block does not establish short-prefix efficiency, and
separate prediction from population preference estimation. These decisions follow
that distinction:

- [Research design](https://docs.google.com/document/d/1ZV1PQ-watURbW9PZrfmbb0D8aNIS3syqUwEg6SMHU1I/edit?tab=t.0): sections 5, 6.2, 6.3 and 8.
- [Research progress](https://docs.google.com/document/d/1ZV1PQ-watURbW9PZrfmbb0D8aNIS3syqUwEg6SMHU1I/edit?tab=t.na2sy116d3ct): common initial tasks, fixed versus response-dependent selection.
- [Report workflow](https://docs.google.com/document/d/1ZV1PQ-watURbW9PZrfmbb0D8aNIS3syqUwEg6SMHU1I/edit?tab=t.syusby9pm3yi): methods and the two evaluation targets.

## Fixed efficient design

`FixedEfficientDesign` estimates task value from a population MNL fitted only on
training respondents, using the same 34 utility features as the final estimator.
It never receives a generator model or test choices. Training profiles supply
the reference covariate distribution; at most 256, deterministically selected
from sorted training UUIDs, are used for the Fisher average by default.

For task $q$ and training profile $i$, let $Z_{iq}$ be its four-by-feature utility
design matrix and $p_{iq}=\operatorname{softmax}(Z_{iq}\widehat\beta_{train})$.
The expected task information is

$$
I_q=\frac{1}{N_{train}}\sum_i
Z_{iq}^{\top}\left[\operatorname{diag}(p_{iq})-p_{iq}p_{iq}^{\top}\right]Z_{iq}.
$$

For each context, an orthonormal basis $B$ spans the supported directions of
$\sum_q I_q$ over the complete context bank. Unidentified directions are dropped
before planning and the rank is recorded. Write $\widetilde I_q=B^{\top}I_qB$.
Starting from the answer-independent common initial set $S_0$, the design uses

$$
M(S)=\lambda I+\sum_{q\in S}\widetilde I_q,\qquad
q^*=\arg\max_{q\in C\setminus S}
\{\log\det M(S\cup\{q\})-\log\det M(S)\}.
$$

The ridge is 1.0 by default and can be set with `--fixed-ridge`. Candidates
within $10^{-12}$ of the maximum use task-ID order to break ties. This generates
nested prefixes for all budgets along one predetermined order. It is a
regularized local D-optimal greedy approximation within the existing task bank.
It does not establish global optimality or inherit the original blocks' claimed
design efficiency.

All plans are constructed **before the first test answer is revealed**. Each
plan includes the actual candidate mask and common initial questions. Prediction
episodes can have different public masks; a distinct plan is precomputed for each
unique mask/initial set. Full-pool preference episodes use the same order for all
respondents in a context. Personal profiles, revealed answers, and the neural
predictor never change a Fixed plan. The learner may still use Fixed's revealed
answers when predicting held-out choices.

`fixed-plans.json` records candidate IDs, initial IDs, order, information rank,
ridge, and prefix log-determinants. `summary.json` binds this file by SHA-256 and
records the training MNL fit and Fisher profile count. This replaces the old
global-rank-then-filter rule, which ignored initial and unavailable tasks during
optimization.

Fixed's training-prior fit, Fisher construction and plan construction times are
recorded separately in `fixed_planning_timing`. Per-policy rollout timing includes
online selection and observation updates. Amortized offline planning should not
be mistaken for zero-cost task design.

## Choice learner for the controlled comparison

Fixed and Non-graph use the **same frozen Non-graph checkpoint** for prediction.
No separate Fixed-only fit or tuning bank is used. Thus the default comparison
holds the prediction model, features, loss, training UUIDs, checkpoint selection
and evaluation weights constant. The population MNL that plans Fixed tasks is
not substituted for its choice reader. A different reader would confound task
selection with predictive model changes.

One shared learner trains on random-subset histories from both tracks. Each
UUID/context contributes a prediction spec and a preference spec; sampling is
uniform over those specs within a respondent bootstrap. Prediction histories
range from zero to fourteen responses; preference histories can reach seventeen,
leaving at least one supervised target. The protocol's initial prefix is included
when history length permits. Prediction holdout tasks stay outside prediction
histories, including for training respondents. Full-pool histories belong to the
separate preference spec. Test UUIDs enter neither training nor validation.

The validation bank is fixed across epochs and ensemble members and records
track counts as well as UUID/context coverage. The acquisition rule is predefined;
optimizer updates fit choice cross-entropy only. No policy rollout labels or test
scores select model weights.

This comparison measures selection effects for this particular reader. The optional
`--common-checkpoint` gives a second frozen readout of exactly the collected
histories. It is a readout-specific diagnostic, not model-free information value.
Random and predictive entropy remain opt-in controls.

## Non-graph specification

The network in `neural.py` already matches the supplied Non-graph specification.
The task comprises four alternatives A/B/C/NONE with thirteen attributes each.

| Specified component | Implementation |
|---|---|
| Respondent features and travel context | Nine public features plus context one-hot |
| Full task plus categorical choice in an event | `vec(X_q)` (52), context (3), choice one-hot (4), MLP 59 → 64 → 64 |
| Order-invariant history | Masked mean of event embeddings; zero vector for empty history |
| Respondent state | Mean (64), profile (9), context (3), `log(1+t)` and empty flag; MLP 78 → 64 → 32 |
| Full candidate choice set | Query encoder 52 → 64 → 32 |
| Shared alternative score | State, query encoding, alternative attributes, label one-hot, context; MLP 84 → 64 → 1 |
| Four choice probabilities | Softmax over the four scores, then mean across ensemble members |
| Identifier exclusion | UUID/task IDs never enter model tensors or embeddings |
| Personal updates | Append actual observed task/answer; global neural weights stay frozen |
| Final preference coefficients | Fit the common population MNL to actual collected responses |

The query encoder is a learned representation of the entire choice set rather
than passing its raw 52 values directly to the score head. Context in the event
encoder and an explicit empty flag are additional public inputs. These are
compatible implementation choices; no neural architecture rewrite is required.
History permutation leaves predictions unchanged. Alternative labels are
categorical one-hot inputs, not ordinal choice numbers. Labeled A/B/C airports
can have different utilities; the model does not assume they are interchangeable.

The acquisition remains weighted JS on predicted choices of other remaining
candidate tasks, using answer-probability weights and temporary answer-conditioned
histories. Neither the highest choice probability nor predictive entropy alone
defines the Non-graph selection rule. Internal embeddings are not final MNL
coefficients.

## Public information and hidden answers

`Task`, `Profile`, `Observation`, and `SurveyState` are immutable public inputs. Policies receive a predictor, current public state, remaining candidates, and a seeded RNG. They never receive a `DataBundle` or `ResponseWorld`.

`ResponseWorld.reveal` checks candidate membership and returns one choice from one generator scenario. Only `score_checkpoint` reads held-out answers in prediction episodes. Preference checkpoints carry no held-out predictions and are not scored. Native/common predictions are saved before evaluation, and scores are never fed back to the policy.

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

The fixed policy follows the pool-conditioned precomputed order above. All its budget subsets are prefixes after the common initial tasks.

Weighted JS measures dependence in the model-induced fantasy-answer/future-choice distribution. It excludes q itself and hidden evaluation tasks. Nonnegativity does not establish useful information acquisition; compare validation rollouts with Random and entropy. A no-history predictor produces zero JS and seeded ties.

Primary adaptive evidence is a within-learner comparison against Fixed. Future native four-system comparisons will include learner differences and need separate controlled readouts.

Population MNL analysis uses the same UUID/context population and fixes the reference feature basis for the subset fit. Under MMNL it reports reference agreement only. Rank and optimization failures are explicit; an unidentified parameter is not an estimated zero.

At budget eighteen, both preference trajectories contain the complete same
response panel. Canonical observation order makes each MNL subset fit identical
to its full-survey reference, an exact boundary check rather than a performance
advantage. Population fitting remains offline and does not feed back into the
survey learner or policy.

## Reproducibility

Split manifests bind the UUID partition, protocol, and input file hashes. Trained checkpoints bind their scenario and source split. History protocol checks include seed, holdout count, and initial count; diagnostic budget overrides preserve these fields.

Ensemble members are not independent experiment replicates. Paired confidence intervals resample UUIDs and preserve their context/budget cells. Incomplete grids are rejected. Training seed variability and new response-world variability must be reported separately.

Training uses uniform-cardinality histories with the protocol's initial prefix,
separate prediction/preference specs, and a frozen, seeded validation bank.
Automatic on-policy history augmentation and posterior-latent information gain
remain future extensions. Existing manifests retain their explicitly saved counts;
new defaults do not silently rewrite old splits. Create a new split and train a
matching checkpoint for the four-initial-question protocol. The scripts and
results in `RECHECK.md` describe commit `f5959e1` and its older two-initial-question
protocol; reproduce that record on its named revision.
