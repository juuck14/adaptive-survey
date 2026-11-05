# Project development

- Treat UUID as a join and split key, never a model feature.
- All contexts and generator scenarios of one UUID belong to one split.
- Policies receive public profiles, candidate task attributes, and revealed choices only.
- Keep holdout choices and the other generator scenario out of policy state and training episodes.
- Counterfactual updates must not mutate the real episode or another respondent's state.
- Freeze learner parameters during evaluation. Offline population estimation must not feed back into a rollout.
- Test GNN adapters with every focal respondent response edge outside the revealed history masked before message passing, including label-derived caches.
- Report weighted JS as a predictive-dependence proxy, not exact Bayesian information gain.
- Never commit respondent data, generator coefficients, trained checkpoints, or signed download URLs.
- Run the tests and a small command-line smoke run before changing evaluation semantics.
