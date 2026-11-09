# Robustness & Resilience KPIs for Railway (KPI-DF-069, KPI-FF-070, KPI-SF-071, KPI-SF-072, KPI-VF-073, KPI-AF-074, KPI-DF-075, KPI-RF-076, KPI-SF-077, KPI-RF-078)

## Overview

This module implements **10 KPIs** for evaluating AI agent robustness and resilience in Flatland railway traffic management scenarios. The evaluation uses multiple adversarial attack strategies to stress-test defender agents.

**Author:** INESC TEC
**Test Runner:** `test_runner_robustness_resilience_kpi_069_077.py`
**Local runner:** `run_local_robustness_resilience_kpi_069_077.py`

---

## Implemented KPIs

### Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)

| KPI ID | UUID | Metric | Description |
|--------|------|--------|-------------|
| KPI-DF-069 | `a94c858e-4bc3-4d67-bd78-5c81506e39f7` | `reward_drop_percent` | Percentage decrease in reward [0-100] |
| KPI-FF-070 | `5abadf6b-991c-4d37-810f-f77bb71d490d` | `action_change_freq` | Proportion of timesteps with changed actions [0-1] |
| KPI-SF-071 | `dce32e78-e827-4994-a0a2-06feee2528cc` | `severity_of_change` | Severity of action changes [0-1, higher=worse] |
| KPI-SF-072 | `e5206c56-75a0-41fa-9db3-bec66359337e` | `n_steps_survived` | Number of timesteps before failure |
| KPI-VF-073 | `0ddba8a7-5ef8-45d1-b0d6-0842bc44d2cc` | `perturb_vulnerability` | Proportion of features vulnerable to attack [0-1] |
| KPI-RF-078 | `8ebc88f0-896c-4910-8997-a44d107e7eb7` | `reward_per_action_ratio` | Delivered value per action, as a fraction of the unperturbed baseline [0-1] |

### Resilience KPIs (Benchmark: 31ea606b-681a-437a-85b9-7c81d4ccc287)

| KPI ID | UUID | Metric | Description |
|--------|------|--------|-------------|
| KPI-AF-074 | `707a1a4e-7073-432b-94fc-af4a5ee9f07d` | `area_between_curves` | Integrated performance degradation |
| KPI-DF-075 | `2c4be118-6108-43b3-b09f-a4bee842167a` | `degradation_time` | Time until performance degrades |
| KPI-RF-076 | `2cac54e0-aaf3-4f22-8307-f23878c432f0` | `restoration_time` | Time to restore performance |
| KPI-SF-077 | `d432299f-dbee-46ba-9e15-77954086440a` | `state_similarity` | Cosine similarity to unperturbed states [-1 to 1] |

---

## Attack Strategies

The evaluation runs the defender agent against **6 different attacker types**:

| Attacker | Description | Pre-trained model |
|----------|-------------|-------------------|
| **LambdaPIR** | Lambda policy/value iteration with Bellman-bootstrapped refinement | no, learns online |
| **SAC** | Discrete soft actor-critic adversary | no, learns online |
| **RLPerturb** | Deep Q-learning perturbation agent | no, learns online |
| **Random** | Random perturbation baseline | no |
| **PPO** | Actor-critic adversary | no, learns online |
| **GEPerturb** | Gradient estimation; scores candidates by querying the defender | no |

Unlike the power grid domain, the railway attackers train from scratch during the evaluation, so no `trained_models/` folder is required.

### Measured results

200 episodes against the bundled reference agent (`framework/agent.zip`). Every attacker
is evaluated on the same scenarios (`env.reset(seed=ep)` for `ep` in `0..199`), so the
rows are paired. The unperturbed baseline delivers all three trains in 168 of 200
episodes, mean arrival 0.875, mean 25.6 steps.

These numbers are from the pipeline with all four fixes in place: the attack decision is
made once and recorded as applied, corruption types resolve against the observation the
policy reads, the action space carries a severity axis, and the query-based attacker
scores candidates against that same observation.

All ten KPIs, per attacker:

| Attacker | VF-073 vulnerability | SF-072 steps survived | SF-071 severity | DF-069 reward drop % | FF-070 freq changed | AF-074 area | DF-075 degradation | RF-076 restoration | SF-077 similarity | RF-078 reward/action |
|----------|------|------|------|------|------|------|------|------|------|------|
| GEPerturb | 0.2530 | 29.6 | 0.3333 | 0.85 | **0.4184** | **5390.6** | 3.8 | 1.1 | 0.9441 | 0.4193 |
| LambdaPIR | 0.0249 | 32.0 | 0.3333 | 0.14 | 0.4130 | 3571.6 | 3.1 | 1.0 | 0.9612 | 0.5047 |
| Random | **0.3040** | 30.6 | **0.3634** | 0.12 | 0.3395 | 5148.3 | 3.8 | 1.1 | 0.9450 | 0.4329 |
| RLPerturb | 0.0422 | 29.2 | 0.3333 | 0.22 | 0.2319 | 3634.6 | 2.8 | 1.0 | 0.9641 | 0.6322 |
| PPO | 0.0162 | 28.5 | 0.3333 | 0.02 | 0.2193 | 2015.0 | 1.7 | 0.6 | 0.9745 | 0.8027 |
| SAC | 0.0045 | 25.8 | 0.3333 | 0.00 | 0.0348 | 367.9 | 0.3 | 0.1 | 0.9961 | 0.9400 |

Operational effect of the same runs -- the fraction of trains that actually arrived:

| Attacker | mean arrival | change vs unperturbed | all-3-trains episodes | trains attacked per perturbed step |
|----------|--------------|-----------------------|-----------------------|------------------------------------|
| GEPerturb | **0.417** | **-0.458** | 21/200 | 1.00 |
| Random | 0.467 | -0.408 | 38/200 | 1.15 |
| LambdaPIR | 0.558 | -0.317 | **2/200** | 1.00 |
| RLPerturb | 0.638 | -0.237 | 64/200 | 1.00 |
| PPO | 0.763 | -0.112 | 119/200 | 1.00 |
| SAC | 0.837 | -0.038 | 154/200 | 1.00 |
| *unperturbed* | 0.875 | n/a | 168/200 | n/a |

Aggregated across all six attackers, which is what the orchestrator reports:

| KPI | Value |
|-----|-------|
| KPI-VF-073 Vulnerability to perturbation | 0.1075 |
| KPI-SF-072 Steps survived with perturbations | 29.2933 |
| KPI-SF-071 Severity of changed output | 0.3383 |
| KPI-DF-069 Drop-off in reward | 0.2259 |
| KPI-FF-070 Frequency changed output | 0.2762 |
| KPI-AF-074 Area between reward curves | 3354.6627 |
| KPI-DF-075 Degradation time | 2.5908 |
| KPI-RF-076 Restorative time | 0.8042 |
| KPI-SF-077 Similarity to unperturbed state | 0.9642 |
| KPI-RF-078 Reward per action | 0.6220 |

#### Does any attacker beat the random baseline?

Paired two-sided Wilcoxon signed-rank over the same 200 seeds, against Random. A
stronger attack changes more actions and leaves fewer trains arriving:

| Attacker | actions changed | z | p | arrival | z | p |
|----------|-----------------|---|---|---------|---|---|
| GEPerturb | 0.4110 | +6.40 | **<0.0001** | 0.4167 | -1.87 | 0.062 |
| Random (the bar) | 0.3424 | - | - | 0.4667 | - | - |
| LambdaPIR | 0.3635 | +1.10 | 0.273 | 0.5583 | +1.33 | 0.184 |
| RLPerturb | 0.2071 | -7.58 | <0.0001 | 0.6383 | +4.84 | <0.0001 |
| PPO | 0.1787 | -7.45 | <0.0001 | 0.7633 | +8.28 | <0.0001 |
| SAC | 0.0273 | -11.66 | <0.0001 | 0.8367 | +9.33 | <0.0001 |

**GEPerturb beats the random baseline** on actions changed (p < 0.0001) and leads on
arrival at p = 0.062, which is not significant at 0.05. It is also ahead on the area
between reward curves (5391 vs 5148). **LambdaPIR is now statistically indistinguishable
from Random** on both metrics; before these fixes it was 0.008 against Random's 0.339,
z = -12.26, p = 1.4e-34. On one measure it is the strongest of the six: it leaves all
three trains arriving in only 2 of 200 episodes against Random's 38, so it rarely
prevents the most damage but almost always prevents complete success.

**PPO, SAC and RLPerturb remain significantly weaker than Random**, and the severity
axis did not rescue them. What each attacker chose over the 200 episodes says why:

| Attacker | most-chosen action | share | distinct actions used |
|----------|--------------------|-------|-----------------------|
| GEPerturb | `direction` @ 1000 | 19.4% | 20 |
| LambdaPIR | `target` @ 1000 | 72.4% | 21 |
| RLPerturb | `target` @ 100 | 19.8% | 21 |
| PPO | `malfunction` @ 10 | 52.1% | 14 |
| SAC | `speed` @ 1000 | 88.5% | 11 |
| Random | `blank` @ 1 | 81.7% | 2 |

Every attacker now reaches high severities, so the axis is being used rather than merely
offered. The split is in what they optimise. GEPerturb scores each candidate by asking
the defender what it would actually do, so it finds `direction` at high severity -- which
the field sweep independently ranks among the strongest single-field attacks -- and it is
the one attacker that beats the baseline. The four reinforcement-learning attackers
optimise `disruption_reward()`, which measures movement in three TreeObs fields rather
than damage to the agent, and they converge on whatever that surrogate pays best: SAC on
`speed`, which is provably inert here because `train_speed` holds exactly one distinct
value in a clean rollout, and PPO on `malfunction`, which the surrogate weights 5x.

So the remaining gap is not the action space, which now contains the effective attack,
and not the application path, which now delivers it. It is the learning signal: the
attacker that measures real damage wins, and the attackers that measure a proxy for it
lose. Aligning `disruption_reward()` with the policy's actual response is the next
change, and `act_with_corruption()` already provides the primitive -- but it makes a
query-based attacker out of what is currently a black-box one, which is a change of
threat model rather than a bug fix.

### What this benchmark actually measures

Measured against the reference BC policy, by writing a chosen value into a chosen group
of maze observation fields and recording how often the policy's action changes
(12 episodes per cell):

| field group written | = -1 | = 0 | = 1 | = 1369 |
|---------------------|------|-----|-----|--------|
| `potential_conflict_dist` (3 fields) | 0.000 | 0.000 | 0.000 | **0.252** |
| `target_dist` (12) | 0.017 | 0.016 | 0.016 | **0.161** |
| `train_speed` (1) | 0.000 | 0.000 | 0.000 | **0.095** |
| `other_train_dist` / `conflicting_train_distance` (6) | 0.000 | 0.001 | 0.002 | **0.245** |
| `deadlock` (12) | 0.001 | 0.000 | 0.002 | **0.189** |
| `obstructed` (3) | 0.000 | 0.000 | 0.001 | **0.189** |
| `unusable_switch_dist` (3) | 0.001 | 0.001 | 0.001 | **0.249** |
| *all 86 fields set to 0, which is what the random baseline does* | | **0.249** | | |

1369 is the map's cell count (the scenario is 37x37), used as a value no distance field
could legitimately hold. What the fields do hold, over 468 observations of 12 clean
episodes:

| field | clean min | clean max |
|-------|-----------|-----------|
| `timesteps_left` | 34 | 82 |
| `target_dist` | -1 | 94 |
| `distance` | -1 | 39 |
| `unusable_switch_dist` | -1 | 24 |
| `other_train_dist` | -1 | 2 |
| `potential_conflict_dist` | -1 | -1 (never fires in these scenarios) |
| `train_speed` | 1 | 1 |
| `deadlock`, `obstructed` | -1 | 0 |

So the largest legitimate value anywhere in the vector is 94, and -1 is the
absent/not-applicable sentinel on about 50 of the 86 positions.

For a corruption confined to one field group, magnitude is the axis that matters.
`potential_conflict_dist` never exceeds 20 in a clean rollout:

| value written | 2 | 5 | 20 (its real maximum) | 50 | 100 | 500 | 1369 |
|---------------|---|---|------------------------|----|-----|-----|------|
| action change | 0.000 | 0.001 | 0.005 | 0.018 | 0.042 | 0.176 | 0.252 |

Inside the range the field actually takes, the attack is worth 0.005; it needs 25-70x the
field's legitimate maximum to work. The same holds when only the sentinel positions are
rewritten and the rest of the vector is left intact — a plausible value is absorbed, an
out-of-range one dominates:

| only the -1 positions rewritten to | 0 | 1 | 94 |
|------------------------------------|---|---|----|
| action change | 0.000 | 0.002 | **0.249** |
| arrival | 0.972 | 0.972 | **0.639** |

Magnitude is **not** the axis once the whole vector is overwritten, which is worth stating
because the obvious generalisation from the table above is wrong:

| all 86 fields set to | -1 | 0 | 1 | 94 | 1369 |
|----------------------|----|---|---|----|------|
| action change | 0.249 | 0.249 | 0.189 | 0.189 | 0.189 |
| arrival | 0.639 | 0.639 | 0.694 | 0.694 | 0.694 |

1, 94 and 1369 are identical to six decimals, as are -1 and 0: there are only two
outcomes, not five. With every input equal to one constant the network emits a fixed
logit vector, so the policy collapses to a constant preference order within the legal
action mask — two constants, two orders. That is the policy ignoring its observation
altogether, and 64% of trains still arrive, because the action mask (which the attacker
correctly cannot touch) still forbids illegal moves. -1 is also a legitimate value, so
an in-range attack reaches that point too.

Three consequences:

1. **The random baseline very nearly saturates the channel.** Blanking gives arrival
   0.639. The strongest attack found by hand — `target_dist` or `other_train_dist` set to
   1369 — gives 0.556, against 0.972 unperturbed. The floor is not 0: it is whatever the
   untouched action mask alone delivers. The learned attackers are competing for about
   0.08 of arrival rate, not for the whole range, which is most of why none of them beats
   the baseline.
2. **That remaining 0.08 is reachable only through magnitude, which the attackers cannot
   express.** Their action space is (which train) x (which corruption type) — 19 actions
   for 3 trains, with no magnitude axis. Extending it to (train x type x severity) is the
   change that would make the headroom learnable, and 0.556 vs 0.639 is the number to
   beat.
3. **`train_speed` is unattackable by construction.** It holds exactly one distinct value
   (1.0) across the clean rollouts, because training used `speed_ratio_map={1.0: 1.0}`.
   The policy never saw it vary and cannot have learned to use it.
4. **Under a perturbation budget, the magnitude attacks score zero.** ATM clips at
   xi = 0.1; railway has no budget. A cross-domain robustness claim between the two
   domains is not comparing like with like until railway is budgeted.

### Corruption types must be resolved against the observation the policy reads

`corrupt_node()` honours a corruption type's meaning on a TreeObs, because a TreeObs has
named fields. The policy reads the maze graph observation instead: one flat vector per
train. `corrupt_vector()` was written to be format-agnostic — "corruptions are
statistical" — so on that vector all six types collapsed to three whole-vector
operations:

| type | whole-vector operation | measured action change |
|------|------------------------|------------------------|
| `blank` | x0 | 0.254 |
| `conflict`, `target`, `malfunction` | x3 | **0.000000** |
| `speed`, `direction` | x-1 | 0.263 |

The x3 class changed the policy's action in **zero** of 25 episodes while still recording
a non-zero perturbation, so an attacker that chose one of those three was scored as having
attacked when nothing reached the decision. `conflict`, `target` and `malfunction` were
identical to each other to six decimals, as were `speed` and `direction`: six choices,
three outcomes.

`disruption_reward()` pays for exactly the three inert ones. It rewards movement in
`dist_potential_conflict` (-> `conflict`), `dist_min_to_target` (-> `target`) and
`num_agents_malfunctioning` (-> `malfunction`), and pays nothing for `blank`, `speed` or
`direction`. The payout ordering — `target` ~ 1e5, `malfunction` ~ 5e2, `conflict` ~ 0 —
predicts what each attacker converged on (LambdaPIR: `target` 86%, `conflict` 11%). The
share of an attacker's choices falling in the inert class predicts its measured
effectiveness at Pearson -0.980, which accounts for the whole attacker ranking.

`perturbation_agents/maze_obs_layout.py` now resolves each corruption type to the maze
observation fields it is about, using maze-flatland's own
`GraphDirectionalObservationConversion.convert_to_dict()` key list at runtime. Neither the
vector width nor the field order is hardcoded here, so a change on the maze-flatland side
is picked up rather than silently mismatched. `target` went from 0.000 to 0.118 action
change, and all six types are now distinct.

On its own this did not make the learned attackers win, and it was never going to: it
restores the semantic axis, while the tables above show that for an attack confined to
one field group the effective axis is magnitude. It took the severity axis below, and
giving the query-based attacker a scoring signal it can actually read, before any
attacker beat the baseline. The measured results at the top of this file are from the
pipeline with all of those in place; see there for where each attacker ended up.

### The attackers can now choose how extreme the lie is

An attack action was (which train) x (which corruption type): 18 actions for three
trains. The field sweep above shows that space does not contain the attack that works.
A corruption confined to the fields its type is about does almost nothing while the
value it writes stays inside the range the simulator produces, and only becomes
effective well outside it, so the one effective dimension was not expressible and every
attacker was searching without it.

`perturbation_agents/attack_space.py` adds severity as a third factor, and is the single
place the action space is enumerated -- the five attackers each built the product
themselves, which is how they could drift apart. An action is now
`(handle, corruption_type, severity)`:

| | two-factor | three-factor |
|-|------------|--------------|
| actions, 3 trains | 18 | 63 |
| severities per type | 1 (the plausible value) | 4 (1, 10, 100, 1000) |

Severity scales the deviation from the clean value,
`written = clean + severity * (plausible - clean)`, so **severity 1.0 reproduces exactly
the old action space** and every earlier result is the severity-1.0 slice of this one.
Higher severities push the same claim past what the simulator could produce, up to the
observation space's own declared bound -- read from the env at runtime
(`Box(low=-1, high=prod(map_size))`, so 1369 on this 37x37 scenario) rather than written
down here, so a differently sized scenario is bounded correctly without a code change.
`blank` takes no severity: it is defined on the whole vector, and duplicate arms would
only cost a learner samples.

Measured against the reference BC policy, one train attacked per step to match the
1.00-1.15 trains per perturbed step the real attackers were measured at (12 episodes per
cell; arrival 0.972 unperturbed, 0.639 for the blank attack the random baseline plays):

| corruption type | severity 1 | 10 | 100 | 1000 |
|-----------------|-----------|----|-----|------|
| `conflict` | 0.972 | 0.972 | 0.722 | 0.639 |
| `target` | 0.750 | **0.611** | **0.611** | **0.611** |
| `speed` | 0.972 | 0.972 | 0.972 | 0.972 |
| `direction` | 0.972 | 0.944 | 0.639 | 0.639 |
| `malfunction` | 0.972 | 0.722 | 0.694 | **0.611** |

(arrival rate; lower is a stronger attack.)

Three things to read from it. The severity-1 column is everything the old action space
could play, and one cell of it does anything at all. Severity is monotone within every
type, so it is a dimension a learner can follow rather than a lottery. And `train_speed`
is flat at every severity, consistent with it holding exactly one distinct value in a
clean rollout: it is unattackable by construction, not under-attacked.

At 100 episodes the margin over the baseline replicates and grows, but the paired
significance test has not been run:

| | arrival | action change |
|-|---------|---------------|
| unperturbed | 0.9033 | 0.0000 |
| `blank` (the random baseline's attack) | 0.6300 | 0.2606 |
| `target` @ severity 10 | **0.5933** | 0.1256 |

(100 episodes, one train attacked per step. A partial run: it covered 2 of 5 planned
configurations before being stopped, and recorded only the means, so there is no p-value
behind the 0.037 difference. The full per-attacker comparison at the top of this file
supersedes it.)

Every attacker does reach the high severities once the axis exists -- the choice table in
the measured results shows the most-chosen action is at severity 100 or 1000 for five of
the six -- so the axis is used rather than merely offered.

One caveat on the baseline: `RandomPerturbAttacker` still plays only the plausible value,
deliberately, so the bar is the same one the earlier tables used. It is therefore not a
matched-space control. The claim it supports is "a learned attacker beats the blanking
baseline", not "beats a random search over the same action space"; establishing the
latter needs a baseline that samples severity too.

### A perturbation budget, so the number is comparable across domains

The power-grid domain bounds its attacker -- `SACAdversarialContinuous` clips the
perturbation to +/- epsilon, 0.1 by default -- and railway did not, so a railway
robustness number was not comparable with a bounded one, and the attacks that work here
are ones no bounded attacker could play. `MazeFlatlandDefender(..., xi=...)` applies a
budget, and `utils.clip_to_budget` enforces it on the delta, with the observation rebuilt
from the clipped delta so the perturbation recorded is the one the policy received.

The bound is relative, not absolute, because these observations are not normalised:
fields run from a -1 "absent" sentinel to distances in the tens, so a single absolute
epsilon would be most of a flag's range and a rounding error on a distance. Each element
may move by `xi` times its own clean magnitude, with a floor of 1.0 so a field reading
exactly 0 can still be perturbed by `xi`. `xi=None` is the default and leaves the
attacker unbounded, which is how every railway result in this file was produced; no
bounded railway KPI has been measured yet.

### When a robustness score is not a robustness score

Only agents that derive their actions from the observation can be scored meaningfully.
Every attacker perturbs the observation, so a policy reading environment state directly
(`ShortestPathDefender`) is unaffected by construction.

An unchanged action is evidence of robustness **only if the attack was actually applied
to the observation the policy read**. Two cases where it was not:

- **The perturbation never reached the policy.** `MazeFlatlandDefender`'s BC policy reads
  the maze observation (`GraphDirectionalObservationConversion` + `train_move_mask`, built
  from `FlatlandMazeState`), not the TreeObs the framework perturbs. In `obs_perturb` mode
  the attacker's chosen corruption is re-applied to that maze observation, so the attack
  does reach the policy and the KPI is valid. Where it cannot reach the policy at all, the
  applied perturbation is identically zero; `result_getter` records this per attacker as
  `attack_reached_policy=False` and prints a warning. **Report those KPIs as NOT
  APPLICABLE, not as high robustness.**
- **The attacker chose blind.** A query-based attacker such as GEPerturb scores candidate
  perturbations through `defender.act()`, which answers from the maze observation and
  ignores the TreeObs it is handed — so every candidate scores identically and its
  selection carries no gradient information. The attack it publishes *is* applied, but it
  was not informed. That is a limitation of the attacker against this defender, not
  robustness of the agent.

### Attack decision vs. attack application

The attacker decides once per environment step, in `perturb()`, and publishes the decision
as `attacker.last_targets = {handle: corruption_type}`. `MazeFlatlandDefender.act_and_step()`
only *applies* that decision; it never calls `perturb()` again and never re-chooses. The
perturbation reported for a step is the delta actually applied to the policy's own
observation, returned by `act_and_step()`, not the TreeObs delta.

An attacker that publishes `{}` corrupted nothing that step and nothing is applied. An
attacker that publishes nothing at all (`None`) leaves the defender no choice but to
corrupt every train, which is logged as a warning because it is not breadth-matched
against targeted attackers.

---

## Configuration

Default settings in `MultiAttackerRobustnessTestRunner`:

```python
ATTACKER_TYPES    = ["GEPerturb", "LambdaPIR", "Random", "PPO", "SAC", "RLPerturb"]
NUM_EPISODES      = 50
BASELINE_DEFENDER = None      # no default; an evaluation without an agent fails loudly

# Flatland environment (matches the reference agent's training setup)
N_TRAINS   = 3
MAP_WIDTH  = 37
MAP_HEIGHT = 37
N_CITIES   = 2
TREE_DEPTH = 2
```

---

## Framework Structure

```
railway/
├── __init__.py
├── orchestrator.py                                   # Celery orchestrator
├── orchestrator_definitions.py                       # KPI UUID -> TestRunner wiring
├── test_runner_robustness_resilience_kpi_069_077.py  # These KPIs implementation
├── run_local_robustness_resilience_kpi_069_077.py   # Local runner (no FAB)
├── README_robustness_resilience_kpi_069_077.md       # This documentation
└── framework/
    ├── agent.zip                                     # Reference agent (see below)
    ├── Environment.py                                # FlatlandEnvironment wrapper
    ├── defenders.py                                  # Defender wrappers
    ├── defenders_maze.py                             # maze-flatland policy adapter
    ├── example_evaluation.py                         # Minimal usage example
    ├── attack_models/                                # Attacker implementations
    │   ├── BaseAgent.py
    │   ├── BaseAttackerClass.py
    │   ├── GEPerturbAttacker.py
    │   ├── LambdaPIRAttacker.py
    │   ├── PPOAttacker.py
    │   ├── RandomPerturbAttacker.py
    │   ├── RLPerturbAttacker.py
    │   └── SACAttacker.py
    ├── perturbation_agents/                          # Perturbation strategies
    ├── evaluation_framework/                         # Metrics computation
    │   ├── metrics.py
    │   ├── plots_and_tables.py
    │   └── result_getter.py
    └── utility/                                      # HTML report helpers
```

---

## Submission Format

Agents are wrapped in a `RailwayDefender` (see `framework/defenders.py`):

| Class | Wraps |
|-------|-------|
| `MazeFlatlandDefender` | a maze-flatland policy; drives its own maze environment |
| `ObsPolicyDefender` | any policy exposing `act_many(handles, obs_list) -> {handle: action}` |
| `ShortestPathDefender` | Flatland's `ShortestPathPolicy` (reads the env, not the observation) |
| `DoNothingDefender` | baseline that takes no action |

Two submission formats are accepted:

1. **Pickle file** (`.pkl`) containing either a `RailwayDefender` instance, used as-is, or a policy object exposing `act_many(handles, obs_list)`, which is wrapped in an `ObsPolicyDefender`.

2. **ZIP file** (`.zip`) containing either
   ```
   state_dict-epoch_N.pt        # maze-flatland checkpoint
   spaces_config.pkl
   ```
   which is wrapped in a `MazeFlatlandDefender`, or
   ```
   agent.pkl                    # or policy.pkl
   ```
   which is treated as case 1.

**Docker submissions are not supported for these KPIs.** The other railway KPIs (`KPI-PF-026`, `KPI-NF-045`) drive containerised policies via `flatland-trajectory-generate-from-policy`, which replays a finished trajectory. These KPIs perturb the agent's observation at every step and therefore require in-process inference.

### Reference agent (`framework/agent.zip`)

A trained maze-flatland behavioural-cloning policy ships with the framework, following the same convention as `power_grid/framework/agent.zip`:

```
state_dict-epoch_200.pt    trained policy weights
spaces_config.pkl          observation/action space config saved with the run
training_config.yaml       the run's hydra config
LICENSE.maze-flatland      MIT licence, (c) 2025 EnliteAI GmbH
```

`build_maze_env()` reproduces the environment this policy was trained in, which `training_config.yaml` documents: 3 trains, 37x37, 2 cities, malfunction rate 0, `DirectionalAC`, `GraphDirectionalObservationConversion`, `FlatlandMaskingWrapper` with `LogicMaskBuilder(mask_out_dead_ends=True, disable_stop_on_switches=False)`, and `SubStepSkippingWrapper(do_skipping_in_reset=True)`.

`MazeFlatlandDefender` sets `drives_env = True`: the maze environment internally advances a `RailEnv`, so `FlatlandEnvironment` drives each transition through `act_and_step()` and reads tree observations from that same `RailEnv`. The environment is seeded per episode so that every submission is scored on the same scenarios.

---

## Usage

### Local

```bash
cd ai4realnet_orchestrators/railway

# The bundled reference agent, 5 episodes, report KPI-VF-073
python run_local_robustness_resilience_kpi_069_077.py --agent framework/agent.zip

# Any KPI, more episodes, print all 10 values from the single run
python run_local_robustness_resilience_kpi_069_077.py --kpi KPI-FF-070 --episodes 50 --all

# Your own submission
python run_local_robustness_resilience_kpi_069_077.py --agent path/to/agent.pkl --all
```

A 50-episode run against the reference agent takes roughly 5 minutes, followed by the per-episode plots (`episodes x attackers x 2` SVG files), which dominate the wall time.

### Via the orchestrator

The orchestrator routes submissions to the appropriate KPI test automatically:

```python
# Example: Evaluating KPI-VF-073 (Vulnerability)
result = run_scenario(
    scenario_id="47a11418-fad5-4d55-a637-6b90a8351500",
    submission_id="your-submission-uuid"
)
# Returns: {"primary": 0.08}  # 8% vulnerability
```

---

## Metrics Interpretation

| Metric | Good Value | Bad Value | Interpretation |
|--------|------------|-----------|----------------|
| `perturb_vulnerability` | Low (0.0) | High (1.0) | Lower = more robust |
| `n_steps_survived` | High | Low (0) | Higher = more robust |
| `severity_of_change` | Low (0.0) | High (1.0) | Lower = more stable |
| `reward_drop_percent` | Low (0%) | High (100%) | Lower = maintains performance |
| `action_change_freq` | Low (0.0) | High (1.0) | Lower = more consistent |
| `area_between_curves` | Low | High | Lower = faster recovery |
| `degradation_time` | High | Low | Higher = slower degradation |
| `restoration_time` | Low | High | Lower = faster recovery |
| `state_similarity` | High (1.0) | Low (-1.0) | Higher = maintains state |
| `reward_per_action_ratio` | High (1.0) | Low (0.0) | Higher = value per intervention preserved |

---

## Railway-specific metric definitions

Three metrics are defined differently from the power grid implementation, because Flatland's reward is sparse and non-positive: it is `0.0` at every step with a single terminal penalty, and exactly `0` for an agent that delivers every train. Each divergence is marked `DIVERGENCE FROM POWER GRID` in the source.

**`reward_drop_percent` (KPI-DF-069)** is normalised by the worst reward reachable over the same number of steps, `N_TRAINS * total_steps`, rather than by the unperturbed total, which is `0` for a successful agent.

**`severity_of_change` (KPI-SF-071)** is divided by the number of changed actions. `metrics.similarity_score` accumulates a sum over changed steps rather than a mean, so the unnormalised value leaves the declared `[0, 1]` range once an attacker changes more than one action.

**The resilience KPIs (AF-074, DF-075, RF-076) and KPI-RF-078** are computed from the fraction of trains that have arrived, recorded per step by `FlatlandEnvironment`, rather than from the reward. The arrivals curve varies through the episode and falls behind the baseline when an attack delays a train, whereas a sparse terminal reward carries no such shape. `get_resilience_metrics_performance_single_ep()` measures the gap between the baseline and perturbed arrivals curves directly: degradation time runs from the gap opening to its widest point, restoration time from that point until it closes (censored at the end of the episode), and the area is the integral of the gap.

`reward_per_action_ratio` is capped at 1.0 for scoring: an attack that suppresses the agent's interventions without reducing trains delivered would otherwise report a value above 1. The uncapped value remains available on `metrics.reward_per_action["reward_per_action_ratio"]`.

Pickles written before the performance signal existed fall back to the reward-curve behaviour, so earlier results still load.

---

## Contact

For questions about this KPI implementation, contact INESC TEC.
