# gtsam-jev

Discrete factors for [GTSAM](https://github.com/borglab/gtsam) from [Jev](https://docs.typesafe.ai),
TypeSafe's calibrated decision model.

Jev answers typed questions with probabilities. This package turns those answers into GTSAM factors, so a
factor graph can use common-sense or text evidence next to its measurements:

* **`mode_factors`**: one unary `DecisionTreeFactor` per discrete variable, from a text description of
  each. For example, a log note per odometry step becomes evidence on that step's motion mode.
* **`bayes_net`**: a `DiscreteBayesNet` whose conditional probability tables are elicited in one request
  from plain-language variable descriptions. This packages Frank Dellaert's
  [Jev + GTSAM experiment](https://gist.github.com/dellaert/ed9c8ed6bbfa22a4f027474b9c3e32b5).
* **`RecordingClient` / `ReplayClient`**: record the answers once and replay them offline, so results
  are reproducible and run without an API key.

```bash
pip install git+https://github.com/jashshah999/gtsam-jev
```

## Example: evidence on hybrid modes

```python
import gtsam_jev as gj
from typesafe_sdk import TypeSafeClient

options = {  # value 0, value 1 of each mode
    "normal": "Traction is normal, so wheel odometry is reliable.",
    "slippery": "The ground is slippery or loose, so the wheels may slip.",
}
notes = ["Sprinklers ran on the lawn crossing at dawn.", "Carpeted corridor B, quiet."]
with TypeSafeClient() as client:
    factors = gj.mode_factors(notes, [(M(1), 2), (M(2), 2)], options, client, floor=0.2)
for factor in factors:
    hybrid_graph.push_back(factor)
```

GTSAM's [SemanticModeEvidence](https://github.com/borglab/gtsam/pull/2832) example does this for a robot
whose odometry steps are normal or slippery. Over 40 simulated routes of 40 steps, evidence from the notes
took mode accuracy from 0.58 (geometry only) and 0.66 (a keyword list) to 0.94, and position RMSE from
0.47 m and 0.45 m to 0.33 m. The notes there are synthetic.

## Example: a Bayes net from descriptions

`examples/asia.py` builds the classic Asia network from eight one-line descriptions and answers evidence
queries with GTSAM. It replays `examples/asia_recording.json`; `--live` asks Jev again.

```
evidence                          tuberculosis  lung cancer  bronchitis
none                                      6.3%         7.3%       26.1%
no Asia visit                             5.0%         7.3%       26.1%
+ abnormal X-ray                         17.5%        25.7%       29.1%
+ shortness of breath                    24.8%        36.4%       40.1%
```

The elicited rows are within 0.09 of the ones in Frank Dellaert's recording.

## Things that matter

* **Floor the probabilities.** Jev often answers exactly 0 or 1, and a factor value of 0 makes a value
  impossible, so no measurement can ever overrule it. Every factor here is clipped to
  `[floor, 1 - floor]` and renormalized. In the hybrid example, raising the floor from 0.05 to 0.2 to 0.3
  let the geometry recover most of the slippery steps whose notes said nothing useful.
* **Ask it to compare described options, not to estimate frequencies.** Rates, noise levels and
  transition probabilities come back near 0.5 whatever the truth is. Take those from data.
* **Wording matters.** Asking "is it true that the patient has lung cancer?" under a smoker condition
  returned 0.00. Asking which truth value best describes it, with the state asking for a probability
  distribution (`CPT_INTERPRETATION`), returned 0.13. In the hybrid example, a shorter task description
  moved "Grease trap inspection scheduled for next week" from 0.14 to 0.58 slippery. An identical request
  repeated changed probabilities by at most 0.02.

## Tests

```bash
pip install -e ".[test]"
pytest            # offline; the live test runs when TYPESAFE_API_KEY is set
```
