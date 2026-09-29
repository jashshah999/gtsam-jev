"""Asia network: Jev fills the CPTs from descriptions, GTSAM answers evidence queries.

Runs offline from asia_recording.json. With --live (and TYPESAFE_API_KEY set) it
asks Jev again and overwrites the recording.
"""
import sys
from pathlib import Path

import gtsam

import gtsam_jev as gj

VARIABLES = {
    "A": ("the patient recently visited Asia", ()),
    "S": ("the patient is a smoker", ()),
    "T": ("the patient has tuberculosis", ("A",)),
    "L": ("the patient has lung cancer", ("S",)),
    "B": ("the patient has bronchitis", ("S",)),
    "E": ("the patient has tuberculosis or lung cancer", ("T", "L")),
    "X": ("the patient's chest X-ray is abnormal", ("E",)),
    "D": ("the patient has shortness of breath (dyspnea)", ("E", "B")),
}
STATE = {
    "task": "Parameterize a small educational Bayesian network about respiratory disease.",
    "population": "Imagine a randomly selected adult patient in a general clinical population. "
    "Use ordinary real-world medical knowledge, not the memorized textbook Asia-network numbers.",
}
RECORDING = Path(__file__).with_name("asia_recording.json")

if "--live" in sys.argv:
    from typesafe_sdk import TypeSafeClient

    RECORDING.unlink(missing_ok=True)
    with TypeSafeClient() as api:
        net, keys = gj.bayes_net(VARIABLES, gj.RecordingClient(api, RECORDING), state=STATE)
else:
    net, keys = gj.bayes_net(VARIABLES, gj.ReplayClient(RECORDING), state=STATE)


def posterior(evidence):
    graph = gtsam.DiscreteFactorGraph(net)
    for name, value in evidence.items():
        graph.add(keys[name], "0 1" if value else "1 0")
    marginals = gtsam.DiscreteMarginals(graph)
    return [marginals.marginalProbabilities(keys[n])[1] for n in ("T", "L", "B")]


print(f"{'evidence':32} {'tuberculosis':>13} {'lung cancer':>12} {'bronchitis':>11}")
for label, evidence in [("none", {}),
                        ("no Asia visit", {"A": 0}),
                        ("+ abnormal X-ray", {"A": 0, "X": 1}),
                        ("+ shortness of breath", {"A": 0, "X": 1, "D": 1})]:
    t, l, b = posterior(evidence)
    print(f"{label:32} {t:13.1%} {l:12.1%} {b:11.1%}")
