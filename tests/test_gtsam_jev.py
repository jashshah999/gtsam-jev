import os
import re

import gtsam
import pytest

import gtsam_jev as gj


class FakeClient:
    """Answers each Choice from a function of its instructions; counts requests."""

    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    def system_one(self, *, model, state, questions):
        self.requests.append(list(questions))
        probabilities = {qid: self.answer(q.instructions, list(q.criteria)) for qid, q in questions.items()}
        return gj._Response(probabilities)


# Frank Dellaert's recorded jev-1.13.0 CPTs for the Asia network (P(false), P(true)),
# from https://gist.github.com/dellaert/ed9c8ed6bbfa22a4f027474b9c3e32b5
ASIA = {
    "A": ("the patient recently visited Asia", ()),
    "S": ("the patient is a smoker", ()),
    "T": ("the patient has tuberculosis", ("A",)),
    "L": ("the patient has lung cancer", ("S",)),
    "B": ("the patient has bronchitis", ("S",)),
    "E": ("the patient has tuberculosis or lung cancer", ("T", "L")),
    "X": ("the patient's chest X-ray is abnormal", ("E",)),
    "D": ("the patient has shortness of breath (dyspnea)", ("E", "B")),
}
ASIA_CPTS = [[0.96, 0.04], [0.8, 0.2], [0.97, 0.03], [0.68, 0.32], [0.99, 0.01], [0.92, 0.08],
             [0.95, 0.05], [0.31, 0.69], [0.99, 0.01], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
             [0.89, 0.11], [0.01, 0.99], [0.93, 0.07], [0.15, 0.85], [0.05, 0.95], [0.03, 0.97]]


def asia_client():
    rows = iter(ASIA_CPTS)
    return FakeClient(lambda text, labels: dict(zip(labels, next(rows))))


def test_floored_clips_and_renormalizes():
    assert gj.floored([0.0, 1.0], 0.05) == pytest.approx([0.05, 0.95])
    assert gj.floored([0.0, 0.0, 1.0], 0.1) == pytest.approx([0.1 / 1.1, 0.1 / 1.1, 0.9 / 1.1])
    assert sum(gj.floored([0.2, 0.3, 0.5], 0.0)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        gj.floored([0.5, 0.5], 0.5)


def test_choice_factor_values():
    factor = gj.choice_factor((7, 3), [0.0, 0.2, 0.8], floor=0.05)
    values = []
    for v in range(3):
        assignment = gtsam.DiscreteValues()
        assignment[7] = v
        values.append(factor(assignment))
    assert values == pytest.approx(gj.floored([0.0, 0.2, 0.8], 0.05))
    with pytest.raises(ValueError):
        gj.choice_factor((7, 2), [0.2, 0.3, 0.5])


def test_ask_batches_and_answers_every_question():
    client = FakeClient(lambda text, labels: {labels[0]: 0.25, labels[1]: 0.75})
    questions = {f"q{i}": (f"question {i}", {"no": "No.", "yes": "Yes."}) for i in range(250)}
    answers = gj.ask(client, {}, questions, batch_size=100)
    assert [len(r) for r in client.requests] == [100, 100, 50]
    assert set(answers) == set(questions)
    assert answers["q249"] == {"no": 0.25, "yes": 0.75}


def test_mode_factors_order_dedup_and_cardinality():
    def answer(text, labels):
        note = re.search(r'"(.*)"', text).group(1)
        return {"normal": 0.1, "slippery": 0.0, "stuck": 0.9} if "mud" in note else \
               {"normal": 0.8, "slippery": 0.15, "stuck": 0.05}

    client = FakeClient(answer)
    notes = ["deep mud", "dry corridor", "deep mud"]
    keys = [(k, 3) for k in range(3)]
    options = {"normal": "Normal.", "slippery": "Slippery.", "stuck": "Stuck."}
    factors = gj.mode_factors(notes, keys, options, client, floor=0.05)
    assert [len(r) for r in client.requests] == [2]  # "deep mud" asked once
    stuck = gtsam.DiscreteValues()
    stuck[0] = 2
    assert factors[0](stuck) == pytest.approx(gj.floored([0.1, 0.0, 0.9], 0.05)[2])
    with pytest.raises(ValueError):
        gj.mode_factors(notes, [(k, 2) for k in range(3)], options, client)


def test_bayes_net_matches_hand_built_asia():
    net, keys = gj.bayes_net(ASIA, asia_client(), floor=0.0)

    expected = gtsam.DiscreteBayesNet()
    rows = iter(ASIA_CPTS)
    for name, (_, parents) in ASIA.items():
        table = " ".join(f"{p0:.12g}/{p1:.12g}" for p0, p1 in
                         (next(rows) for _ in range(2 ** len(parents))))
        if parents:
            parent_keys = gtsam.DiscreteKeys()
            for parent in parents:
                parent_keys.push_back(keys[parent])
            expected.add(keys[name], parent_keys, table)
        else:
            expected.add(keys[name], table)

    def posterior(bn, evidence):
        graph = gtsam.DiscreteFactorGraph(bn)
        for name, value in evidence.items():
            graph.add(keys[name], "0 1" if value else "1 0")
        marginals = gtsam.DiscreteMarginals(graph)
        return [marginals.marginalProbabilities(keys[n])[1] for n in ("T", "L", "B")]

    for evidence in ({}, {"A": 0}, {"A": 0, "X": 1}, {"A": 0, "X": 1, "D": 1}):
        assert posterior(net, evidence) == pytest.approx(posterior(expected, evidence), abs=1e-12)


def test_bayes_net_rejects_parent_after_child():
    with pytest.raises(ValueError):
        gj.bayes_net({"B": ("b", ("A",)), "A": ("a", ())}, asia_client())


def test_recording_then_replay(tmp_path):
    path = tmp_path / "jev.json"
    recorder = gj.RecordingClient(asia_client(), path)
    net, keys = gj.bayes_net(ASIA, recorder)
    replayed, _ = gj.bayes_net(ASIA, gj.ReplayClient(path))
    assignment = gtsam.DiscreteValues()
    for name in ASIA:
        assignment[keys[name][0]] = 1
    assert replayed.evaluate(assignment) == pytest.approx(net.evaluate(assignment))
    with pytest.raises(KeyError):
        gj.bayes_net(ASIA, gj.ReplayClient(path), state={"different": "state"})


@pytest.mark.skipif(not os.getenv("TYPESAFE_API_KEY"), reason="needs TYPESAFE_API_KEY")
def test_live_mode_factors():
    from typesafe_sdk import TypeSafeClient

    options = {"normal": "Traction is normal.", "slippery": "The ground is slippery or loose."}
    with TypeSafeClient() as client:
        factors = gj.mode_factors(["Floor was mopped a minute ago and is still wet.",
                                   "Dry carpeted office corridor."],
                                  [(0, 2), (1, 2)], options, client)
    slippery = [gtsam.DiscreteValues() for _ in range(2)]
    slippery[0][0] = 1
    slippery[1][1] = 1
    assert factors[0](slippery[0]) > 0.5 > factors[1](slippery[1])
