"""Discrete factors for GTSAM from TypeSafe's Jev.

Jev answers typed questions with calibrated probabilities. This package turns
those answers into GTSAM factors:

* ``mode_factors``: one unary ``DecisionTreeFactor`` per discrete variable, from
  a text description of each (e.g. a log note per odometry step).
* ``bayes_net``: a ``DiscreteBayesNet`` whose conditional probability tables are
  elicited in one request from plain-language variable descriptions.

Probabilities are clipped to ``[floor, 1 - floor]`` and renormalized before they
become factors: Jev often answers with exact 0 or 1, and a zero factor makes a
value impossible, so no other evidence could ever overrule it.

Ask Jev to *compare* described options. Frequencies it would have to estimate
(noise levels, how often something happens) come back near 0.5 regardless of
the truth; learn those from data instead.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import gtsam

__all__ = [
    "floored",
    "choice_factor",
    "ask",
    "mode_factors",
    "bayes_net",
    "RecordingClient",
    "ReplayClient",
]

DEFAULT_MODEL = "jev-1.13.0"

# How ``bayes_net`` asks for a CPT row. Asking "is it true that ...?" gets a strict
# yes/no judgement (P(lung cancer | smoker) came back 0.00); asking for the truth
# value that best describes it, under an explicit distribution instruction, gets
# graded probabilities. Wording from Frank Dellaert's Jev + GTSAM experiment.
CPT_INTERPRETATION = (
    "For each question, return a probability distribution for the child variable under exactly "
    "the stated parent assignment. Treat unspecified variables as unknown; do not assume them false."
)

# A question is (instructions, {label: criterion}); answers are {label: probability}.
Question = tuple[str, Mapping[str, str]]
Answer = dict[str, float]


def floored(probabilities: Sequence[float], floor: float = 0.05) -> list[float]:
    """Clip to [floor, 1 - floor] and renormalize so the values sum to one."""
    if not 0 <= floor < 1 / max(len(probabilities), 1):
        raise ValueError(f"floor {floor} must be in [0, 1/{len(probabilities)})")
    clipped = [min(max(p, floor), 1 - floor) for p in probabilities]
    total = sum(clipped)
    return [p / total for p in clipped]


def choice_factor(key: tuple[int, int], probabilities: Sequence[float],
                  floor: float = 0.05) -> gtsam.DecisionTreeFactor:
    """Unary factor on a discrete key from probabilities over its values."""
    if len(probabilities) != key[1]:
        raise ValueError(f"{len(probabilities)} probabilities for cardinality {key[1]}")
    return gtsam.DecisionTreeFactor(key, " ".join(f"{p:.12g}" for p in floored(probabilities, floor)))


def ask(client, state: Mapping, questions: Mapping[str, Question], *,
        model: str = DEFAULT_MODEL, batch_size: int = 100) -> dict[str, Answer]:
    """Ask Choice questions in batches of ``batch_size``; return probabilities per question id.

    ``client`` is a ``typesafe_sdk.TypeSafeClient`` (whose retry policy handles
    rate limits) or anything with the same ``system_one`` method, such as
    ``ReplayClient``.
    """
    from typesafe_sdk import Choice

    ids = list(questions)
    answers: dict[str, Answer] = {}
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        response = client.system_one(
            model=model, state=dict(state),
            questions={qid: Choice(instructions=questions[qid][0], criteria=dict(questions[qid][1]))
                       for qid in batch},
        )
        for qid in batch:
            answers[qid] = dict(response.answers[qid].probabilities)
    return answers


def mode_factors(descriptions: Sequence[str], keys: Sequence[tuple[int, int]],
                 options: Mapping[str, str], client, *,
                 state: Mapping | None = None,
                 instructions: Callable[[str], str] = lambda d: f'Description: "{d}" Which option applies?',
                 floor: float = 0.05, model: str = DEFAULT_MODEL,
                 batch_size: int = 100) -> list[gtsam.DecisionTreeFactor]:
    """One unary factor per key from a text description of that variable.

    ``options`` maps each label to its criterion, in the order of the key's values
    (the first label is value 0). Identical descriptions are asked once.
    """
    if len(descriptions) != len(keys):
        raise ValueError("need one description per key")
    labels = list(options)
    for key in keys:
        if key[1] != len(labels):
            raise ValueError(f"key {key} has cardinality {key[1]}, but there are {len(labels)} options")
    unique = list(dict.fromkeys(descriptions))
    answers = ask(client, state or {}, {f"d{i}": (instructions(d), options) for i, d in enumerate(unique)},
                  model=model, batch_size=batch_size)
    by_description = {d: answers[f"d{i}"] for i, d in enumerate(unique)}
    return [choice_factor(key, [by_description[d][label] for label in labels], floor)
            for d, key in zip(descriptions, keys)]


def bayes_net(variables: Mapping[str, tuple[str, Sequence[str]]], client, *,
              state: Mapping | None = None, floor: float = 0.05,
              model: str = DEFAULT_MODEL, batch_size: int = 100,
              ) -> tuple[gtsam.DiscreteBayesNet, dict[str, tuple[int, int]]]:
    """Binary Bayes net with CPTs elicited from Jev, one row per parent assignment.

    ``variables`` maps a name to ``(meaning, parents)``, where ``meaning`` is a
    statement such as "the patient is a smoker" and parents come earlier in the
    mapping. Value 1 means the statement is true. ``state`` should describe the
    population the probabilities are about; ``CPT_INTERPRETATION`` is added to it
    unless it already has an ``interpretation`` entry. Returns the net and the
    ``(key, 2)`` used for each name.
    """
    state = {"interpretation": CPT_INTERPRETATION, **(state or {})}
    names = list(variables)
    keys = {name: (i, 2) for i, name in enumerate(names)}
    for name, (_, parents) in variables.items():
        for parent in parents:
            if names.index(parent) >= names.index(name):
                raise ValueError(f"parent {parent} of {name} must come before it")
    rows, questions = [], {}
    for name, (meaning, parents) in variables.items():
        for values in itertools.product((0, 1), repeat=len(parents)):
            condition = " ".join(
                f"It is {'true' if v else 'false'} that {variables[p][0]}." for p, v in zip(parents, values)
            ) or "No other facts are given."
            qid = f"q{len(rows)}"
            questions[qid] = (
                f"Given the stated condition, which truth value best describes whether {meaning}? "
                f"Condition: {condition}",
                {"false": f"It is false that {meaning}.", "true": f"It is true that {meaning}."},
            )
            rows.append((name, qid))
    answers = ask(client, state, questions, model=model, batch_size=batch_size)

    net = gtsam.DiscreteBayesNet()
    for name, (_, parents) in variables.items():
        table = " ".join(
            "/".join(f"{p:.12g}" for p in floored([answers[qid]["false"], answers[qid]["true"]], floor))
            for row_name, qid in rows if row_name == name
        )
        if parents:
            parent_keys = gtsam.DiscreteKeys()
            for parent in parents:
                parent_keys.push_back(keys[parent])
            net.add(keys[name], parent_keys, table)
        else:
            net.add(keys[name], table)
    return net, keys


class RecordingClient:
    """Wraps a client and appends every request and answer to a JSON file."""

    def __init__(self, client, path: str | Path):
        self._client, self._path = client, Path(path)
        self._records = json.loads(self._path.read_text()) if self._path.exists() else []

    def system_one(self, *, model, state, questions):
        response = self._client.system_one(model=model, state=state, questions=questions)
        self._records.append({
            "model": model,
            "state": state,
            "questions": {qid: q.model_dump(mode="json") for qid, q in questions.items()},
            "probabilities": {qid: dict(response.answers[qid].probabilities) for qid in questions},
        })
        self._path.write_text(json.dumps(self._records, indent=1))
        return response


class ReplayClient:
    """Answers from a file written by ``RecordingClient``; no API key needed.

    A request must match a recorded one exactly (model, state and questions),
    so a replay can never silently answer a different question.
    """

    def __init__(self, path: str | Path):
        self._records = json.loads(Path(path).read_text())

    def system_one(self, *, model, state, questions):
        asked = {qid: q.model_dump(mode="json") for qid, q in questions.items()}
        for record in self._records:
            if record["model"] == model and record["state"] == state and record["questions"] == asked:
                return _Response(record["probabilities"])
        raise KeyError("request not found in recording")


class _Answer:
    def __init__(self, probabilities):
        self.probabilities = probabilities


class _Response:
    def __init__(self, probabilities):
        self.answers = {qid: _Answer(p) for qid, p in probabilities.items()}
