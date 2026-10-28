import itertools
import math
import re

import numpy as np


FIELDS = ("version", "platform", "hardware", "backend", "settings")


def parsed_scope(scope):
    """Project recognized restrictions; return opaque predicates separately."""
    known, opaque = {}, []
    for key, value in scope.items():
        if key == "allowed":
            known[key] = {k: v for k, v in value.items() if k in FIELDS}
            opaque.extend(f"{k}={v}" for k, v in value.items() if k not in FIELDS)
        elif key == "relation":
            known[key] = [
                {k: v for k, v in row.items() if k in FIELDS} for row in value
            ]
            opaque.extend(
                f"relation {k}={v}"
                for row in value
                for k, v in row.items()
                if k not in FIELDS
            )
        elif key == "implications":
            known[key] = []
            for implication in value:
                if (set(implication["if"]) | set(implication["then"])) <= set(FIELDS):
                    known[key].append(implication)
                else:
                    opaque.append(str(implication))
        elif key == "version_range":
            try:
                for bound in ["min", "max"]:
                    if bound in value:
                        version(value[bound])
                known[key] = value
            except ValueError:
                opaque.append(str(value))
        else:
            opaque.append(f"{key}={value}")
    return known, opaque


def version(value):
    parts = re.findall(r"\d+", str(value))
    if not parts:
        raise ValueError("Version needs a numeric release identifier")
    result = tuple(int(p) for p in parts)
    return result + (0,) * max(0, 4 - len(result))


class Domain:
    def __init__(self, values, limit=100000):
        missing = set(FIELDS) - set(values)
        if missing:
            raise ValueError("Missing finite domain fields: " + str(sorted(missing)))
        count = math.prod(len(values[k]) for k in FIELDS)
        if count > limit or count == 0:
            raise ValueError(
                "Finite configuration domain must be nonempty and at most "
                + str(limit)
                + " cells"
            )
        self.configurations = [
            dict(zip(FIELDS, row))
            for row in itertools.product(*(values[k] for k in FIELDS))
        ]

    def mask(self, scope):
        allowed = scope.get("allowed", {})
        relation = scope.get("relation")
        bounds = scope.get("version_range")
        result = []
        for config in self.configurations:
            valid = all(
                config[field] in permitted for field, permitted in allowed.items()
            )
            if bounds:
                value = version(config["version"])
                valid &= "min" not in bounds or value >= version(bounds["min"])
                valid &= "max" not in bounds or (
                    value <= version(bounds["max"])
                    if bounds.get("max_inclusive", False)
                    else value < version(bounds["max"])
                )
            if relation is not None:
                valid &= any(
                    all(config[key] == value for key, value in row.items())
                    for row in relation
                )
            for implication in scope.get("implications", []):
                premise = all(
                    config[k] in vals for k, vals in implication["if"].items()
                )
                conclusion = all(
                    config[k] in vals for k, vals in implication["then"].items()
                )
                valid &= not premise or conclusion
            result.append(valid)
        return np.asarray(result, dtype=bool)

    def applicability(
        self, query_mask, fact_mask, residual_state="entailed", parsed=True
    ):
        if not (query_mask & fact_mask).any() or residual_state == "inconsistent":
            return "inconsistent"
        if np.all(~query_mask | fact_mask) and parsed and residual_state == "entailed":
            return "entailed"
        return "unknown"

    def explicit(self, mask):
        return [self.configurations[i] for i in np.flatnonzero(mask)]
