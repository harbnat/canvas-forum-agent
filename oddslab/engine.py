"""Exact probability engine for dice and card mechanics.

All numbers the agent posts come from this module, never from the model. The
model only translates a plain-English question into a `Spec`; the code here
computes the answer exactly with fractions.

Also here:
  * `independent_check` - a second, separate implementation (brute force or a
    seeded simulation of the raw physical process) used to cross-check answers.
  * `MUTANTS` - deliberately broken variants of the engine, used to measure how
    many bugs a set of peer-supplied test cases would catch.
"""

from __future__ import annotations

import math
import operator
import random
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, product
from typing import Literal, Optional

from pydantic import BaseModel, Field

MAX_OUTCOMES = 2_000_000      # exact enumeration budget for dice with keep/face rules
MAX_BRUTE = 300_000           # brute-force budget for the independent check
EXPLODE_DEPTH = 4             # an exploding die rolls again at most 4 extra times
SIM_TRIALS = 200_000

Op = Literal[">=", ">", "<=", "<", "==", "!="]
OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt,
       "==": operator.eq, "!=": operator.ne}
OFF_BY_ONE = {">=": ">", ">": ">=", "<=": "<", "<": "<=", "==": "==", "!=": "!="}

RankPattern = Literal["at_least_pair", "exactly_one_pair", "at_least_two_pair",
                      "at_least_three_of_a_kind", "full_house_or_better",
                      "four_of_a_kind", "all_distinct_ranks"]

MUTANTS = {
    "keep_wrong_end": "keeps the lowest dice when asked for the highest (and vice versa)",
    "off_by_one": "treats >= as > and <= as < (and vice versa)",
    "ignore_reroll_explode": "ignores reroll and exploding-dice rules",
    "cards_with_replacement": "draws cards with replacement",
}


class SpecError(ValueError):
    """The question is outside what the engine supports, or is malformed."""


# --------------------------------------------------------------------- spec


class DiceGroup(BaseModel):
    count: int = Field(description="number of dice of this kind, 1-10")
    sides: int = Field(description="faces per die, 2-100")
    reroll_once: list[int] = Field(default_factory=list,
                                   description="face values that are rerolled once (keep the reroll)")
    explode_on_max: bool = Field(default=False,
                                 description="rolling the maximum adds another roll of that die")


class Keep(BaseModel):
    which: Literal["highest", "lowest"]
    n: int


class Event(BaseModel):
    kind: Literal["total", "all_equal", "face_count", "card_count", "rank_pattern", "suit_max"]
    op: Optional[Op] = Field(default=None, description="comparison for total/face_count/card_count/suit_max")
    value: Optional[int] = Field(default=None, description="threshold for the comparison")
    face: Optional[int] = Field(default=None, description="face value counted by face_count")
    category: Optional[Literal["rank", "suit", "card"]] = Field(
        default=None, description="card_count: count cards of one rank, one suit, or one specific card")
    category_value: Optional[str] = Field(default=None, description="e.g. 'A', 'hearts', 'A of spades'")
    pattern: Optional[RankPattern] = None


class Spec(BaseModel):
    kind: Literal["dice", "cards", "unsupported"]
    question: Literal["probability", "expectation", "distribution", "expected_trials_until"] = "probability"
    dice: list[DiceGroup] = Field(default_factory=list)
    keep: Optional[Keep] = None
    modifier: int = 0
    draw: Optional[int] = Field(default=None, description="cards drawn without replacement")
    ranks: int = 13
    suits: int = 4
    event: Optional[Event] = None
    note: str = Field(default="", description="if unsupported: why")


@dataclass
class Result:
    value: Fraction                       # the requested number (probability, mean, or trials)
    question: str
    method: str
    outcomes: int = 0
    distribution: dict[int, Fraction] = field(default_factory=dict)
    truncated_mass: Fraction = Fraction(0)


# ----------------------------------------------------------------- helpers


def _cmp(op: str, a: int, b: int, mutant: str | None) -> bool:
    if mutant == "off_by_one":
        op = OFF_BY_ONE[op]
    return OPS[op](a, b)


def validate(spec: Spec) -> None:
    if spec.kind == "unsupported":
        raise SpecError(spec.note or "question is outside dice and card probabilities")
    needs_event = spec.question in ("probability", "expected_trials_until")
    if needs_event and spec.event is None:
        raise SpecError("a probability needs an event")
    ev = spec.event
    if ev and ev.kind in ("total", "face_count", "card_count", "suit_max") and (ev.op is None or ev.value is None):
        raise SpecError(f"event '{ev.kind}' needs a comparison and a value")
    if spec.kind == "dice":
        if not 1 <= len(spec.dice) <= 4:
            raise SpecError("between 1 and 4 groups of dice")
        total = sum(g.count for g in spec.dice)
        if not 1 <= total <= 10:
            raise SpecError("between 1 and 10 dice")
        for g in spec.dice:
            if not 2 <= g.sides <= 100:
                raise SpecError("dice must have 2-100 sides")
            if any(not 1 <= v <= g.sides for v in g.reroll_once) or len(set(g.reroll_once)) >= g.sides:
                raise SpecError("reroll values must be real faces, and not every face")
        if spec.keep and not 1 <= spec.keep.n <= total:
            raise SpecError("can only keep between 1 and all of the dice")
        if ev and ev.kind not in ("total", "all_equal", "face_count"):
            raise SpecError(f"event '{ev.kind}' does not apply to dice")
        if ev and ev.kind == "face_count" and ev.face is None:
            raise SpecError("face_count needs a face")
    elif spec.kind == "cards":
        if not (1 <= spec.ranks <= 13 and 1 <= spec.suits <= 4):
            raise SpecError("deck must have 1-13 ranks and 1-4 suits")
        if spec.draw is None or not 1 <= spec.draw <= min(7, spec.ranks * spec.suits):
            raise SpecError("draw between 1 and 7 cards")
        if ev is None or ev.kind not in ("card_count", "rank_pattern", "suit_max"):
            raise SpecError("card questions need a card_count, rank_pattern or suit_max event")
        if ev.kind == "card_count" and ev.category is None:
            raise SpecError("card_count needs a category (rank, suit or card)")
        if ev.kind == "rank_pattern" and ev.pattern is None:
            raise SpecError("rank_pattern needs a pattern")
        if spec.question == "distribution":
            raise SpecError("distributions are supported for dice totals only")
        if spec.question == "expectation" and ev.kind != "card_count":
            raise SpecError("averages for cards are supported for card counts only")


# --------------------------------------------------------------------- dice


def die_distribution(g: DiceGroup, mutant: str | None = None) -> dict[int, Fraction]:
    s = g.sides
    one = Fraction(1, s)
    dist = {v: one for v in range(1, s + 1)}
    if mutant == "ignore_reroll_explode":
        return dist
    rr = set(g.reroll_once)
    if rr:
        p_rr = Fraction(len(rr), s)
        dist = {v: (Fraction(0) if v in rr else one) + p_rr * one for v in range(1, s + 1)}
    if g.explode_on_max:
        base, out = dist, {}

        def chain(total: int, depth: int, prob: Fraction) -> None:
            for v, p in base.items():
                if v == s and depth < EXPLODE_DEPTH:
                    chain(total + v, depth + 1, prob * p)
                else:
                    out[total + v] = out.get(total + v, Fraction(0)) + prob * p

        chain(0, 0, Fraction(1))
        dist = out
    return dist


def _kept(values: tuple[int, ...], keep: Keep | None, mutant: str | None) -> tuple[int, ...]:
    if keep is None:
        return values
    which = keep.which
    if mutant == "keep_wrong_end":
        which = "lowest" if which == "highest" else "highest"
    ordered = sorted(values, reverse=(which == "highest"))
    return tuple(ordered[: keep.n])


def _dice_event(kept: tuple[int, ...], total: int, ev: Event, mutant: str | None) -> bool:
    if ev.kind == "total":
        return _cmp(ev.op, total, ev.value, mutant)
    if ev.kind == "all_equal":
        return len(set(kept)) == 1
    if ev.kind == "face_count":
        return _cmp(ev.op, sum(1 for v in kept if v == ev.face), ev.value, mutant)
    raise SpecError(f"event '{ev.kind}' does not apply to dice")


def _convolve(a: dict[int, Fraction], b: dict[int, Fraction]) -> dict[int, Fraction]:
    out: dict[int, Fraction] = {}
    for x, px in a.items():
        for y, py in b.items():
            out[x + y] = out.get(x + y, Fraction(0)) + px * py
    return out


def compute_dice(spec: Spec, mutant: str | None = None) -> Result:
    dists = [die_distribution(g, mutant) for g in spec.dice for _ in range(g.count)]
    ev = spec.event
    truncated = sum((Fraction(1, g.sides) ** (EXPLODE_DEPTH + 1)) * g.count
                    for g in spec.dice if g.explode_on_max and mutant != "ignore_reroll_explode")
    simple = spec.keep is None and (ev is None or ev.kind == "total")

    totals: dict[int, Fraction] = {}
    p_event = Fraction(0)
    if simple:
        acc = {0: Fraction(1)}
        for d in dists:
            acc = _convolve(acc, d)
        totals = {t + spec.modifier: p for t, p in acc.items()}
        if ev is not None:
            p_event = sum((p for t, p in totals.items() if _cmp(ev.op, t, ev.value, mutant)), Fraction(0))
        outcomes = math.prod(len(d) for d in dists)
        method = "exact convolution of the per-die distributions"
    else:
        outcomes = math.prod(len(d) for d in dists)
        if outcomes > MAX_OUTCOMES:
            raise SpecError(f"too many outcomes to enumerate exactly ({outcomes:,})")
        items = [list(d.items()) for d in dists]
        for combo in product(*items):
            prob = Fraction(1)
            values = []
            for v, p in combo:
                prob *= p
                values.append(v)
            kept = _kept(tuple(values), spec.keep, mutant)
            total = sum(kept) + spec.modifier
            totals[total] = totals.get(total, Fraction(0)) + prob
            if ev is not None and _dice_event(kept, total, ev, mutant):
                p_event += prob
        method = f"exact enumeration of all {outcomes:,} outcomes"
    return _finish(spec, totals, p_event, method, outcomes, truncated)


def _finish(spec: Spec, totals: dict[int, Fraction], p_event: Fraction, method: str,
            outcomes: int, truncated: Fraction = Fraction(0)) -> Result:
    if spec.question == "probability":
        value = p_event
    elif spec.question == "expected_trials_until":
        if p_event == 0:
            raise SpecError("the event never happens, so the expected number of tries is infinite")
        value = 1 / p_event
        method += "; expected tries = 1 / P(event) for independent repeats"
    elif spec.question == "expectation":
        value = sum((t * p for t, p in totals.items()), Fraction(0))
    else:  # distribution: report the mean as the headline number
        value = sum((t * p for t, p in totals.items()), Fraction(0))
    dist = dict(sorted(totals.items())) if spec.question == "distribution" else {}
    return Result(value=value, question=spec.question, method=method, outcomes=outcomes,
                  distribution=dist, truncated_mass=truncated)


# -------------------------------------------------------------------- cards


def _compositions(n: int, parts: int, cap: int):
    """All ways to write n as an ordered sum of `parts` integers in [0, cap]."""
    if parts == 0:
        if n == 0:
            yield ()
        return
    for c in range(min(n, cap) + 1):
        for rest in _compositions(n - c, parts - 1, cap):
            yield (c,) + rest


def _rank_pattern(counts: tuple[int, ...], pattern: str) -> bool:
    big = sorted((c for c in counts if c), reverse=True)
    top = big[0] if big else 0
    pairs = sum(1 for c in big if c >= 2)
    if pattern == "at_least_pair":
        return top >= 2
    if pattern == "exactly_one_pair":
        return pairs == 1 and top == 2
    if pattern == "at_least_two_pair":
        return pairs >= 2
    if pattern == "at_least_three_of_a_kind":
        return top >= 3
    if pattern == "full_house_or_better":
        return top >= 4 or (top >= 3 and pairs >= 2)
    if pattern == "four_of_a_kind":
        return top >= 4
    if pattern == "all_distinct_ranks":
        return top <= 1
    raise SpecError(f"unknown pattern {pattern}")


def _category_size(spec: Spec) -> int:
    cat = spec.event.category
    return spec.suits if cat == "rank" else spec.ranks if cat == "suit" else 1


def compute_cards(spec: Spec, mutant: str | None = None) -> Result:
    R, S, n = spec.ranks, spec.suits, spec.draw
    N = R * S
    ev = spec.event
    replace = mutant == "cards_with_replacement"
    total_ways = math.comb(N, n)

    if ev.kind == "card_count":
        K = _category_size(spec)
        dist: dict[int, Fraction] = {}
        for k in range(0, n + 1):
            if replace:
                p = math.comb(n, k) * Fraction(K, N) ** k * Fraction(N - K, N) ** (n - k)
            else:
                p = Fraction(math.comb(K, k) * math.comb(N - K, n - k), total_ways)
            if p:
                dist[k] = p
        p_event = sum((p for k, p in dist.items() if _cmp(ev.op, k, ev.value, mutant)), Fraction(0))
        method = ("binomial (with replacement)" if replace else
                  f"hypergeometric count over all {total_ways:,} hands")
        return _finish(spec, dist, p_event, method, total_ways)

    groups, cap = (R, S) if ev.kind == "rank_pattern" else (S, R)
    p_event = Fraction(0)
    for counts in _compositions(n, groups, n if replace else cap):
        if replace:
            ways = Fraction(math.factorial(n), math.prod(math.factorial(c) for c in counts)) / groups ** n
        else:
            ways = Fraction(math.prod(math.comb(cap, c) for c in counts), total_ways)
        if ev.kind == "rank_pattern":
            hit = _rank_pattern(counts, ev.pattern)
        else:
            hit = _cmp(ev.op, max(counts), ev.value, mutant)
        if hit:
            p_event += ways
    what = "rank" if ev.kind == "rank_pattern" else "suit"
    method = f"exact count over {what} patterns of all {total_ways:,} hands"
    return _finish(spec, {}, p_event, method, total_ways)


def compute(spec: Spec, mutant: str | None = None) -> Result:
    validate(spec)
    return compute_dice(spec, mutant) if spec.kind == "dice" else compute_cards(spec, mutant)


# --------------------------------------------- independent cross-checks


@dataclass
class Check:
    value: float
    method: str
    tolerance: float      # how far the exact answer may be from `value` and still agree
    exact: bool


def _roll(rng: random.Random, g: DiceGroup) -> int:
    total, depth = 0, 0
    while True:
        v = rng.randint(1, g.sides)
        if v in g.reroll_once:
            v = rng.randint(1, g.sides)
        total += v
        if g.explode_on_max and v == g.sides and depth < EXPLODE_DEPTH:
            depth += 1
            continue
        return total


def _sample_once(spec: Spec, rng: random.Random) -> tuple[bool, float]:
    """One play of the raw physical process. Returns (event happened, quantity)."""
    ev = spec.event
    if spec.kind == "dice":
        values = [_roll(rng, g) for g in spec.dice for _ in range(g.count)]
        if spec.keep:
            values = sorted(values, reverse=spec.keep.which == "highest")[: spec.keep.n]
        total = sum(values) + spec.modifier
        hit = False
        if ev is not None:
            if ev.kind == "total":
                hit = OPS[ev.op](total, ev.value)
            elif ev.kind == "all_equal":
                hit = len(set(values)) == 1
            else:
                hit = OPS[ev.op](values.count(ev.face), ev.value)
        return hit, total
    deck = [(r, s) for r in range(spec.ranks) for s in range(spec.suits)]
    hand = rng.sample(deck, spec.draw)
    if ev.kind == "card_count":
        if ev.category == "rank":
            k = sum(1 for r, _ in hand if r == 0)
        elif ev.category == "suit":
            k = sum(1 for _, s in hand if s == 0)
        else:
            k = sum(1 for c in hand if c == (0, 0))
        return OPS[ev.op](k, ev.value), k
    if ev.kind == "rank_pattern":
        counts = [sum(1 for r, _ in hand if r == x) for x in range(spec.ranks)]
        return _rank_pattern(tuple(counts), ev.pattern), 0
    counts = [sum(1 for _, s in hand if s == x) for x in range(spec.suits)]
    return OPS[ev.op](max(counts), ev.value), 0


def simulate(spec: Spec, seed: int, trials: int = SIM_TRIALS) -> Check:
    validate(spec)
    rng = random.Random(seed)
    hits, qty = 0, 0.0
    for _ in range(trials):
        hit, q = _sample_once(spec, rng)
        hits += hit
        qty += q
    p = hits / trials
    if spec.question in ("probability", "expected_trials_until"):
        se = math.sqrt(max(p * (1 - p), 1e-12) / trials)
        if spec.question == "probability":
            return Check(p, f"simulation of {trials:,} plays (seed {seed})", 5 * se + 1e-4, False)
        if p == 0:
            return Check(float("inf"), f"simulation of {trials:,} plays (seed {seed})", float("inf"), False)
        est = 1 / p
        return Check(est, f"simulation of {trials:,} plays (seed {seed})", 5 * se / p ** 2 + 1e-3, False)
    mean = qty / trials
    return Check(mean, f"simulation of {trials:,} plays (seed {seed})", max(0.05 * abs(mean), 0.05), False)


def _brute(spec: Spec) -> Check | None:
    """Plain brute force over every raw outcome, sharing no code with `compute`."""
    ev = spec.event
    if spec.kind == "dice":
        if any(g.reroll_once or g.explode_on_max for g in spec.dice):
            return None
        faces = [range(1, g.sides + 1) for g in spec.dice for _ in range(g.count)]
        if math.prod(len(f) for f in faces) > MAX_BRUTE:
            return None
        hits = tot = n = 0
        for roll in product(*faces):
            vals = sorted(roll, reverse=True) if spec.keep and spec.keep.which == "highest" else sorted(roll)
            vals = vals[: spec.keep.n] if spec.keep else list(roll)
            total = sum(vals) + spec.modifier
            n += 1
            tot += total
            if ev is None:
                continue
            if ev.kind == "total":
                hits += OPS[ev.op](total, ev.value)
            elif ev.kind == "all_equal":
                hits += len(set(vals)) == 1
            else:
                hits += OPS[ev.op](vals.count(ev.face), ev.value)
        return _brute_result(spec, hits, tot, n)
    N = spec.ranks * spec.suits
    if math.comb(N, spec.draw) > MAX_BRUTE:
        return None
    hits = qty = n = 0
    for hand in combinations(range(N), spec.draw):
        ranks = [c % spec.ranks for c in hand]
        suits = [c // spec.ranks for c in hand]
        n += 1
        if ev.kind == "card_count":
            k = (ranks.count(0) if ev.category == "rank" else
                 suits.count(0) if ev.category == "suit" else hand.count(0))
            qty += k
            hits += OPS[ev.op](k, ev.value)
        elif ev.kind == "rank_pattern":
            hits += _rank_pattern(tuple(ranks.count(x) for x in range(spec.ranks)), ev.pattern)
        else:
            hits += OPS[ev.op](max(suits.count(x) for x in range(spec.suits)), ev.value)
    return _brute_result(spec, hits, qty, n)


def _brute_result(spec: Spec, hits: int, qty: int, n: int) -> Check:
    method = f"brute force over all {n:,} raw outcomes"
    if spec.question == "probability":
        return Check(hits / n, method, 1e-12, True)
    if spec.question == "expected_trials_until":
        return Check(n / hits if hits else float("inf"), method, 1e-9, True)
    return Check(qty / n, method, 1e-9, True)


def independent_check(spec: Spec, seed: int) -> Check:
    return _brute(spec) or simulate(spec, seed)


def agrees(exact: Fraction, check: Check) -> bool:
    return abs(float(exact) - check.value) <= check.tolerance


# ----------------------------------------------------- reading answers


def parse_claim(text: str) -> tuple[Fraction, float] | None:
    """A claimed answer as (value, tolerance). Accepts 7/36, 0.1944, 19.44%."""
    t = text.strip().replace(",", "").replace("≈", "").strip()
    if not t:
        return None
    try:
        if "/" in t:
            a, b = t.split("/", 1)
            return Fraction(int(a.strip()), int(b.strip())), 0.0
        pct = t.endswith("%")
        num = t.rstrip("%").strip()
        decimals = len(num.split(".", 1)[1]) if "." in num else 0
        value = Fraction(num) / (100 if pct else 1)
        decimals += 2 if pct else 0
        return value, max(0.5 * 10 ** -decimals, 1e-9) if decimals >= 3 else 5e-4
    except (ValueError, ZeroDivisionError):
        return None


def matches(claim: tuple[Fraction, float], exact: Fraction) -> bool:
    value, tol = claim
    return value == exact if tol == 0 else abs(float(value) - float(exact)) <= tol + 1e-12


def describe(spec: Spec) -> str:
    """Plain-English restatement, so the other agent can see how we read their question."""
    ev = spec.event
    if spec.kind == "dice":
        parts = []
        for g in spec.dice:
            s = f"{g.count}d{g.sides}"
            if g.reroll_once:
                s += f" (reroll {', '.join(map(str, g.reroll_once))} once)"
            if g.explode_on_max:
                s += " (exploding)"
            parts.append(s)
        what = " + ".join(parts)
        if spec.keep:
            what += f", keep the {spec.keep.which} {spec.keep.n}"
        if spec.modifier:
            what += f", {'+' if spec.modifier > 0 else ''}{spec.modifier}"
        subject = "total"
    else:
        deck = "standard 52-card deck" if (spec.ranks, spec.suits) == (13, 4) else \
            f"{spec.ranks}-rank x {spec.suits}-suit deck"
        what = f"draw {spec.draw} cards from a {deck} without replacement"
        subject = "count"
    if ev is None:
        cond = ""
    elif ev.kind == "total":
        cond = f"total {ev.op} {ev.value}"
    elif ev.kind == "all_equal":
        cond = "all kept dice equal"
    elif ev.kind == "face_count":
        cond = f"number of {ev.face}s {ev.op} {ev.value}"
    elif ev.kind == "card_count":
        cond = f"number of {ev.category_value or ev.category} cards {ev.op} {ev.value}"
    elif ev.kind == "rank_pattern":
        cond = ev.pattern.replace("_", " ")
    else:
        cond = f"most cards of one suit {ev.op} {ev.value}"
    q = {"probability": f"P({cond})", "expectation": f"average {subject}",
         "distribution": f"distribution of the {subject}",
         "expected_trials_until": f"expected number of tries until {cond}"}[spec.question]
    return f"{what}; {q}"
