#!/usr/bin/env python3
"""B-K2 prompt: the system prompt (role, rules, the agent-visible key, the module 'Należności i płynność' with its
four functions in a given order, the scale) and the parsing of the agent's answer.

Built from data/reference/b_k2_outcome_pairs.json (function texts, scale, availability) and
data/reference/b_k2_key.json (what every card field means). The agent never sees the pair, the variant, nor why a
function could be useful. The order of the functions is a permutation fixed per base profile (the same for its
four controlled versions and for every variant), recorded with each item.

    python3 scripts/b_k2_prompt.py --show            # the system prompt with the default order
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "data", "reference", "b_k2_outcome_pairs.json")
KEY = os.path.join(ROOT, "data", "reference", "b_k2_key.json")

SYSTEM = """Jesteś osobą z firmy opisanej na karcie firmy: {role}. Znasz firmę tak, jak opisuje ją karta. Bank główny firmy pyta Cię, czy funkcje nowego modułu bankowości firmowej byłyby przydatne dla Twojej firmy.

Zasady:
- Opierasz się tylko na karcie firmy, na kluczu pól poniżej i na opisie funkcji. Nie dopisujesz faktów o firmie, których na karcie nie ma.
- Każde twierdzenie o firmie, na którym opierasz ocenę, wpisujesz do "claims" z rodzajem:
  "given" (fakt z karty: identyfikator pola z nawiasu kwadratowego, bez nawiasów, np. as_of, i fragment tekstu z linii karty tego pola, przepisany dosłownie, na którym opierasz twierdzenie; dla as_of sama data),
  "derived" (obliczone z faktów karty: pola i obliczenie),
  "assumption" (czego na karcie nie ma, także Twoja interpretacja wartości z karty; jawnie jako założenie).
- Każdą funkcję oceniasz osobno, na podstawie sytuacji firmy z karty. {availability}

Klucz pól karty:
{key}

Oceniany moduł: „Należności i płynność”. Funkcje:
{functions}

Skala przydatności (liczba całkowita 1-5, osobno dla każdej funkcji):
{scale}
{no_basis}
W "reasons" podaj uzasadnienie dla każdej funkcji, także ocenionej jako null (do 30 słów; liczby i skróty liczą się jak słowa). Jeśli żadna funkcja nie ma oceny null, "no_basis" to pusty obiekt {{}}.

Odpowiedz jednym obiektem JSON i niczym więcej:
{{"ratings": {{{rating_keys}}},
 "no_basis": {{"<id funkcji>": "<powód, do 25 słów>"}},
 "claims": [{{"type": "given", "field": "<pole>", "value": "<wartość jak na karcie>"}},
            {{"type": "derived", "fields": ["<pole>", ...], "computation": "<obliczenie>", "value": "<wynik>"}},
            {{"type": "assumption", "text": "<założenie>"}}],
 "reasons": {{{reason_keys}}}}}"""

USER_MESSAGE = "Karta firmy:\n{card}\n\nOceń funkcje modułu zgodnie z zasadami."


def load() -> tuple[dict, dict]:
    with open(PLAN, encoding="utf-8") as fh:
        plan = json.load(fh)
    with open(KEY, encoding="utf-8") as fh:
        key = json.load(fh)
    return plan, key


def order_for(seed: int, ids: list[str]) -> list[str]:
    """The order of the functions for one base profile (a permutation from its seed)."""
    return [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]


def key_text(key: dict) -> str:
    lines = [f"- {g}" for g in key["general"]]
    lines += [f"- [{fid}]: {text}" for fid, text in key["fields"].items()]
    return "\n".join(lines)


def system_prompt(role: str, order: list[str] | None = None) -> str:
    plan, key = load()
    funcs = {f["id"]: f for f in plan["functions"]}
    order = order or list(funcs)
    if sorted(order) != sorted(funcs):
        raise ValueError("the order must name every function once")
    pts = plan["scale"]["points"]
    return SYSTEM.format(
        role=role, availability=plan["scale"]["availability"], key=key_text(key),
        functions="\n".join(f"- {fid}: {funcs[fid]['text']}" for fid in order),
        scale="\n".join(f"- {k}: {pts[k]}" for k in sorted(pts)),
        no_basis="Brak podstaw: " + plan["scale"]["no_basis"].replace("null", "null", 1),
        rating_keys=", ".join(f'"{fid}": <1-5 albo null>' for fid in order),
        reason_keys=", ".join(f'"{fid}": "<do 30 słów>"' for fid in order))


def parse(text: str) -> dict:
    """The agent's answer, checked in full; any defect raises ValueError with what is wrong:
    - ratings: every function, an integer 1-5 or null (never a string or a bool);
    - no_basis: an object; every function rated null has a non-empty text reason; nothing else in it;
    - reasons: every function, a non-empty text (also for null);
    - claims: a list of objects: given {field, value} or derived {fields (non-empty list), computation, value} or
      assumption {text}, all non-empty texts.
    Whether the claims are true to the card is a separate step of the analysis."""
    try:
        plan, _ = load()
        ids = [f["id"] for f in plan["functions"]]
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("no JSON object")
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("the answer is not an object")
        txt = lambda v: isinstance(v, str) and v.strip() != ""  # noqa: E731
        r = obj.get("ratings")
        if not isinstance(r, dict) or set(r) != set(ids):
            raise ValueError("ratings must name every function")
        for fid, v in r.items():
            if v is not None and not (isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5):
                raise ValueError(f"{fid}: rating {v!r} is not an integer 1-5 or null")
        nb = obj.get("no_basis", {})
        if not isinstance(nb, dict):
            raise ValueError("no_basis must be an object")
        nulls = {fid for fid, v in r.items() if v is None}
        if set(nb) - nulls:
            raise ValueError("no_basis names a function that has a rating")
        for fid in nulls:
            if not txt(nb.get(fid)):
                raise ValueError(f"{fid}: null without a text reason in no_basis")
        rs = obj.get("reasons")
        if not isinstance(rs, dict) or set(rs) != set(ids) or not all(txt(v) for v in rs.values()):
            raise ValueError("reasons must give a non-empty text for every function")
        cl = obj.get("claims")
        if not isinstance(cl, list):
            raise ValueError("claims must be a list")
        need = {"given": ("field", "value"), "derived": ("computation", "value"), "assumption": ("text",)}
        for c in cl:
            if not isinstance(c, dict) or c.get("type") not in need or not all(txt(c.get(k)) for k in need[c["type"]]):
                raise ValueError(f"a claim is not well formed: {c!r}"[:200])
            if c["type"] == "derived" and not (isinstance(c.get("fields"), list) and c["fields"] and all(txt(x) for x in c["fields"])):
                raise ValueError("a derived claim needs its fields")
        return obj
    except ValueError:
        raise
    except Exception as e:  # any other defect of the structure: the same error kind
        raise ValueError(f"malformed answer: {type(e).__name__}") from None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--role", default="właściciel")
    a = ap.parse_args()
    if a.show:
        print(system_prompt(a.role))


if __name__ == "__main__":
    main()
