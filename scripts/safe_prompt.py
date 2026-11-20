#!/usr/bin/env python3
"""SAFE validation: the card of a SAFE record, the agent's prompt and the parser of its answer.

The card (data/reference/safe_card_key.json) is rendered from the record's own fields in three variants:
- "full": size and sector, then turnover, age, ownership, export, sources of financing, applications and outcomes;
- "segment": size and sector only, with a line saying that the other information is not given on this card (since
  0.2: without it the key's rule on missing lines could be read as "no source of financing was relevant");
- "shuffled": size and sector of the record, the other block (everything after them) from a donor record of the same
  size x sector cell (a whole real block, so the questionnaire's admissible combinations are kept).
The agent gives explicit probabilities of the answers 1-10 to the held-out question; the parser accepts one JSON
object with exactly the keys "probabilities" (an object with the keys "1".."10", numbers 0-1) and "reason" (text),
and a sum of probabilities in [1 - SUM_TOL, 1 + SUM_TOL]; it returns the probabilities divided by their sum. The
reason may be empty (it is not scored). Everything else is a ValueError (a technical error of the model, never a survey
answer "don't know"), including text nested too deeply for the JSON reader (RecursionError).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_PATH = ROOT / "data/reference/safe_card_key.json"
SUM_TOL = 0.02
SCALE = [str(i) for i in range(1, 11)]
VARIANTS = ("full", "segment", "shuffled")


def load_key() -> dict:
    return json.loads(KEY_PATH.read_text(encoding="utf-8"))


def norm(v) -> str:
    v = ("" if v is None else str(v)).strip()
    if v in ("", ".", "nan"):
        return ""
    return v[:-2] if v.endswith(".0") else v


def _value(field: dict, rec: dict) -> str:
    v = norm(rec.get(field["safe"]))
    if v not in field["values"]:
        raise ValueError(f"{field['safe']}: code {v!r} has no label")
    return field["values"][v]


def segment_lines(rec: dict, key: dict) -> list:
    return [f"- {f['label']}: {_value(f, rec)}" for f in key["fields"] if f["block"] == "segment"]


def other_lines(rec: dict, key: dict) -> list:
    out = [f"- {f['label']}: {_value(f, rec)}" for f in key["fields"] if f["block"] == "other"]
    src = key["sources_of_financing"]
    lines = []
    for it in src["items"]:
        v = norm(rec.get(it["safe"]))
        if v == "":
            continue
        if v not in src["codes"]:
            raise ValueError(f"{it['safe']}: code {v!r} has no label")
        lines.append(f"  - {it['label']}: {src['codes'][v].format(used=it['used'])}")
    if lines:
        out += [f"- {src['intro']}:"] + lines
    app = key["applications"]
    lines = []
    for it in app["items"]:
        v = norm(rec.get(it["safe"]))
        if v == "":
            continue
        if v not in app["codes"]:
            raise ValueError(f"{it['safe']}: code {v!r} has no label")
        s = f"  - {it['label']}: {app['codes'][v]}"
        o = norm(rec.get(it["outcome"]))
        if v == "1" and o:
            if o not in app["outcome_codes"]:
                raise ValueError(f"{it['outcome']}: code {o!r} has no label")
            s += f"; wynik: {app['outcome_codes'][o]}"
        lines.append(s)
    if lines:
        out += [f"- {app['intro']}:"] + lines
    else:  # since 0.3 the part is always shown, so a full card always has its four parts
        out.append(f"- {app['intro']}: {app['none_asked']}")
    return out


def card(rec: dict, variant: str, key: dict, donor: dict | None = None) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}")
    lines = segment_lines(rec, key)
    if variant == "segment":
        lines.append(f"- {key['segment_note']}")
    elif variant == "full":
        lines += other_lines(rec, key)
    elif variant == "shuffled":
        if donor is None:
            raise ValueError("the shuffled card needs a donor record")
        lines += other_lines(donor, key)
    return "Karta firmy:\n" + "\n".join(lines)


COUNTS_INSTRUCTION = (
    "Karta nie mówi o firmie wszystkiego, więc nie podajesz jednej oceny. Wyobraź sobie 100 firm wybranych losowo "
    "spośród firm działających w Polsce, które mają wszystkie cechy podane na tej karcie, a różnią się tym, czego karta "
    "nie mówi (to nie jest liczba takich firm w Polsce), i podaj, ile z nich wybrałoby każdą z ocen od 1 do 10 "
    "(odpowiedzi „nie wiem” nie bierzemy pod uwagę). Opieraj liczby na tym, jak zwykle odpowiadają w Polsce firmy o "
    "tych samych cechach w typowych warunkach ostatnich lat (dokładnego okresu nie znasz; to nie jest dopisywanie faktów "
    "o firmie); im wyraźniej karta wskazuje odpowiedź, tym bardziej liczby się skupią. Zero daj ocenie, którą wybrałaby "
    "mniej niż 1 na 100 takich firm; ocenie, którą wybrałaby co najmniej 1 na 100, daj jej liczbę, nawet jeśli to 1 "
    "albo 2. Podaj liczby całkowite od 0 do "
    "100. Nie musisz pilnować, żeby suma wynosiła dokładnie 100: liczby przeliczymy na udziały.\n\n")
ZERO_RULE = ("Zero daj ocenie, którą wybrałaby mniej niż 1 na 100 takich firm; ocenie, którą wybrałaby co najmniej 1 na "
             "100, daj jej liczbę, nawet jeśli to 1 albo 2.")
ZERO_RULE_PILOT1 = "Ocenie, której nie możesz wykluczyć, nie dawaj zera."  # the first pilot's rule, word for word
assert ZERO_RULE in COUNTS_INSTRUCTION
COUNTS_NOZERO_INSTRUCTION = COUNTS_INSTRUCTION.replace(ZERO_RULE, ZERO_RULE_PILOT1)  # arm B: only the zero rule differs
COUNTS_FORMAT = (
    '{"firms_per_100": {"1": <0-100>, "2": <0-100>, "3": <0-100>, "4": <0-100>, "5": <0-100>, "6": <0-100>, '
    '"7": <0-100>, "8": <0-100>, "9": <0-100>, "10": <0-100>}, "reason": "<po polsku, do 30 słów: na czym opierasz liczby>"}')


def system_prompt(key: dict, mode: str = "counts") -> str:
    """mode "counts" (since 0.4, the default: firms out of 100 with an identical card, integers, normalised by the
    parser) or "probabilities" (the first development pilot's instruction and answer format; the card and the
    card key are the current ones, so since 0.5 it no longer reproduces the first pilot's items byte for byte; it is
    kept for the comparison of formats on the same cards)."""
    q = key["question"]["pl"]
    rules = "\n".join(f"- {s}" for s in key["agent_key"])
    if mode in ("counts", "counts_nozero"):  # counts_nozero (arm B of the elicitation comparison): only the zero rule differs
        instr = COUNTS_INSTRUCTION if mode == "counts" else COUNTS_NOZERO_INSTRUCTION
        return ("Jesteś osobą odpowiedzialną za finanse firmy opisanej na karcie. Firma działa w Polsce. Znasz firmę tak, "
                "jak opisuje ją karta, i nie dopisujesz faktów o niej, których na karcie nie ma.\n\n"
                f"W ankiecie odpowiadasz w imieniu firmy na pytanie:\n„{q}”\n{key['question']['battery_pl']}\n\n"
                + instr + f"Klucz karty:\n{rules}\n\n"
                "Odpowiedz wyłącznie jednym obiektem JSON, bez tekstu przed nim i po nim:\n" + COUNTS_FORMAT)
    if mode != "probabilities":
        raise ValueError(f"mode {mode!r}")
    return (
        "Jesteś osobą odpowiedzialną za finanse firmy opisanej na karcie. Firma działa w Polsce. Znasz firmę tak, "
        "jak opisuje ją karta, i nie dopisujesz faktów o niej, których na karcie nie ma.\n\n"
        f"W ankiecie odpowiadasz w imieniu firmy na pytanie:\n„{q}”\n{key['question']['battery_pl']}\n\n"
        "Karta nie mówi o firmie wszystkiego, więc nie podajesz jednej oceny. Podajesz rozkład: jak rozkładałyby się "
        "odpowiedzi wielu różnych firm z identyczną kartą, czyli jak często osoba odpowiadająca w imieniu firmy o takiej "
        "karcie wybrałaby każdą z ocen od 1 do 10 (odpowiedzi „nie wiem” nie bierzemy pod uwagę). Jeśli karta mówi "
        "niewiele, oprzyj rozkład na tym, jak zwykle odpowiadają w Polsce firmy o tych samych cechach, które podaje "
        "karta, w typowych warunkach ostatnich lat (dokładnego okresu nie znasz; to nie jest dopisywanie faktów o "
        "firmie); jeśli karta wyraźnie wskazuje odpowiedź, taki rozkład będzie bardziej skupiony. Ocenie, której nie możesz wykluczyć, nie dawaj zera. Podaj prawdopodobieństwa z dokładnością do "
        "trzech miejsc po przecinku; suma ma wynosić 1 (z dokładnością do 0,01).\n\n"
        f"Klucz karty:\n{rules}\n\n"
        "Odpowiedz wyłącznie jednym obiektem JSON, bez tekstu przed nim i po nim:\n"
        '{"probabilities": {"1": <0-1>, "2": <0-1>, "3": <0-1>, "4": <0-1>, "5": <0-1>, "6": <0-1>, "7": <0-1>, '
        '"8": <0-1>, "9": <0-1>, "10": <0-1>}, "reason": "<po polsku, do 30 słów: na czym opierasz rozkład>"}'
    )


def messages(rec: dict, variant: str, key: dict, donor: dict | None = None, mode: str = "counts") -> list:
    return [{"role": "system", "content": system_prompt(key, mode)}, {"role": "user", "content": card(rec, variant, key, donor)}]


def _no_dupes(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key {k!r}")
        d[k] = v
    return d


def parse(text: str) -> dict:
    """The answer as {"probabilities": {"1": p, ..., "10": p} (normalised), "sum": raw sum, "reason": text}."""
    t = (text or "").strip()
    try:
        obj = json.loads(t, object_pairs_hook=_no_dupes)
    except json.JSONDecodeError as e:
        raise ValueError(f"not one JSON object: {e}") from None
    except RecursionError:
        raise ValueError("not one JSON object: nested too deeply") from None
    if not isinstance(obj, dict) or set(obj) != {"probabilities", "reason"}:
        raise ValueError(f"keys must be probabilities and reason, got {sorted(obj) if isinstance(obj, dict) else type(obj).__name__}")
    p = obj["probabilities"]
    if not isinstance(p, dict) or set(p) != set(SCALE):
        raise ValueError("probabilities must have exactly the keys 1..10")
    vals = {}
    for k in SCALE:
        v = p[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0 <= v <= 1):
            raise ValueError(f"probability of {k} is not a number in [0, 1]: {v!r}")
        vals[k] = float(v)
    s = sum(vals.values())
    if abs(s - 1) > SUM_TOL:
        raise ValueError(f"probabilities sum to {s:.4f}, outside 1 +/- {SUM_TOL}")
    if not isinstance(obj["reason"], str):
        raise ValueError("reason must be a text")
    return {"probabilities": {k: v / s for k, v in vals.items()}, "sum": s, "reason": obj["reason"]}


def parse_counts(text: str) -> dict:
    """The answer of the counts prompt: one JSON object with exactly the keys "firms_per_100" (an object with the keys
    "1".."10", integers 0-100, not all 0) and "reason" (text). Returned as {"probabilities": counts / their sum,
    "sum": raw sum, "reason"}; the sum need not be 100 (the prompt says the counts are turned into shares)."""
    t = (text or "").strip()
    try:
        obj = json.loads(t, object_pairs_hook=_no_dupes)
    except json.JSONDecodeError as e:
        raise ValueError(f"not one JSON object: {e}") from None
    except RecursionError:
        raise ValueError("not one JSON object: nested too deeply") from None
    if not isinstance(obj, dict) or set(obj) != {"firms_per_100", "reason"}:
        raise ValueError(f"keys must be firms_per_100 and reason, got {sorted(obj) if isinstance(obj, dict) else type(obj).__name__}")
    c = obj["firms_per_100"]
    if not isinstance(c, dict) or set(c) != set(SCALE):
        raise ValueError("firms_per_100 must have exactly the keys 1..10")
    vals = {}
    for k in SCALE:
        v = c[k]
        if isinstance(v, bool) or not isinstance(v, int) or not (0 <= v <= 100):
            raise ValueError(f"count of {k} is not an integer in [0, 100]: {v!r}")
        vals[k] = v
    s = sum(vals.values())
    if s == 0:
        raise ValueError("all counts are 0")
    if not isinstance(obj["reason"], str):
        raise ValueError("reason must be a text")
    return {"probabilities": {k: v / s for k, v in vals.items()}, "sum": s, "reason": obj["reason"]}
