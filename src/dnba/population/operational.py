"""Layer 2, minimal version for hypothesis B-K1 (invoicing module with KSeF).

Controlled, varied operational situations with explicit assumptions, not an estimated distribution
of Polish firms (docs/tor-b-etap1.md, assumptions A0-14 to A0-20 in docs/etap0-populacja.md). Size
class and PKD section change probabilities but never fix a profile; consistency rules remove
impossible combinations only. Drawn deterministically from the firm seed and the variant.

The data flow is three separate links (sending to KSeF, passing to the books, matching payments),
each manual, by files or automatic (and for the books also: the office takes the invoices from KSeF
itself), because the module automates some links and not others.

Variants (sensitivity of the pilot): "base", "manual" (more manual work), "integrated" (more
integration). The firm's own features come from one random stream shared by all variants; the tool
and the links are chosen from their own uniform numbers, shared by all variants, so a variant changes
only their probabilities (and what follows from the links: who passes documents on). The same firm gets the same profile for the same variant.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

VERSION = "0.4.0"
VARIANTS = ("base", "manual", "integrated")
TOOLS = ("none", "invoicing_program", "erp")
LINK_MODES = ("manual", "files", "automatic", "office_from_ksef")
MODES_OF = {"ksef_submission": ("manual", "files", "automatic"),
            "to_accounting": ("manual", "files", "automatic", "office_from_ksef"),
            "payment_reconciliation": ("manual", "files", "automatic")}
LINKS = ("ksef_submission", "to_accounting", "payment_reconciliation")
ACCOUNTING = ("own", "external", "mixed")

# A0-15: median invoices in a typical month by size class (issued, received); lognormal, wide spread
ISSUED_MEDIAN = {"0-9": 8, "10-49": 40, "50-249": 150, "250+": 600}
RECEIVED_MEDIAN = {"0-9": 10, "10-49": 40, "50-249": 120, "250+": 400}
ISSUED_SIGMA, RECEIVED_SIGMA = 1.4, 1.1

# A0-14: share of sales value to consumers; mixture of bands per PKD section (weights for
# none, low 0-20%, mid 20-80%, high 80-100%)
B2C_BANDS = ((0.0, 0.0), (0.0, 0.2), (0.2, 0.8), (0.8, 1.0))
B2C_BY_SECTION = {
    "G": (0.30, 0.25, 0.25, 0.20), "I": (0.05, 0.10, 0.25, 0.60), "S": (0.10, 0.15, 0.25, 0.50),
    "Q": (0.15, 0.20, 0.30, 0.35), "P": (0.20, 0.20, 0.30, 0.30), "R": (0.20, 0.20, 0.30, 0.30),
    "L": (0.30, 0.30, 0.25, 0.15), "H": (0.55, 0.25, 0.15, 0.05), "F": (0.50, 0.25, 0.20, 0.05),
}
B2C_DEFAULT = (0.65, 0.25, 0.08, 0.02)

# A0-16: invoicing tool by monthly issued volume band (up to 10, 11-50, 51-300, over 300)
TOOL_BY_VOLUME = ((0.45, 0.50, 0.05), (0.20, 0.65, 0.15), (0.06, 0.59, 0.35), (0.02, 0.38, 0.60))
ERP_SIZE_SHIFT = {"0-9": 0.0, "10-49": 0.05, "50-249": 0.12, "250+": 0.20}
# A0-17: each data link given the tool (manual, files, automatic); automatic needs a tool. KSeF without a
# tool is manual: the free KSeF application of the Ministry of Finance or retyping an invoice made in an
# editor (no files). A program sends to KSeF automatically in most cases (2026).
LINK_BY_TOOL = {
    "ksef_submission": {"none": (1.0, 0.0, 0.0), "invoicing_program": (0.10, 0.05, 0.85), "erp": (0.05, 0.15, 0.80)},
    "to_accounting": {"none": (0.75, 0.25, 0.0), "invoicing_program": (0.30, 0.50, 0.20), "erp": (0.15, 0.35, 0.50)},
    "payment_reconciliation": {"none": (0.85, 0.15, 0.0), "invoicing_program": (0.50, 0.35, 0.15), "erp": (0.20, 0.40, 0.40)},
}
OFFICE_FROM_KSEF = 0.30  # A0-17: an external office downloads the invoices from KSeF itself (moved from the other modes)
VARIANT_SHIFT = 0.15  # A0-20: "manual" moves this much probability to none/manual, "integrated" to erp/automatic


def rng_for(firm_seed: int | str, stream: str) -> np.random.Generator:
    """A random stream of one firm. "base" draws the firm's own features; "tool" draws the uniform
    numbers of the tool and the links, the same for every variant (common random numbers)."""
    digest = hashlib.sha256(f"layer2/{VERSION}/{firm_seed}/{stream}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _pick(options: tuple, p: np.ndarray, u: float) -> str:
    """Inverse-CDF choice: with the same u, a variant that shifts probability toward an option
    changes the choice only where the shift crosses u."""
    c = np.cumsum(p / p.sum())
    return options[min(int(np.searchsorted(c, u, side="right")), len(options) - 1)]


def _shift(p: tuple, toward: int, amount: float) -> np.ndarray:
    """Move `amount` of probability mass to index `toward`, proportionally from the others. Zero
    entries stay zero (an impossible option stays impossible)."""
    p = np.array(p, dtype=float)
    if p[toward] == 0:  # an impossible option stays impossible, also as the target of a shift
        return p / p.sum()
    others = [i for i in range(len(p)) if i != toward and p[i] > 0]
    take = min(amount, sum(p[i] for i in others))
    if take > 0:
        tot = sum(p[i] for i in others)
        for i in others:
            p[i] -= take * p[i] / tot
        p[toward] += take
    p = np.clip(p, 0.0, None)  # rounding when the whole remaining mass is taken
    return p / p.sum()


def _band(n: int) -> int:
    return 0 if n <= 10 else 1 if n <= 50 else 2 if n <= 300 else 3


def _persons(authority: dict | None) -> dict[str, list[str]]:
    """Persons of the firm by role, from the authority generator (owner and board functions,
    company roles)."""
    out: dict[str, list[str]] = {"head": [], "accountant": [], "accounting_firm": [], "assistant": [],
                                 "dedicated_employee": [], "cfo": []}
    for p in (authority or {}).get("persons", []):
        if p.get("function") in ("owner", "president", "partner", "general_partner", "member", "vice_president"):
            out["head"].append(p["id"])
        for r in p.get("company_roles", []):
            if r in out:
                out[r].append(p["id"])
    return out


def draw_layer2(firm: dict, authority: dict | None = None, variant: str = "base", *,
                suspended: bool = False, p_external: float | None = None,
                invoices=None, tool_adjust=None, solo: bool = False, b2c_only: bool = False,
                b2c_weights: tuple | None = None) -> dict:
    """Layer-2 fields of one firm (dict with firm_seed, size_class, pkd_section), as in docs/tor-b-etap1.md.

    Version 0.5 (layer2.py) passes two more facts; without them the result is exactly that of 0.4.0
    (B-K1 is frozen on it): suspended (the firm issues no invoices and receives few) and p_external (the
    chance that the books are kept outside when nobody with bank access keeps them). `invoices`, a function
    of the B2C share returning (issued, received) a month, replaces the 0.4 draw of the invoice counts
    (A0-15); the random numbers of 0.4 are still drawn, so the other features keep their values.
    `tool_adjust`, a function of the tool probabilities (none, program, ERP), changes them before the pick.
    `solo` (layer 2 0.6.13): only the owner works in the firm, so a person with a finance role and bank access
    (accountant, CFO, assistant, dedicated employee, administrator) is outside it: counted as an accounting office for the books
    and the division of work, as bank_operator does for the bank. `b2c_only` (0.6.15): all sales to consumers (the
    random numbers are drawn as before, so the other features keep their values). `b2c_weights` (layer 2 0.6.17): the
    weights of the four B2C bands for this firm instead of those of its PKD section (same random numbers)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}, expected one of {VARIANTS}")
    rng = rng_for(firm["firm_seed"], "base")  # the firm's own features: the same in every variant
    size, section = firm["size_class"], firm["pkd_section"]

    w = np.array(B2C_BY_SECTION.get(section, B2C_DEFAULT))
    if b2c_weights is None:  # 0.4.0 and the sections without division weights: exactly as before
        lo, hi = B2C_BANDS[int(rng.choice(4, p=w / w.sum()))]
        b2c = round(float(rng.uniform(lo, hi)), 2) if hi > lo else lo
    else:
        # layer 2 0.6.17: the division's band from the SAME uniform that picked the section's band (rng.choice(4, p)
        # takes one uniform and searches the cumulative weights), and the base stream consumed exactly as with the
        # section's weights (a position uniform only when the section's band is not 'none'), so that every later
        # draw of the firm keeps its value; a firm moved out of 'none' takes its position from its own stream
        u0 = rng.random()
        band_sec = int(np.searchsorted(np.cumsum(w / w.sum()), u0, side="right"))
        wn = np.array(b2c_weights)
        band = min(3, int(np.searchsorted(np.cumsum(wn / wn.sum()), u0, side="right")))
        t = rng.random() if B2C_BANDS[min(3, band_sec)][1] > B2C_BANDS[min(3, band_sec)][0] else rng_for(firm["firm_seed"], "b2c_pos").random()
        lo, hi = B2C_BANDS[band]
        b2c = round(float(lo + (hi - lo) * t), 2) if hi > lo else lo
    if b2c_only:  # layer 2 0.6.15: e.g. a tax card in health or veterinary services covers services to consumers only
        b2c = 1.0

    # issued invoices: B2B sales are invoiced, B2C mostly receipts with invoices on demand (A0-15)
    med = ISSUED_MEDIAN[size] * (0.15 + 0.85 * (1 - b2c))
    issued = int(round(math.exp(math.log(med) + ISSUED_SIGMA * rng.standard_normal())))
    received = max(1, int(round(math.exp(math.log(RECEIVED_MEDIAN[size]) + RECEIVED_SIGMA * rng.standard_normal()))))
    if invoices is not None:
        issued, received = invoices(b2c)
        received = max(1, int(received))
    # active counterparties in the typical month: at most the invoices (A0-18)
    if invoices is None:  # 0.4.0 path, frozen: the uniform is drawn only when the firm issues invoices
        customers = 0 if issued == 0 else max(1, int(round(issued * rng.uniform(0.15, 1.0))))
    else:  # 0.5+ (layer2 0.6.11): always drawn, so a revenue change that moves issued across 0 shifts no other draw
        u_customers = rng.uniform(0.15, 1.0)
        customers = 0 if issued == 0 else max(1, int(round(issued * u_customers)))
    suppliers = max(1, int(round(received * rng.uniform(0.2, 1.0))))
    u_office, u_issuer, u_issuer_pick = rng.random(), rng.random(), rng.random()
    if suspended:  # a suspended firm sells nothing; costs (e.g. rent, leasing) may still come in
        issued, customers = 0, 0
        received = max(1, int(round(received * 0.1)))
        suppliers = min(suppliers, received)

    # tool and links: own stream of uniform numbers, the variant changes only the probabilities (A0-20)
    u = rng_for(firm["firm_seed"], "tool").random(1 + len(LINKS))
    p_tool = _shift(np.array(TOOL_BY_VOLUME[_band(issued)]), 2, ERP_SIZE_SHIFT[size])
    if tool_adjust is not None:
        p_tool = tool_adjust(p_tool)
    p_tool = _shift(p_tool, 0, VARIANT_SHIFT) if variant == "manual" else \
        _shift(p_tool, 2, VARIANT_SHIFT) if variant == "integrated" else p_tool
    tool = _pick(TOOLS, p_tool, u[0])
    # accounting organisation follows the persons the firm has in the bank (A0-19)
    ps = _persons(authority)
    if solo:  # nobody works besides the owner: the finance persons with access are outside the firm
        admins = [q["id"] for q in (authority or {}).get("persons", []) if "administrator" in q.get("company_roles", [])]
        outside = ps["accounting_firm"] + [x for r in ("accountant", "cfo", "assistant", "dedicated_employee") for x in ps[r]] + admins
        book_keepers = list(dict.fromkeys(ps["accounting_firm"] + ps["accountant"] + ps["cfo"]))
        ps = {**ps, "accounting_firm": list(dict.fromkeys(outside)), "accountant": [], "cfo": [], "assistant": [], "dedicated_employee": []}
    has_firm, has_acc = bool(ps["accounting_firm"]), bool(ps["accountant"] or ps["cfo"])
    if has_firm and has_acc:
        accounting = "mixed"
    elif has_firm:
        accounting = "external"
    elif has_acc:
        accounting = "own"
    else:  # nobody with bank access does the books: an office without access, or the head does them
        if p_external is None:
            p_external = {"0-9": 0.8, "10-49": 0.5, "50-249": 0.2, "250+": 0.05}[size]
        accounting = "external" if u_office < p_external else "own"

    links = {}
    for k, link in enumerate(LINKS):  # each connection separately: the module automates some and not others
        p = np.array(LINK_BY_TOOL[link][tool], dtype=float)
        if link == "to_accounting":  # an office can take the invoices from KSeF itself
            p = np.append(p * (1 - OFFICE_FROM_KSEF), OFFICE_FROM_KSEF) if accounting in ("external", "mixed") else np.append(p, 0.0)
        p = _shift(p, 0, VARIANT_SHIFT) if variant == "manual" else _shift(p, 2, VARIANT_SHIFT) if variant == "integrated" else p
        links[link] = _pick(MODES_OF[link], p, u[1 + k])

    # division of work: who issues, who passes documents on, who books
    head = ps["head"][:1] or ["owner"]
    issuer_pool = (ps["assistant"] + ps["dedicated_employee"] + ps["accountant"]) if issued > 30 else []
    issuer = [issuer_pool[int(u_issuer_pick * len(issuer_pool))]] if issuer_pool and u_issuer < 0.7 else head
    inside = (ps["accountant"] or ps["cfo"] or head)[:1]
    # an office without bank access has no person id; in a solo firm only an office or an accountant (now outside)
    # keeps the books, not an outside assistant or administrator
    office = (book_keepers[:1] if solo else ps["accounting_firm"][:1]) or ["accounting_office"]
    booker = inside if accounting == "own" else office if accounting == "external" else inside + office
    passes_on = passes_on_for(links["to_accounting"], issuer, booker)
    out = {
        "layer2_version": VERSION, "variant": variant,
        "invoices_issued_month": issued, "invoices_received_month": received,
        "customers_active_month": customers, "suppliers_active_month": suppliers,
        "b2c_share_of_sales": b2c, "invoicing_tool": tool, **links, "accounting": accounting,
        "division_of_work": {"issues": issuer, "passes_on": passes_on, "books": booker},
    }
    problems = consistency(out)
    assert not problems, problems  # the draw itself must never produce an impossible profile
    return out


def passes_on_for(to_accounting: str, issuer: list, booker: list) -> list:
    """Who passes the documents on to the books: nobody when the link is automatic, when the office takes
    the invoices from KSeF itself or when the issuer keeps the books; otherwise the issuer."""
    return [] if to_accounting in ("automatic", "office_from_ksef") or set(booker) <= set(issuer) else list(issuer)


def consistency(p: dict) -> list[str]:
    """Impossible combinations (not atypical ones). Also used to check interventions."""
    bad = []
    if p["customers_active_month"] > p["invoices_issued_month"]:
        bad.append("more active customers than issued invoices")
    if p["suppliers_active_month"] > p["invoices_received_month"]:
        bad.append("more active suppliers than received invoices")
    if p["invoices_issued_month"] > 0 and p["customers_active_month"] == 0:
        bad.append("issued invoices without a customer")
    for link in LINKS:
        if p[link] not in MODES_OF[link]:
            bad.append(f"{link}: unknown mode {p[link]}")
        if p[link] == "automatic" and p["invoicing_tool"] == "none":
            bad.append(f"{link}: automatic without an invoicing tool")
    if p["ksef_submission"] == "files" and p["invoicing_tool"] == "none":
        bad.append("KSeF by files without an invoicing tool")
    if p["to_accounting"] == "office_from_ksef" and p["accounting"] == "own":
        bad.append("the office takes invoices from KSeF but the books are kept in the firm")
    if not 0.0 <= p["b2c_share_of_sales"] <= 1.0:
        bad.append("B2C share outside 0-1")
    d = p["division_of_work"]
    if not d["issues"] or not d["books"]:
        bad.append("nobody issues or nobody books")
    if p["accounting"] in ("external", "mixed") and p["to_accounting"] not in ("automatic", "office_from_ksef") \
            and not set(d["books"]) <= set(d["issues"]) and not d["passes_on"]:
        bad.append("external accounting and nobody passes the documents on")
    return bad
