"""Layer 2 (VERSION below; first written as 0.5): the operational profile of 0.4 (operational.py) with features anchored in data.

New in 0.5 (assumptions A0-21 to A0-23 in docs/etap0-populacja.md):
- form of bookkeeping (full books, KPiR, flat rate, tax card): what the law allows for the legal form,
  shares for natural-person firms anchored in GUS (Działalność przedsiębiorstw niefinansowych 2024,
  micro firms 2024); it sets the chance that an office keeps the books when nobody with bank access does;
- age and activity status of sole proprietorships (JDG) by PKD division from the CEIDG data warehouse
  (data/reference/layer2_ceidg_age_status.json): about a quarter are suspended, which REGON does not tell;
  a suspended firm stays in the population (it has an account) but issues no invoices; age of civil
  partnerships from the same source.

The 0.4 features are drawn from the same random numbers as in 0.4, so a firm keeps its invoices,
counterparties and tool unless 0.5 changes their premise (suspension, bookkeeping). B-K1 stays on 0.4.0.
"""
from __future__ import annotations

import functools
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import operational as op

VERSION = "0.6.17"
STREAM = "0.5.0"  # random streams of 0.5: kept, so a later 0.5.x changes only what it fixes
BOOKKEEPING = ("full_books", "kpir", "flat_rate", "tax_card")
AGES = ("<1", "1-2", "3-5", "6-10", "11-20", "21+")
REF = Path(__file__).resolve().parents[3] / "data" / "reference" / "layer2_ceidg_age_status.json"
SBS = Path(__file__).resolve().parents[3] / "data" / "reference" / "layer2_sbs_gus_2024.json"
PRODUCTS = Path(__file__).resolve().parents[3] / "data" / "reference" / "layer2_products_safe.json"
VAT = Path(__file__).resolve().parents[3] / "data" / "reference" / "layer2_vat_status.json"
OWNERSHIP = Path(__file__).resolve().parents[3] / "data" / "reference" / "layer2_ownership.json"
FOREIGN_GUS = Path(__file__).resolve().parents[3] / "data" / "reference" / "layer2_foreign_gus.json"
LIQUIDITY = Path(__file__).resolve().parents[3] / "data" / "reference" / "layer2_liquidity_pie.json"
BANKING = Path(__file__).resolve().parents[3] / "data" / "reference" / "layer2_banking.json"
GENERATED_SOLO = Path(__file__).resolve().parents[3] / "data" / "reference" / "layer2_generated_solo_share.json"

# A0-21: which forms may keep simplified books (act on accounting, art. 2: natural persons, civil
# partnerships and general partnerships of natural persons, professional partnerships, below EUR 2.5m of
# net revenue of the previous year since the financial year 2025, EUR 2m before (see the revenue limits below);
# partnerships only with natural persons as partners, not checked here); every commercial company otherwise (capital companies, limited and limited joint-stock
# partnerships), cooperatives and branches keep full books. Flat rate: JDG, civil and general
# partnerships of natural persons. Tax card: JDG and civil partnerships that had it before 2022 (no new
# ones since 2022), so only firms of 6 years or more here (the 3-5 bucket straddles 2022: left out).
SIMPLIFIED_OK = {"099", "019", "118", "115"}
FLAT_RATE_OK = {"099", "019", "118"}
TAX_CARD_OK = {"099", "019"}
TAX_CARD_AGES = {"6-10", "11-20", "21+"}
# 0.6.15 (audit 13.11.2026; refined after the legal review): the tax card is open only to the activities of the
# flat-rate act art. 23 (Dz.U. 2025 poz. 843): annex 3 part I (services and crafts of annex 4, with the catch-all
# "usługi inne, gdzie indziej niewymienione", open only to firms without employees), part II-III retail trade (without
# fuels, motor vehicles and parts, licensed goods), IV gastronomy without alcohol, V transport with one vehicle, VI
# entertainment, VII home meals, VIII health care, IX veterinary services, X home care, XI lessons. The population has
# the PKD division only, so a division is open if it holds such activities (an approximation: e.g. 45 holds car repair,
# open, and car trade, not open; 47 food retail and fuels; 49 taxis and heavy freight, only the first open).
# TAX_CARD_DIVISIONS: divisions with activities named in the annexes; TAX_CARD_SOLO_DIVISIONS: divisions open only
# through the catch-all (firms where only the owner works); TAX_CARD_NOT_CIVIL: divisions closed to a civil partnership
# (art. 25 ust. 5: pkt 1-6 only, so not ust. 1a either; parts VIII and IX). 55: annex 3 part XII lp 4, renting rooms
# and holiday cottages (up to 12 rooms), open to natural persons (art. 23 ust. 1a), not to civil partnerships. 49: only
# taxis, goods transport up to 2 t and animal-drawn passenger transport; 50: only inland water transport without engine.
# Closed: wholesale (46), IT and publishing (58-63), finance,
# real estate, professional services other than photography and veterinary, public administration, and the rest of
# the catch-all's exclusions.
TAX_CARD_DIVISIONS = frozenset({"01", "10", "13", "14", "15", "16", "18", "22", "23", "25", "31", "32", "33", "41", "42",
                                "43", "45", "47", "49", "50", "52", "55", "56", "74", "75", "81", "82", "85", "86", "88", "90",
                                "93", "95", "96"})
TAX_CARD_SOLO_DIVISIONS = frozenset({"02", "53", "78", "80"})
TAX_CARD_NOT_CIVIL = frozenset({"55", "75", "85", "86", "88"})
# health care (part VIII) and veterinary services (part IX) under the tax card cover services to consumers only
TAX_CARD_B2C_ONLY = frozenset({"75", "86"})
# the tax-card probability of an open firm is raised so that the EXPECTED share of tax-card firms stays as before:
# definition of the target: forms 099 + 019, size 0-9, active weights > 0 in the GUS scope (gus_dpn) of the population
# file, each firm's drawn age, the normalisation of op._pick included: 0.02697 before and after (test_layer2.py pins it).
# The tax-card mass removed in closed divisions goes proportionally to the other forms there, and the factor thins
# them in open divisions; their overall split is kept in expectation only.
TAX_CARD_FACTOR = 1.4229  # 0.6.15: keeps the expected tax-card share of active natural-person micro firms at 2.70%
# Shares measured by GUS (Dziedzinowa Baza Wiedzy, variable 1220, section 704, 2024: active non-financial
# enterprises by basic legal form x size x form of bookkeeping; data/reference/sources/gus_dbw_2024/).
# Natural persons (JDG; GUS counts civil partnerships with them), (full, kpir, flat rate, tax card):
#   0-9: 15,210 / 1,279,888 / 686,583 / 56,690 of 2,038,372; 10-49: 4,606 / 11,362 / 1,167 / 20 of 17,155;
#   50-249: 956 / 136 / 14 / 0 of 1,106; 250+: 76 of 76 full books.
# The tax card is open only to firms of 6 years or more here: raised to 4.5% for them, so that it stays
# near the measured 2.8% over all ages.
BOOKS_NATURAL = {"0-9": (0.0075, 0.628, 0.337, 0.045), "10-49": (0.268, 0.662, 0.068, 0.0),
                 "50-249": (0.864, 0.123, 0.013, 0.0), "250+": (1.0, 0.0, 0.0, 0.0)}
# Legal persons and units without legal personality: 10,071 micro firms with KPiR and 307 on flat rate
# (of 269,466), 1,357 and 46 of 31,274 with 10-49 persons; only general and professional partnerships of
# natural persons may have them. Their share among active micro firms of these forms is small in the
# population (about 3%), so most of them must keep KPiR to give the GUS count: 90% (the population still
# falls short, a layer-1 signal: dormant companies are scaled down evenly); 10-49: 27% KPiR.
BOOKS_PARTNERSHIP = {"0-9": (0.08, 0.90, 0.02, 0.0), "10-49": (0.72, 0.27, 0.01, 0.0),
                     "50-249": (1.0, 0.0, 0.0, 0.0), "250+": (1.0, 0.0, 0.0, 0.0)}
# A0-27 (0.5.6): having an ERP system is not the same as invoicing in it. Eurostat (isoc_eb_iip 2025,
# Poland) measures firms with ERP software to share information between functional areas: 10-49 31.9%,
# 50-249 68.7%, 250+ 94.1% (no data for micro firms). `has_erp` is true for a firm that invoices in its
# ERP (invoicing_tool "erp", A0-16) and otherwise true with P_ERP_EXTRA, set so that the active firms
# reach the Eurostat shares (scripts/layer2_check.py). How the two go together is an assumption; micro
# firms have an ERP only when they invoice in it. The invoicing tool itself stays as in 0.4 (A0-16).
P_ERP_EXTRA = {"0-9": 0.0, "10-49": 0.07, "50-249": 0.32, "250+": 0.8}
# 0.5.7: with invoices from revenue, A0-16 would put invoicing in an ERP in more firms of 10-49 persons than
# have an ERP at all (Eurostat). A firm invoicing in an ERP has one, so the ERP probability of the tool is
# scaled by ERP_TOOL_FACTOR (the rest moves to an invoicing program), set with P_ERP_EXTRA to the shares:
# active firms 31.5 / 67.2 / 94.8% in 0.6.11 (Eurostat 31.9 / 68.7 / 94.1%).
ERP_TOOL_FACTOR = {"0-9": 1.0, "10-49": 0.74, "50-249": 0.9, "250+": 1.0}


def tool_adjust_for(size: str):
    f = ERP_TOOL_FACTOR[size]

    def adjust(p: np.ndarray) -> np.ndarray:
        p = np.array(p, dtype=float)
        moved = p[2] * (1 - f)
        p[2] -= moved
        p[1] += moved
        return p
    return adjust


# A0-22: the chance that an office keeps the books of a micro firm in which nobody with bank access
# keeps them, by form of bookkeeping (0.4 used 0.8 for every micro firm); larger firms as in 0.4
P_EXTERNAL_MICRO = {"full_books": 0.95, "kpir": 0.80, "flat_rate": 0.55, "tax_card": 0.40}


# A0-25: persons working in a micro firm, "1" (the owner alone) or "2-9". GUS micro firms 2025: 1,916,131
# of 2,460,654 (77.9%) have at most one person working. Rule: a suspended JDG works alone (suspension
# needs no employees); civil and general partnerships have at least two working partners; a firm with an
# employee among its bank users (assistant, dedicated employee, CFO, administrator; an accountant may be
# an outside person) employs; the others work alone with P_SOLO, set so that the active micro firms give
# the GUS share (scripts/layer2_check.py). Same P_SOLO for natural and legal persons: an assumption.
EMPLOYEE_ROLES = {"assistant", "dedicated_employee", "cfo", "administrator"}
# 0.5.5: the chance to work alone depends on the PKD division (Eurostat SBS 2024: share of micro firms with
# 0-1 persons, e.g. 93% in division 62, 53% in 56). For a firm not forced to employ (partnerships,
# employees among bank users): P = min(1, share of the division x SOLO_SCALE). SOLO_SCALE 1.6 brings the
# active micro firms to 76.2% alone (GUS 77.9%; the ceiling with P = 1 everywhere is 77.0%, set by A0-12)
# and keeps the lower shares of the divisions below 62.5% (gastronomy, retail and others).
SOLO_SCALE = 1.6


def p_solo(division: str, section: str) -> float:
    ref = sbs()
    share = (ref["micro"].get(str(division).zfill(2), {}).get("solo_share")
             or ref["micro_solo_share_by_section"].get(section) or ref["micro_solo_share_all"])
    return min(1.0, share * SOLO_SCALE)


def persons_working(firm: dict, authority: dict | None, status: str, u: float) -> str:
    if firm["size_class"] != "0-9":
        return firm["size_class"]
    if status == "suspended":
        return "1"
    if firm["legal_form"] in ("019", "118"):
        return "2-9"
    roles = {r for p in (authority or {}).get("persons", []) for r in p.get("company_roles", [])}
    if roles & EMPLOYEE_ROLES:
        return "2-9"
    return "1" if u < p_solo(firm["pkd_division"], firm["pkd_section"]) else "2-9"


# A0-26: yearly revenue (PLN). Mean by PKD section x size class from GUS (active non-financial enterprises
# 2024, revenue / enterprises), varied inside the section by the SBS ratio of the division's mean turnover
# to its section's; micro firms split by persons working, in proportion to persons per firm (SBS 0-1 vs
# 2-9; an assumption: the same revenue per person). Lognormal with the cell mean kept; the spread is an
# assumption. A suspended firm has no revenue.
REVENUE_SIGMA = {"0-9": 1.1, "10-49": 0.9, "50-249": 0.9, "250+": 1.1}


def revenue_mean(firm: dict, working: str) -> float:
    ref, size = sbs(), firm["size_class"]
    sec = ref["revenue_mean_pln"].get(firm["pkd_section"]) or ref["revenue_mean_pln"]["all"]
    mean = (sec.get(size) or ref["revenue_mean_pln"]["all"][size]) * ref["division_factor"].get(size, {}).get(str(firm["pkd_division"]).zfill(2), 1.0)
    if size == "0-9":
        m = ref["micro"].get(str(firm["pkd_division"]).zfill(2), {"solo_share": ref["micro_solo_share_all"], "p_solo_firm": 1.0, "p_employing_firm": 3.3})
        # 0.5.8: the split keeps the source mean of the cell only with the share of solo firms the generator
        # makes there (scripts/build_layer2_solo.py), not the SBS share (e.g. division 56: 56.7% vs 53.2%)
        gen = generated_solo()
        s_gen = gen.get(str(firm["pkd_division"]).zfill(2), gen.get("all", m["solo_share"]))
        avg_persons = s_gen * m["p_solo_firm"] + (1 - s_gen) * m["p_employing_firm"]
        mean *= (m["p_solo_firm"] if working == "1" else m["p_employing_firm"]) / avg_persons
    return mean


# 0.6.8, corrected in 0.6.10 and 0.6.11: legal revenue limits of the simplified forms. revenue_year_pln is the
# revenue of 2024 (the year of its GUS DBW 2024 calibration), so the forms describe the financial year 2025 and its
# limits apply, converted at the NBP mid rate of the first working day of October of the year before the
# financial year (1.10.2024: 4.2846 PLN/EUR; act on accounting art. 3 ust. 3, flat-rate act art. 4 ust. 2):
# - KPiR (and, as an upper bound only, the tax card): books of account are obligatory from EUR 2.5m of net
#   revenue of the previous year (act on accounting art. 2 ust. 1 pkt 2, EUR 2.5m since the financial year 2025,
#   act of 6.12.2024, Dz.U. 2024 poz. 1863; PIT art. 24a ust. 4): PLN 10,711,500;
# - flat-rate tax (ryczałt ewidencjonowany): at most EUR 2m (flat-rate act art. 6 ust. 4 pkt 1): PLN 8,569,200;
#   a firm starting its business may use it whatever its revenue (art. 6 ust. 4 pkt 2): no truncation under a year.
# The tax card has no revenue limit; its eligibility (continuation since 31.12.2021, the activities of art. 23 and
# the headcount limits) is not modelled beyond A0-21 (known gap). The GUS bookkeeping shares of 2024 were formed
# under the earlier limit (EUR 2m, PLN 9,218,200), and GUS revenue is total revenue, not net sales: both noted.
# A firm on a simplified form draws its revenue from the same lognormal truncated below its own limit, at the
# same rank (u -> u x F(limit)). The revenue cut off belongs to firms on full books in reality (GUS gives no
# revenue by bookkeeping, nor micro revenue by legal form): their revenue is multiplied by
# FULL_BOOKS_REVENUE_FACTOR by size, set in 0.6.8 so that the size totals of active firms stayed as before (calibrated to
# GUS DBW 2024); not re-tuned since: with 0.6.13 (young firms no longer truncated) the annual totals of 0-9 and 10-49
# rose by about 1% (2,156 -> 2,181 and 914 -> 926 bn PLN). An assumption, and keeping totals does not show that the
# split between firms is right
# (scripts/layer2_check.py reports the bank segment with and without it). The bank-segment threshold of EUR 2m
# (A0-42) is a separate, operational assumption, not a tax limit.
REVENUE_YEAR = 2024
# 0.6.13: chronology. The age bands of JDG are as of 11.11.2026 (CEIDG). A firm founded after 2024 had no revenue in
# 2024, and one founded during 2024 had it for part of the year. The founding date is drawn uniformly inside the
# two young bands (stream "founding"); older bands were founded by 11.11.2023 and companies have no drawn age
# (treated as active in all of 2024: an assumption). revenue_annual_pln is the firm's current yearly scale, at the
# GUS 2024 level (it drives invoices, flows and the bank segment); revenue_year_pln is its revenue of 2024 =
# revenue_annual_pln x months_active_2024 / 12, null (not_applicable) when the firm was founded after 2024.
# The legal limits of the simplified forms bind the revenue of the previous year: a firm with no revenue in 2024
# (founded later) may use any form in its first year, so its annual revenue is not truncated.
AGE_REF_DATE = date(2026, 11, 11)
YOUNG_AGE_BANDS = {"<1": (date(2025, 11, 12), date(2026, 11, 11)), "1-2": (date(2023, 11, 12), date(2025, 11, 11))}


def months_active_2024(age: str | None, u: float) -> int:
    """Months of 2024 in which the firm existed (founded in month m of 2024: 13 - m), from a date drawn in its band."""
    if age not in YOUNG_AGE_BANDS:
        return 12
    lo, hi = YOUNG_AGE_BANDS[age]
    d = lo + timedelta(days=min(int(u * ((hi - lo).days + 1)), (hi - lo).days))
    return 0 if d.year > REVENUE_YEAR else 12 if d.year < REVENUE_YEAR else 13 - d.month
EUR_PLN_LIMITS = 4.2846  # NBP table 191/A/NBP/2024 of 1.10.2024, for the financial year 2025
KPIR_LIMIT_PLN = 2_500_000 * EUR_PLN_LIMITS
FLAT_RATE_LIMIT_PLN = 2_000_000 * EUR_PLN_LIMITS
SIMPLIFIED_LIMIT_PLN = {"kpir": KPIR_LIMIT_PLN, "tax_card": KPIR_LIMIT_PLN, "flat_rate": FLAT_RATE_LIMIT_PLN}
FULL_BOOKS_REVENUE_FACTOR = {"0-9": 2.1317, "10-49": 1.1785, "50-249": 1.0175, "250+": 1.0}
USE_FULL_BOOKS_FACTOR = True


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def revenue_cap(books: str, months: int) -> float | None:
    """Highest revenue of 2024 allowed by the form (KPiR and tax card below the limit, flat rate at most the limit),
    None when no limit binds (full books, or no revenue in 2024)."""
    if books == "full_books" or books not in SIMPLIFIED_LIMIT_PLN or months <= 0:
        return None
    limit = SIMPLIFIED_LIMIT_PLN[books]
    return math.floor(limit / 100) * 100 if books == "flat_rate" else math.ceil(limit / 100) * 100 - 100


def revenue_for(firm: dict, working: str, status: str, z: float, books: str = "full_books", months: int = 12) -> int:
    """The firm's annual revenue at its current scale (revenue_annual_pln). On a simplified form with revenue in
    2024 the draw is truncated so that the revenue of 2024 (annual x months / 12) keeps to the form's limit."""
    if status == "suspended":
        return 0
    sigma = REVENUE_SIGMA[firm["size_class"]]
    mu = math.log(revenue_mean(firm, working)) - sigma ** 2 / 2
    cap = revenue_cap(books, months)
    if cap is not None:
        from statistics import NormalDist
        limit = cap * 12 / months
        f_limit = _ncdf((math.log(limit) - mu) / sigma)
        u = min(max(_ncdf(z) * f_limit, 1e-12), 1 - 1e-12)
        z = NormalDist().inv_cdf(u)
        return int(min(round(math.exp(mu + sigma * z), -2), math.floor(limit / 100) * 100))
    factor = FULL_BOOKS_REVENUE_FACTOR[firm["size_class"]] if books == "full_books" and USE_FULL_BOOKS_FACTOR else 1.0
    return int(round(math.exp(mu + sigma * z) * factor, -2))


def revenue_2024(annual: int, books: str, months: int) -> int | None:
    """Revenue of 2024: None for a firm founded after 2024, else the annual revenue for its months, within the cap."""
    if months <= 0:
        return None
    r = int(round(annual * months / 12, -2))
    cap = revenue_cap(books, months)
    return min(r, int(cap)) if cap is not None else r


# A0-28 (0.5.7): invoices from revenue (replaces the A0-15 draw of 0.4). A month of revenue is split into
# sales to firms (each invoiced) and to consumers (invoiced only where an invoice goes with every sale, as
# in utilities and telecoms, or on demand, as in retail and gastronomy, where receipts are the rule).
# issued = revenue/12 x [(1 - b2c) / value_b2b + b2c x B2C_INVOICED[section] / value_b2c];
# received = revenue/12 x COST_SHARE / value_received. Values: medians by size, lognormal spread across
# firms; all assumptions, with one factor INV_SCALE on the values set so that the active firms give the
# monthly KSeF total (scripts/layer2_check.py). Since 0.5.8 only the invoices that go to KSeF are compared
# (A0-29): INV_SCALE 1.75 was set for 95.0 million a month in KSeF at a B2C share in KSeF of 0.5 (KSeF: over 102
# million from about 2.37 million issuers, issuers outside the population included); since the revenue changes of
# 0.6.8-0.6.11 it gives 97.3 million expected (100.0 million drawn) and 132.6 million invoices in all: not re-tuned. A solo firm in a division of professional services works
# for a single client with P_SINGLE_CLIENT: one invoice a month (the LFS 2017 bound: about 9% of the
# self-employed have one client, 6% one dominant client).
INV_VALUE_B2B = {"0-9": 2500, "10-49": 6000, "50-249": 12000, "250+": 20000}
INV_VALUE_B2C = 250
INV_VALUE_RECEIVED = {"0-9": 1500, "10-49": 4000, "50-249": 9000, "250+": 15000}
B2C_INVOICED = {"D": 1.0, "E": 1.0, "J": 0.5, "L": 0.6, "H": 0.3, "N": 0.3, "P": 0.3, "Q": 0.2, "R": 0.2, "S": 0.2,
                "G": 0.05, "I": 0.05}
B2C_INVOICED_DEFAULT = 0.2
COST_SHARE = 0.6
INV_SIGMA = 0.8
INV_SCALE = 1.75
SINGLE_CLIENT_DIVISIONS = {"62", "69", "70", "71", "74", "86"}
P_SINGLE_CLIENT = 0.4


# A0-29 (0.5.8): invoices in KSeF are not all invoices. B2B invoices go to KSeF (obligatory from 2026),
# except that until the end of 2026 a taxpayer whose sales documented by invoices are at most 10,000 PLN
# gross a month may issue them outside KSeF: half of such firms do (assumption). Invoices to consumers
# may go to KSeF but need not: a firm puts them there with B2C_IN_KSEF (assumption, 0.5; the check also
# shows 0.2 and 0.9 and the INV_SCALE each would need). Only the KSeF count is compared with the KSeF
# total.
B2C_IN_KSEF = 0.5
SMALL_SELLER_GROSS = 10_000
P_SMALL_OUTSIDE = 0.5


def invoices_for(firm: dict, revenue: int, working: str, z: np.ndarray, parts: dict):
    """A function of the B2C share giving (issued, received) a month, for operational.draw_layer2; it
    also fills `parts` with the B2B and B2C counts and what goes to KSeF."""
    size, month = firm["size_class"], revenue / 12

    def counts(b2c: float) -> tuple[int, int]:
        parts.update(b2b=0, b2c=0, ksef=0, b2c_in_ksef=False, single_client=False)
        if revenue <= 0:
            return 0, 1
        q = B2C_INVOICED.get(firm["pkd_section"], B2C_INVOICED_DEFAULT)
        if working == "1" and str(firm["pkd_division"]).zfill(2) in SINGLE_CLIENT_DIVISIONS and z[3] < P_SINGLE_CLIENT and b2c < 0.2:
            n_b2b, n_b2c = 1, 0
            parts["single_client"] = True
        else:
            v_b2b = INV_VALUE_B2B[size] * INV_SCALE * math.exp(INV_SIGMA * z[0])
            v_b2c = INV_VALUE_B2C * INV_SCALE * math.exp(INV_SIGMA * z[1])
            n_b2b = int(round(month * (1 - b2c) / v_b2b))
            n_b2c = int(round(month * b2c * q / v_b2c))
        small = month * ((1 - b2c) + b2c * q) * 1.23 <= SMALL_SELLER_GROSS
        b2b_ksef = 0 if small and z[4] < P_SMALL_OUTSIDE else n_b2b
        b2c_in = bool(z[5] < B2C_IN_KSEF)
        parts.update(b2b=n_b2b, b2c=n_b2c, ksef=b2b_ksef + (n_b2c if b2c_in else 0), b2c_in_ksef=b2c_in)
        v_rec = INV_VALUE_RECEIVED[size] * INV_SCALE * math.exp(INV_SIGMA * z[2])
        return n_b2b + n_b2c, max(1, int(round(month * COST_SHARE / v_rec)))
    return counts


# A0-30 (0.5.9; 0.6.6: 30,873 companies): ownership of KRS companies from the BIR (REGON) sample of active KRS companies
# (layer2_ownership.json): groups domestic persons, domestic entities (usually a group), foreign, public,
# mixed. Micro firms: shares of their legal form. Limited liability companies with more persons: the shares
# of companies with a larger share capital, as a proxy of size (the population has no capital; foreign
# ownership rises with capital from 14% to 35%): an assumption. The register's empty ownership form
# (mostly young companies) is left out of the shares. JDG and civil partnerships are owned by natural
# persons by definition; a branch of a foreign entrepreneur is foreign.
# 0.6.12: for 50+ persons a company of the forms in layer2_foreign_gus.json (joint-stock, limited liability,
# limited partnership, partnership limited by shares) is majority-foreign with p = min(p_max, k(size, section) x
# relative risk of the form), k set so that the expected count of the population equals the estimated GUS count
# of majority-foreign enterprises (2024, by PKD section and size class; scripts/build_layer2_foreign.py). Foreign
# or not is drawn from a second uniform of the same stream; the other groups keep their BIR shares. Before 0.6.12
# the capital proxy gave 250+ about half the GUS count and a section mix unlike GUS.
OWNERSHIP_GROUPS = ("domestic_persons", "domestic_entities", "foreign", "public", "mixed_no_majority")


def ownership_for(firm: dict, u: float, u_foreign: float | None = None) -> str:
    form, size = firm["legal_form"], firm["size_class"]
    if form in ("099", "019"):
        return "natural_person_owner"
    if form == "179":
        return "foreign"
    ref = ownership_ref()
    cell = (ref["llc_by_size_proxy"].get(size) if form == "117" and size != "0-9" else None) \
        or ref["by_form"].get(form) or ref["by_form"]["117"]
    p = dict(cell["p"])
    fg = foreign_gus()
    by_size = fg["by_size"].get(size, {})
    if form in fg["foreign_forms"] and "k" in by_size:
        # 0.6.12: foreign or not from its own uniform, so that the split of the other groups does not move with it
        k = by_size["k"].get(firm.get("pkd_section"), by_size["k"]["_default"])
        pf = min(fg["p_max"], k * fg["relative_risk_by_form"][form])
        if u_foreign is None:  # one uniform only: split it (foreign below pf, the rest rescaled above)
            if u < pf:
                return "foreign"
            u, u_foreign = (u - pf) / (1 - pf), 1.0
        if u_foreign < pf:
            return "foreign"
        p = {g: v for g, v in p.items() if g != "foreign"}
        if not p:
            return "foreign"
    keys = [k for k in OWNERSHIP_GROUPS if k in p]
    return op._pick(tuple(keys), np.array([p[k] for k in keys]), u)


# A0-31 (0.5.10): VAT status of KRS companies from WL2 (layer2_vat_status.json; random sample of active KRS
# companies): active, exempt, not in the register, deregistered. Micro firms: legal form x PKD division
# where the cell has 100 answers, else legal form. Larger limited liability companies: the shares of
# companies with a larger share capital, as a proxy of size (VAT-active rises from 52% at 5k PLN of capital
# to 90% at 5m and more): an assumption. JDG and civil partnerships: not measured yet (the white-list
# sample of JDG will tell). calibrate_active.py uses the status to tell dormant companies apart.
VAT_STATUSES = ("active", "exempt", "not_in_register", "not_registered")


# A0-32 (0.6.0): financing products USED in the past six months (SAFE Q4 = 1), drawn jointly, so that
# products used together stay together: the combination of credit line/overdraft, bank loan, trade credit,
# leasing and factoring from the firm-weighted SAFE shares of the firm's size class (combinations under 10
# firms pooled as 'other', drawn product by product from the shares inside it). SAFE covers firms with
# employees only: a firm working alone uses them with SOLO_USE_FACTOR of that chance (an assumption;
# variants 0.25 and 0.75); a suspended firm uses none. Unmet need for a bank loan and for a credit line
# (discouraged, limited amount, refused because of the cost, rejected) from SAFE, apart from use. Holding a
# product (WBES: loan or line at the time of the survey) is not modelled.
SAFE_SIZE = {"0-9": "1-9", "10-49": "10-49", "50-249": "50-249", "250+": "250+"}
SOLO_USE_FACTOR = 0.5
# factoring of solo firms tuned to the GUS count of factoring clients (31,709 in 2025, all firms): a solo
# firm drawn to use factoring keeps it with this chance (total about 32 thousand instead of 44)
SOLO_FACTORING_KEEP = 0.46


@functools.lru_cache(maxsize=1)
def products_ref() -> dict:
    return json.loads(PRODUCTS.read_text(encoding="utf-8"))


def products_for(firm: dict, working: str, status: str, u: np.ndarray) -> dict:
    ref = products_ref()
    names = ref["products"]
    none = {"products_used_6m": {p: False for p in names}, "unmet_need_bank_loan": False, "unmet_need_credit_line": False}
    if status == "suspended":
        return none
    size = SAFE_SIZE[firm["size_class"]]
    unmet = {k: bool(u[k_i] < ref["unmet_need"][k][size]) for k_i, k in ((1, "unmet_bank_loan"), (2, "unmet_credit_line"))}
    out = {"unmet_need_bank_loan": unmet["unmet_bank_loan"], "unmet_need_credit_line": unmet["unmet_credit_line"]}
    if working == "1" and u[3] >= SOLO_USE_FACTOR:
        return {**out, "products_used_6m": {p: False for p in names}}
    j = ref["joint_use_6m"][size]
    keys = list(j["combinations"]) + (["other"] if "other" in j else [])
    p = np.array([j["combinations"][k] for k in j["combinations"]] + ([j["other"]["pct"]] if "other" in j else []))
    pick = op._pick(tuple(keys), p, u[0])
    if pick == "other":
        inside = j["other"]["product_shares_inside"]
        used = {n: bool(u[4 + i] < inside[n]) for i, n in enumerate(names)}
    else:
        used = {n: pick[i] == "1" for i, n in enumerate(names)}
    if working == "1" and used.get("factoring") and u[9] >= SOLO_FACTORING_KEEP:
        used["factoring"] = False
    return {**out, "products_used_6m": used}


# A0-33 (0.6.1): three payment features kept apart. accepts_cards: card acceptance (terminal or online), more
# likely with more sales to consumers, P = min(0.98, CARD_SCALE x (CARD_BASE + b2c)); CARD_SCALE set so that
# the active firms accepting cards match the NBP count of acceptors (651,771, 2026Q2). receives_transfers: a
# firm with sales to firms or invoices; a firm selling only to consumers with P_B2C_TRANSFERS.
# e_payment_share_of_sales (value): sales to firms paid electronically at E_B2B, sales to consumers at
# E_B2C_CARDS with card acceptance, else E_B2C_NO_CARDS (online transfers, BLIK, cash on the rest); checked
# against WBES (fin28, firms of 5+ persons). All parameters are assumptions anchored on these totals.
CARD_BASE, CARD_SCALE = 0.03, 0.855  # 0.6.17: re-set to the NBP count after the B2C share by division (was 1.0)
P_B2C_TRANSFERS = 0.5
E_B2B = {"0-9": 0.82, "10-49": 0.88, "50-249": 0.95, "250+": 0.98}
E_B2C_CARDS, E_B2C_NO_CARDS = 0.75, 0.15


def payments_for(firm: dict, out: dict, status: str, u: np.ndarray) -> dict:
    if status == "suspended":
        return {"accepts_cards": False, "receives_transfers": False, "e_payment_share_of_sales": 0.0}
    b2c = out["b2c_share_of_sales"]
    cards = bool(u[0] < min(0.98, CARD_SCALE * (CARD_BASE + b2c)))
    transfers = bool(b2c < 1 or out["invoices_issued_month"] > 0 or u[1] < P_B2C_TRANSFERS)
    share = (1 - b2c) * E_B2B[firm["size_class"]] + b2c * (E_B2C_CARDS if cards else E_B2C_NO_CARDS)
    return {"accepts_cards": cards, "receives_transfers": transfers, "e_payment_share_of_sales": round(share, 3)}


# A0-34 (0.6.2): declared liquidity, the answer to the PIE monthly survey question (MIK, question 7): is the
# firm's liquidity enough to keep running for more than 3 months, 2-3 months, 1 month, or not even a month.
# Drawn by size class from the mean of the last 12 months (scripts/build_layer2_liquidity.py), independent of
# sector, payments, products and counterparties: how liquidity goes with counterparty concentration and
# payment delays is not measured and stays an assumption (independence). A declared continuity of
# operation, not cash at zero inflows. A suspended firm has none (None).
LIQUIDITY_CATS = ("over_3_months", "2_3_months", "1_month", "insufficient_1_month")


@functools.lru_cache(maxsize=1)
def liquidity_ref() -> dict:
    return json.loads(LIQUIDITY.read_text(encoding="utf-8"))


def liquidity_for(firm: dict, status: str, u: float) -> str | None:
    if status == "suspended":
        return None
    p = liquidity_ref()["by_size"][firm["size_class"]]
    return op._pick(LIQUIDITY_CATS, np.array([p[c] for c in LIQUIDITY_CATS]), u)


# A0-35..A0-37 (0.6.3, EXPERIMENTAL): counterparties, payment terms and amounts not settled on time. Mostly
# assumptions (Z); results that depend on them are not estimates for Polish firms. Each field has its own
# uniform number; `dependence` couples some of them through a Gaussian copula, which keeps the theoretical
# distribution of every field (given size, sector, B2B share and the logical bounds) the same in every variant,
# not the shares in a generated sample (scripts/profile_check.py reports the weighted differences):
# - "independent" (default, the reference): conditionally independent given size, sector, B2B share and
#   the bounds below; not "the most realistic" one;
# - "mrit_rank": overdue receivables and payables linked with a normal correlation RHO_MRIT, the Pearson
#   value of a Spearman rank correlation 0.42 (MRiT 2025, large reporters, scripts/mrit_delays.py); for
#   firms below 250 persons a further assumption;
# - "stress_coupled": as mrit_rank, plus one common factor (loading STRESS_LOADING) on the concentration of
#   B2B sales, the overdue receivables and declared liquidity (worse liquidity with more concentration and
#   more overdue receivables): a sensitivity scenario, its strength an assumption.
DEPENDENCE = ("independent", "mrit_rank", "stress_coupled")
RHO_MRIT = 2 * math.sin(math.pi * 0.42 / 6)
STRESS_LOADING = 0.5
CLIENT_BANDS = ("0", "1", "2", "3-5", "6-20", "21-100", "101+")
CONCENTRATION = ("<25%", "25-<50%", "50-<75%", ">=75%")
TERMS = ("0", "1-7", "8-14", "15-30", "31-60", "61+")
OVERDUE = ("0", "(0,5%]", "(5,15%]", "(15,30%]", ">30%")
# A0-35: distinct B2B clients in 12 months. Median (12 x B2B invoices a month) ** CLIENTS_EXP, lognormal
# spread CLIENTS_SIGMA; bounded below by the B2B part of the active customers of a month and above by the B2B
# invoices of a year (each client gets an invoice); a firm without B2B sales has 0, one with B2B sales too
# small for a monthly invoice 1-6; the single-client rule of A0-28 gives 1. Largest client's share of B2B
# sales: 1/n + (1 - 1/n) x u ** CONC_K[size] (one client: 100%, two: at least 50%), more concentrated in the
# divisions of A0-28 (k x 0.7). Checked loosely against LFS 2017 (self-employed, all clients, dominant = 75%
# of income): a transfer D+Z, not fitted.
CLIENTS_EXP, CLIENTS_SIGMA = 0.55, 0.7
CONC_K = {"0-9": 2.0, "10-49": 3.0, "50-249": 4.0, "250+": 6.0}
# A0-36: typical contractual term of B2B sales and of purchases (bands of days; 0 = payment at once), by
# size, all assumptions bounded by law (B2B 60 days unless agreed and not grossly unfair; a large buyer
# from an SME at most 60; a public buyer 30): small buyers get shorter terms, construction, manufacturing
# and wholesale give longer ones (TERMS_LONGER of the mass moved to 31-60).
TERMS_SALES = {"0-9": (0.12, 0.18, 0.30, 0.30, 0.09, 0.01), "10-49": (0.05, 0.10, 0.25, 0.38, 0.20, 0.02),
               "50-249": (0.03, 0.06, 0.18, 0.40, 0.29, 0.04), "250+": (0.02, 0.04, 0.12, 0.38, 0.38, 0.06)}
TERMS_PURCHASES = {"0-9": (0.22, 0.22, 0.27, 0.24, 0.045, 0.005), "10-49": (0.10, 0.15, 0.30, 0.33, 0.11, 0.01),
                   "50-249": (0.05, 0.08, 0.22, 0.40, 0.23, 0.02), "250+": (0.02, 0.04, 0.12, 0.37, 0.40, 0.05)}
TERMS_LONGER_SECTIONS, TERMS_LONGER = {"C", "F"}, 0.10
TERMS_LONGER_DIVISIONS = {"46"}
# A0-37 (definition fixed in 0.6.15, audit 13.11.2026): of the B2B sales invoices (receivables; None without B2B
# clients in the last 12 months: nothing invoiced to firms in the window) or of the purchase invoices (payables) issued
# in the last 12 months whose due date has passed, the share of value paid late or still unpaid. Invoices not yet due
# do not count. Payment terms and this share are drawn independently: terms of 0 days mean due on the invoice date,
# so a large overdue share is possible with them. 250+: transfer
# from MRiT 2025 large reporters (rows and columns of the published joint table; the merged band 0 + (0,5%] split by
# assumption); a transfer between different measures (MRiT counts payments due in the reporting year, also from
# earlier invoices, and older unpaid amounts; the field counts invoices of the last 12 months): the direction of
# the bias is not known (audit of 0.6.16). Smaller firms: assumptions, more mass at 0 and at the extremes
# with fewer counterparties.
OVERDUE_REC = {"0-9": (0.30, 0.12, 0.16, 0.17, 0.25), "10-49": (0.12, 0.14, 0.20, 0.22, 0.32),
               "50-249": (0.06, 0.11, 0.17, 0.23, 0.43), "250+": (0.02, 0.098, 0.112, 0.224, 0.546)}
OVERDUE_PAY = {"0-9": (0.40, 0.15, 0.17, 0.13, 0.15), "10-49": (0.20, 0.20, 0.22, 0.17, 0.21),
               "50-249": (0.10, 0.18, 0.23, 0.20, 0.29), "250+": (0.04, 0.208, 0.225, 0.204, 0.323)}


def _band_clients(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3-5" if n <= 5 else "6-20" if n <= 20 \
        else "21-100" if n <= 100 else "101+"


def _band_share(s: float) -> str:
    return CONCENTRATION[min(3, int(s * 4 + 1e-12))] if s < 1 else ">=75%"


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _z(u: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(min(max(u, 1e-12), 1 - 1e-12))


def coupled_uniforms(firm: dict, dependence: str = "independent", replicate: int = 0) -> tuple[float, float, float, float]:
    """The uniform numbers of concentration, overdue receivables, overdue payables and declared liquidity: as drawn
    in 'independent' (no round trip through the normal), coupled through the Gaussian copula in the other variants.
    replicate > 0 (draw_population only): the common factor and the noise of the coupled scores come from the stream
    'coupling-<r>' instead of the firm's own numbers; the receivables of mrit_rank (not re-assigned) stay the firm's."""
    seed = firm["firm_seed"]
    u = rng_for(seed, "counterparties").random(7)
    u_liq = rng_for(seed, "liquidity").random()
    if replicate and dependence != "independent":
        r = rng_for(seed, f"coupling-{replicate}").random(4)
        u = u.copy()
        u[1], u[5], u[6] = r[0], r[1], r[2]
        u_liq = r[3]
        if dependence == "stress_coupled":
            u[4] = rng_for(seed, f"coupling-{replicate}-rec").random()
    v_conc, v_rec, v_pay, v_liq = u[1], u[4], u[5], u_liq
    if dependence != "independent":
        z_rec, z_pay = _z(u[4]), _z(u[5])
        if dependence == "stress_coupled":
            f, c = _z(u[6]), math.sqrt(1 - STRESS_LOADING ** 2)
            z_rec = STRESS_LOADING * f + c * z_rec
            v_conc, v_liq = _phi(STRESS_LOADING * f + c * _z(u[1])), _phi(STRESS_LOADING * f + c * _z(u_liq))
            v_rec = _phi(z_rec)
        v_pay = _phi(RHO_MRIT * z_rec + math.sqrt(1 - RHO_MRIT ** 2) * z_pay)
    return v_conc, v_rec, v_pay, v_liq


# 0.6.15 (audit 13.11.2026): variants for comparing agents keep the generated shares exactly. draw_population()
# draws every firm in 'independent' and, for a coupled variant, re-assigns within each cell the values of the coupled
# fields by rank: the firm with the k-th smallest coupled uniform gets the k-th smallest value of the cell in the
# independent variant (category order; ties by firm seed). A cell is size x PKD section x weight class (the active
# weight, so that weighted shares stay as well; a deviation from the pre-registered size x section, declared) x, for
# concentration, the band of B2B clients where it bounds the share (1, 2, 3-5; 6+ pooled); only non-null values. Every
# share over groups made of whole cells is then identical to 'independent'; joint distributions with fields outside
# the cell (terms, client band 6+, age, persons) are not kept, and the dependence holds within cells only (weaker in
# small ones; profile_check reports the achieved coupling with bootstrap intervals). replicate > 0 re-draws the noise
# of the coupled scores, so that agent results can be averaged over several assignments. The per-firm
# draw(..., dependence=...) keeps the theoretical distribution only, not the sample shares.
COUPLED_FIELDS = {"mrit_rank": ("overdue_payables_share",),
                  "stress_coupled": ("largest_b2b_client_share", "overdue_receivables_share", "overdue_payables_share",
                                     "liquidity_declared")}
_FIELD_ORDER = {"largest_b2b_client_share": "CONCENTRATION", "overdue_receivables_share": "OVERDUE",
                "overdue_payables_share": "OVERDUE", "liquidity_declared": "LIQUIDITY_CATS"}
_FIELD_UNIFORM = {"largest_b2b_client_share": 0, "overdue_receivables_share": 1, "overdue_payables_share": 2,
                  "liquidity_declared": 3}


def draw_population(firms: list[dict], authorities: dict | None, dependence: str, weight_class: dict,
                    replicate: int = 0, base: list[dict] | None = None) -> list[dict]:
    """Profiles of a population of firms (the order of `firms` does not matter); `authorities` by firm_id;
    `weight_class` by firm_id (e.g. round(weight_active, 6)), part of the cell. 'independent' is draw() of each firm; a
    coupled variant is 'independent' with the values of COUPLED_FIELDS re-assigned by rank inside each cell (see above).
    A firm's coupled profile depends on the whole set of firms and on the weight classes: draw cases on the same
    population (the active firms in the GUS scope with their weight classes) and select them afterwards. The
    operational variant is always 'base'. Shares are identical to 'independent' only over groups made of whole cells."""
    if dependence not in DEPENDENCE:
        raise ValueError(f"dependence {dependence!r}, expected one of {DEPENDENCE}")
    missing = [f.get("firm_id") for f in firms if f.get("firm_id") not in weight_class]
    if missing:
        raise ValueError(f"weight_class missing for {len(missing)} firms: without it the cells would ignore the weights")
    authorities = authorities or {}
    if base is None:  # the 'independent' profiles of the same firms may be passed in (same order) to save drawing them
        base = [draw(f, authorities.get(f.get("firm_id"))) for f in firms]
    if dependence == "independent":
        return base
    out = [{**p, "dependence": dependence} for p in base]
    for field in COUPLED_FIELDS[dependence]:
        order = {v: i for i, v in enumerate(globals()[_FIELD_ORDER[field]])}
        cells: dict = {}
        for i, (f, p) in enumerate(zip(firms, base)):
            if p.get(field) is None:
                continue
            key = (f["size_class"], f["pkd_section"], weight_class[f["firm_id"]])
            if field == "largest_b2b_client_share":  # bounds: 1 client 100%, 2 at least 50%, 3-5 at least 20%; 6+ free
                band = p.get("b2b_clients_year")
                key += (band if band in ("1", "2", "3-5") else "6+",)
            cells.setdefault(key, []).append(i)
        for idx in cells.values():
            values = sorted((order[base[i][field]], int(firms[i]["firm_seed"]), base[i][field]) for i in idx)
            ranked = sorted(idx, key=lambda i: (coupled_uniforms(firms[i], dependence, replicate)[_FIELD_UNIFORM[field]],
                                                int(firms[i]["firm_seed"])))
            for i, (_, _, v) in zip(ranked, values):
                out[i][field] = v
    for i, (f, p) in enumerate(zip(firms, out)):
        p["null_reasons"] = null_reasons(p, p["activity_status"], f)
    return out


# 0.6.17 (blind checks of the B-K2 cards): the B2C share of sales by PKD division where a section mixes divisions that
# sell to different buyers. Section G (A0-14) gave wholesale (46), retail (47) and motor trade (45) one mixture, so
# 44% of retail firms sold at most 10% to consumers. By the PKD definitions, retail (47) sells goods to households for
# personal use (to the general public: "ogółowi społeczeństwa") and wholesale (46) to retailers, firms and institutions;
# the weights are assumptions on that basis (none, 0-20%, 20-80%, 80-100%); motor trade (45) keeps the mixture of
# section G. The band is picked from the same uniform and the base stream is consumed as before (op.draw_layer2), so
# only the B2C share and what follows from it change.
B2C_BY_DIVISION = {"47": (0.02, 0.08, 0.25, 0.65), "46": (0.60, 0.30, 0.08, 0.02),
                   # review of 0.6.17: veterinary (75) serves mostly animal owners, partly farms (and its tax card is
                   # consumer-only, TAX_CARD_B2C_ONLY); travel agencies (79) sell mostly to consumers; specialised
                   # construction (43: installation and finishing) works for households more than 41-42 do
                   "75": (0.05, 0.10, 0.30, 0.55), "79": (0.05, 0.10, 0.35, 0.50), "43": (0.30, 0.25, 0.30, 0.15)}


def counterparties_for(firm: dict, out: dict, status: str, parts: dict, dependence: str = "independent") -> dict:
    """The six experimental fields (A0-35..A0-37) and declared liquidity (A0-34, coupled only in
    stress_coupled), from the firm's own uniform numbers."""
    if dependence not in DEPENDENCE:
        raise ValueError(f"dependence {dependence!r}, expected one of {DEPENDENCE}")
    seed, size = firm["firm_seed"], firm["size_class"]
    u = rng_for(seed, "counterparties").random(7)  # clients, concentration, 2 terms, receivables, payables, factor
    v_conc, v_rec, v_pay, v_liq = coupled_uniforms(firm, dependence)
    res = {"liquidity_declared": liquidity_for(firm, status, v_liq)}
    div, sec = str(firm["pkd_division"]).zfill(2), firm["pkd_section"]
    longer = sec in TERMS_LONGER_SECTIONS or div in TERMS_LONGER_DIVISIONS
    p_pur = op._shift(TERMS_PURCHASES[size], 4, TERMS_LONGER) if longer else np.array(TERMS_PURCHASES[size])
    res["payment_terms_purchases_days"] = op._pick(TERMS, p_pur, u[3])
    res["overdue_payables_share"] = op._pick(OVERDUE, np.array(OVERDUE_PAY[size]), v_pay)
    if status == "suspended":  # 0.6.11: the past 12 months of a suspended firm are not known, sales and purchases alike
        return {**res, "b2b_clients_year": None, "largest_b2b_client_share": None, "payment_terms_sales_days": None,
                "overdue_receivables_share": None, "payment_terms_purchases_days": None, "overdue_payables_share": None}
    b2b_month = parts.get("b2b", 0)
    if out["b2c_share_of_sales"] >= 1:
        n = 0
    elif parts.get("single_client"):
        n = 1
    else:
        lower = max(1, out["customers_active_month"] - parts.get("b2c", 0))
        upper = 12 * b2b_month if b2b_month > 0 else 6
        med = max(1.0, (12 * max(b2b_month, 0.25)) ** CLIENTS_EXP)
        n = int(min(upper, max(lower, round(math.exp(math.log(med) + CLIENTS_SIGMA * _z(u[0]))))))
    res["b2b_clients_year"] = _band_clients(n)
    if n == 0:
        return {**res, "largest_b2b_client_share": None, "payment_terms_sales_days": None, "overdue_receivables_share": None}
    k = CONC_K[size] * (0.7 if div in SINGLE_CLIENT_DIVISIONS else 1.0)
    share = 1.0 if n == 1 else 1 / n + (1 - 1 / n) * v_conc ** k
    res["largest_b2b_client_share"] = _band_share(share)
    p_sal = op._shift(TERMS_SALES[size], 4, TERMS_LONGER) if longer else np.array(TERMS_SALES[size])
    res["payment_terms_sales_days"] = op._pick(TERMS, p_sal, u[2])
    res["overdue_receivables_share"] = op._pick(OVERDUE, np.array(OVERDUE_REC[size]), v_rec)
    return res


# A0-38..A0-39 (0.6.4, EXPERIMENTAL): the firm's banking relationship. Results that depend on these fields are
# not estimates for Polish firms (sources and gaps: layer2_anchors.json, business_banking).
# A0-38: separate business account and mixing of private and business money (two measures, not merged into
# disjoint "business only / private only / both", which nobody has published).
# - has_separate_business_account: companies and partnerships True; a JDG has none with
#   P = min(0.95, SEP_SCALE x weight), weight by persons and revenue (a solo JDG with little revenue most
#   often runs the business through its private account; an assumption; 0 for 50+ persons and for revenue at or
#   above the EUR 2m bank threshold, 0.6.11), SEP_SCALE set so that active JDG give the 20% of the self-employed
#   without a separate account of SME Finance Tracking 2026 (press only). Law: the payment-account rule for B2B
#   payments over PLN 15,000 (Prawo przedsiębiorców art. 19) is met by any payment account, a private ROR
#   included; but a ROR is never on the VAT white list (VAT act art. 96b ust. 3 pkt 13), so a VAT-active payee
#   exposes its buyers to the white-list sanctions (PIT art. 22p, CIT art. 15d, Ordynacja art. 117ba): the
#   pressure for a separate account falls on VAT-active JDG with large B2B payments (flag in profile_check).
# - mixes_private_and_business: True for a JDG without a separate account; otherwise, for firms run by natural
#   persons only (JDG, partnerships of natural persons, companies owned by domestic persons; 0.6.11), P =
#   MIX_GIVEN_SEPARATE by size, set so that active firms give BIG InfoMonitor 2016 (micro 51-59%, small 21-25%,
#   medium almost none; the 51/59 split is by clients paying late, not a range); 250+ an assumption (none).
SEP_WEIGHT = {"solo_low": 3.0, "solo_mid": 1.5, "solo_high": 0.5, "employing": 0.3}
SEP_SCALE = 0.13
MIX_GIVEN_SEPARATE = {"0-9": 0.48, "10-49": 0.27, "50-249": 0.047, "250+": 0.0}
# A0-39: banks and the main bank.
# - banks_count (banks with accounts used by the business; status transfer): companies from the VAT white list
#   by legal form (10+ limited liability companies by capital as a size proxy); JDG transferred from MGBI 2025 (white
#   list of ALL VAT taxpayers, not a JDG measurement; its '3+ 2%' split into 3: 2%, 4+: 0% is an assumption), SME
#   Finance Tracking 2026 as the variant "survey" (JDG_BANKS).
# - main bank: the bank of the main current account, through which the revenue comes in (assumption; the
#   INDICATOR 2015 definition). changed_main_bank_since_start: INDICATOR/WIB 2015 (turnover up to PLN 3.6m 30%,
#   3.6-30m 38%) mapped to size, scaled by the firm's age where it is known (CHANGED_AGE, normalised to about
#   1 over JDG ages); a firm under a year old has not changed.
# - owner_private_account_same_bank: owner's private account at the main bank, for firms run by natural
#   persons (run_by_natural_persons, 0.6.16) up to 49 persons: INDICATOR/WIB 2015 85% (up to PLN 3.6m) and 72%
#   (3.6-30m), mapped to 0-9 and 10-49; a JDG without a separate account True by definition; 50+ None (not
#   measured), others None (not run by natural persons).
JDG_BANKS = "base"
CHANGED_BASE = {"0-9": 0.30, "10-49": 0.34, "50-249": 0.38, "250+": 0.38}
CHANGED_AGE = {"<1": 0.0, "1-2": 0.4, "3-5": 0.8, "6-10": 1.1, "11-20": 1.4, "21+": 1.6}
SAME_BANK = {"0-9": 0.85, "10-49": 0.72}
BANK_BANDS = ("1", "2", "3", "4+")


@functools.lru_cache(maxsize=1)
def banking_ref() -> dict:
    return json.loads(BANKING.read_text(encoding="utf-8"))


def _sep_weight(working: str, revenue: int, size: str = "0-9") -> float:
    if size not in ("0-9", "10-49") or revenue >= SEGMENT_SIMPLIFIED_PLN:  # 0.6.11: not for 50+ persons nor large revenue
        return 0.0
    if working != "1":
        return SEP_WEIGHT["employing"]
    return SEP_WEIGHT["solo_low"] if revenue < 100_000 else SEP_WEIGHT["solo_mid"] if revenue < 1_000_000 else SEP_WEIGHT["solo_high"]


def run_by_natural_persons(form: str, ownership: str | None) -> bool:
    """0.6.16 (audit of 0.6.14): one rule for every field about the owners' private money and its null reasons, the
    same as for the bookkeeping of general partnerships: JDG and civil partnerships (natural persons in the
    population), professional partnerships (natural persons by law, KSH art. 87), general partnerships whose owners
    are domestic or foreign persons (foreign assumed natural persons, as in bookkeeping_for), and companies owned by
    domestic persons."""
    if form in ("099", "019", "115"):
        return True
    if form == "118":
        return ownership in (None, "domestic_persons", "foreign")
    return ownership == "domestic_persons"


def banking_for(firm: dict, out: dict, age: str | None, working: str, revenue: int, ownership: str, u: np.ndarray) -> dict:
    form, size = firm["legal_form"], firm["size_class"]
    jdg = form == "099"
    separate = not jdg or u[0] >= min(0.95, SEP_SCALE * _sep_weight(working, revenue, size))
    # 0.6.11: private money is mixed only where natural persons run the firm (JDG, partnerships of natural persons,
    # companies owned by domestic persons)
    run_by_persons = run_by_natural_persons(form, ownership)
    mixes = (not separate) or (run_by_persons and bool(u[1] < MIX_GIVEN_SEPARATE[size]))
    ref = banking_ref()
    if jdg:
        p = ref["jdg"]["base" if JDG_BANKS == "base" else "variant_survey"]["p"]
    elif form == "117" and size != "0-9":
        p = ref["llc_by_size_proxy"][size]["p"]
    else:
        p = (ref["by_form"].get(form) or ref["all_companies"])["p"]
    banks = op._pick(BANK_BANDS, np.array([p[b] for b in BANK_BANDS]), u[2])
    p_changed = 0.0 if age == "<1" else min(0.95, CHANGED_BASE[size] * (CHANGED_AGE[age] if age else 1.0))
    # 0.6.16 (review): one rule for every form run by natural persons: measured up to 49 persons; 50+ unknown, except a
    # JDG without a separate account (its private account is the business account, True by definition)
    personal = run_by_persons and (size in SAME_BANK or not separate)
    same = (True if not separate else bool(u[4] < SAME_BANK.get(size, SAME_BANK["10-49"]))) if personal else None
    return {"has_separate_business_account": bool(separate), "mixes_private_and_business": bool(mixes), "banks_count": banks,
            "changed_main_bank_since_start": bool(u[3] < p_changed), "owner_private_account_same_bank": same}


# A0-40..A0-41 (0.6.5, EXPERIMENTAL): channels and frequency of business banking. Mostly assumptions; results
# that depend on them are not estimates for Polish firms.
# A0-40: primary_channel (web on a computer / mobile_app / branch / h2h_api: host-to-host or API from an ERP),
# by size; anchored on the channel of a one-off transfer of small firms (Mastercard / Minds & Roses 2024,
# n=1003: internet 74%, app 16%, branch 9%, phone 1%) and on the primary channel of SMEs in 2015 (UG,
# Finanse MSP: internet 67-71%, branch 26-30%, mobile ~0%), moved toward the app since; h2h_api only for
# firms with an ERP (has_erp), share an assumption (no source); a JDG without a separate account banks in
# its private (retail) app more often (APP_SHIFT_PRIVATE). uses_mobile_app: True for the app as primary
# channel, otherwise P by size, between the panel of entrepreneurs (about one in three run the business in
# the bank's app, 2023) and the client-sample bounds of banks' reports (44% and 96% of corporate clients).
# uses_branch_12m: True for the branch as primary channel, otherwise P by size (Mastercard 2024: 334 of
# 1003 small firms handle some matters in a branch). bank_operator: who uses the business banking, derived
# from the persons with access (authority): owner or board only, in-house finance staff, the accounting
# office (or an outside accountant), or office and in-house staff (mixed); in a firm where only the owner works,
# a finance person with access counts as outside (0.6.11).
CHANNELS = ("web", "mobile_app", "branch", "h2h_api")
PRIMARY_CHANNEL = {"0-9": (0.68, 0.22, 0.10, 0.0), "10-49": (0.80, 0.12, 0.07, 0.01),
                   "50-249": (0.82, 0.07, 0.04, 0.07), "250+": (0.70, 0.04, 0.02, 0.24)}
APP_SHIFT_PRIVATE = 0.25
USES_APP = {"0-9": 0.15, "10-49": 0.38, "50-249": 0.60, "250+": 0.80}
USES_BRANCH = {"0-9": 0.26, "10-49": 0.27, "50-249": 0.36, "250+": 0.50}
# A0-41: outgoing transfers a month = one per active supplier of the month (E_PAY_PURCHASES of them paid
# electronically; invoices of one supplier are often paid together) + one per
# person paid (persons working minus the owner, by size: PAYROLL) + TAXES (social insurance, income tax, VAT);
# a suspended firm pays its social insurance only. banking_frequency (daily / weekly / monthly or less) from the
# transfers with a lognormal spread (FREQ_SIGMA): daily at 20+ a month, weekly at 4+. Checks: Szczepaniec 2004
# (median transfers a month 10 / 30 / 105 for 0-9 / 10-49 / 50-249, historical), ZBP NetB@nk (19 transfers per
# active SME client, period not stated), Dominiak 2015 (Wielkopolska: over 40% of firms daily, about 20%
# several times a week; regional).
E_PAY_PURCHASES = 0.9
PAYROLL = {"1": 0, "2-9": 3, "10-49": 20, "50-249": 100, "250+": 500}
TAXES = 3
FREQ_SIGMA = 0.6
FREQUENCY = ("daily", "weekly", "monthly_or_less")


def bank_operator(authority: dict | None, working: str | None = None) -> str:
    roles = {r for p in (authority or {}).get("persons", []) for r in p.get("company_roles", [])}
    office = "accounting_firm" in roles
    finance = bool(roles & {"accountant", "cfo", "assistant", "dedicated_employee", "administrator"})
    if working == "1":  # 0.6.11: a firm where only the owner works has no staff: a finance person with access is outside
        office, finance = office or finance, False
    return "mixed" if office and finance else "accounting_office" if office else "in_house_finance" if finance else "owner_or_board"


def channels_for(firm: dict, out: dict, status: str, working: str, authority: dict | None, u: np.ndarray, z: float) -> dict:
    size = firm["size_class"]
    p = np.array(PRIMARY_CHANNEL[size], dtype=float)
    if not out.get("has_erp"):
        p[3] = 0.0
    if out.get("has_separate_business_account") is False:  # the business runs through a private (retail) account
        p[3] = 0.0
        p = op._shift(p, 1, APP_SHIFT_PRIVATE)
    primary = op._pick(CHANNELS, p, u[0])
    app = primary == "mobile_app" or bool(u[1] < USES_APP[size])
    branch = primary == "branch" or bool(u[2] < USES_BRANCH[size])
    if status == "suspended":
        transfers = 1
    else:
        transfers = int(round(out["suppliers_active_month"] * E_PAY_PURCHASES)) + PAYROLL.get(working, 0) + TAXES
    t = transfers * math.exp(FREQ_SIGMA * z)
    freq = "daily" if t >= 20 else "weekly" if t >= 4 else "monthly_or_less"
    if primary == "h2h_api":
        freq = "daily"
    if status == "suspended":  # 0.6.11: one transfer a month (social insurance) is monthly, without noise
        freq = "monthly_or_less"
    return {"primary_channel": primary, "uses_mobile_app": bool(app), "uses_branch_12m": bool(branch),
            "bank_operator": bank_operator(authority, working), "transfers_out_month": transfers, "banking_frequency": freq}


# A0-42 (0.6.7-0.6.8, EXPERIMENTAL): bank_segment, an OPERATIONAL segmentation of the synthetic population,
# not an empirically confirmed market standard: "retail_family" (served in the retail model with business
# functions) or "business" (served in a dedicated business-banking model); bank_segment_rule names the rule.
# Banks in Poland use two families of rules (public reports of 10 banks, layer2_anchors.json
# bank_segmentation): turnover only (PLN 5m / 10m / 15m) and bookkeeping with or without turnover (simplified
# books below EUR 2m; full books below PLN 4m). The default "combined" joins them: retail_family with simplified
# books and revenue below EUR 2m (at the rate of the revenue year's limits), or with full books and revenue below PLN 4m;
# business otherwise; "unknown" when bookkeeping or revenue is missing. The number of business firms is a
# model result under the rule, not an estimate of the business-banking market; banks' segment counts serve as
# a plausibility test only. Bank organisation, client classification and platform need not share boundaries:
# bank_platform is a separate feature. A0-43: retail_family -> business_platform with P_PLATFORM_UP (1 of 10
# banks serves sole proprietors in its business system), business -> retail_app_business_profile with
# P_PLATFORM_DOWN when revenue is below PLN 10m (3 of 10 banks keep firms up to PLN 5-15m in retail); both
# assumptions from the counts of banks.
SEGMENT_RULES = ("combined", "bookkeeping", "turnover_5m", "turnover_10m", "legal_form_s14")
SEGMENT_SIMPLIFIED_PLN = 2_000_000 * EUR_PLN_LIMITS  # banks' EUR 2m for simplified books (operational, not a tax limit)
P_PLATFORM_UP, P_PLATFORM_DOWN = 0.10, 0.30
PLATFORMS = ("retail_app_business_profile", "business_platform")


def bank_segment(firm: dict, p: dict, rule: str = "combined") -> str:
    books, revenue = p.get("bookkeeping_form"), p.get("revenue_annual_pln", p.get("revenue_year_pln"))
    if rule == "legal_form_s14":  # NBP / ESA 2010: sole proprietors with up to 9 persons are households
        return "retail_family" if firm["legal_form"] == "099" and firm["size_class"] == "0-9" else "business"
    if rule not in SEGMENT_RULES:
        raise ValueError(f"rule {rule!r}, expected one of {SEGMENT_RULES}")
    if (rule in ("combined", "bookkeeping") and books is None) or (rule != "bookkeeping" and revenue is None):
        return "unknown"
    full = books == "full_books"
    if rule == "combined":
        retail = (not full and revenue < SEGMENT_SIMPLIFIED_PLN) or (full and revenue < 4e6)
    elif rule == "bookkeeping":
        retail = not full
    elif rule == "turnover_5m":
        retail = revenue < 5e6
    else:
        retail = revenue < 10e6
    return "retail_family" if retail else "business"


def bank_platform(segment: str, revenue: int, u: float, separate_account: bool = True) -> str | None:
    if segment in ("retail_family", "business") and not separate_account:  # 0.6.11: a private account is retail
        return "retail_app_business_profile"
    if segment == "retail_family":
        return "business_platform" if u < P_PLATFORM_UP else "retail_app_business_profile"
    if segment == "business":
        return "retail_app_business_profile" if revenue < 10e6 and u < P_PLATFORM_DOWN else "business_platform"
    return None


@functools.lru_cache(maxsize=1)
def vat_ref() -> dict:
    return json.loads(VAT.read_text(encoding="utf-8"))


def vat_status_for(firm: dict, u: float) -> str:
    form, size = firm["legal_form"], firm["size_class"]
    if form in ("099", "019"):
        return "not_measured"
    ref = vat_ref()
    cell = (ref["llc_by_size_proxy"].get(size) if form == "117" and size != "0-9" else None) \
        or (ref["by_form_division"].get(f"{form}|{str(firm['pkd_division']).zfill(2)}") if size == "0-9" else None) \
        or ref["by_form"].get(form) or ref["by_form"]["117"]
    keys = [k for k in VAT_STATUSES if k in cell["p"]]
    return op._pick(tuple(keys), np.array([cell["p"][k] for k in keys]), u)


@functools.lru_cache(maxsize=1)
def ownership_ref() -> dict:
    return json.loads(OWNERSHIP.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def foreign_gus() -> dict:
    return json.loads(FOREIGN_GUS.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def generated_solo() -> dict:
    return json.loads(GENERATED_SOLO.read_text(encoding="utf-8"))["share"] if GENERATED_SOLO.exists() else {}


@functools.lru_cache(maxsize=1)
def sbs() -> dict:
    return json.loads(SBS.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def ceidg() -> dict:
    return json.loads(REF.read_text(encoding="utf-8"))


def rng_for(firm_seed, stream: str) -> np.random.Generator:
    digest = hashlib.sha256(f"layer2/{STREAM}/{firm_seed}/{stream}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _cell(table: dict, division: str) -> dict:
    return (table.get(str(division).zfill(2)) or table["all"])["p"]


def age_and_status(firm: dict, u: float) -> tuple[str | None, str]:
    """A0-23: JDG by PKD division from CEIDG (suspended only in the 0-9 class: 0.5.1); civil partnerships: age from CEIDG, active (CEIDG does
    not show their suspension); companies: age not drawn in 0.5 (KRS and BIR later), active."""
    form, div = firm["legal_form"], firm["pkd_division"]
    if form == "099":
        p = _cell(ceidg()["jdg_age_status"], div)
        keys = list(p)
        if firm["size_class"] != "0-9":  # only a JDG without employees may suspend (Prawo przedsiębiorców, art. 22)
            keys = [k for k in keys if k.endswith("|active")]
        age, status = op._pick(tuple(keys), np.array([p[k] for k in keys]), u).split("|")
        return age, status
    if form == "019":
        p = _cell(ceidg()["civil_partnership_age"], div)
        keys = list(p)
        return op._pick(tuple(keys), np.array([p[k] for k in keys]), u), "active"
    return None, "active"


def tax_card_open(firm: dict, age: str | None, working: str | None = None) -> bool:
    """Form, age (continuation since 2021, approximated by 6+ years) and activity of art. 23; the catch-all divisions
    only where the owner works alone (working '1'); not for a civil partnership in health, veterinary, home care,
    lessons. Headcount limits of the annex are not checked beyond the micro class."""
    div = str(firm.get("pkd_division", "")).zfill(2)
    if firm["legal_form"] not in TAX_CARD_OK or age not in TAX_CARD_AGES:
        return False
    if firm["legal_form"] == "019" and div in TAX_CARD_NOT_CIVIL:
        return False
    return div in TAX_CARD_DIVISIONS or (div in TAX_CARD_SOLO_DIVISIONS and working == "1")


def bookkeeping_for(firm: dict, age: str | None, u: float, ownership: str | None = None, working: str | None = None) -> str:
    """0.6.15: a general partnership (118) whose partners are not all natural persons (ownership other than domestic
    persons or foreign, assumed foreign natural persons) keeps full books (act on accounting art. 2 ust. 1 pkt 2:
    the simplified forms are for general partnerships of natural persons); a professional partnership (115) has
    natural persons only by law (KSH art. 87)."""
    form, size = firm["legal_form"], firm["size_class"]
    if form not in SIMPLIFIED_OK:
        return "full_books"
    if form == "118" and not run_by_natural_persons(form, ownership):  # 0.6.16: the one rule
        return "full_books"
    p = np.array((BOOKS_NATURAL if form in ("099", "019") else BOOKS_PARTNERSHIP)[size], dtype=float)
    if form not in FLAT_RATE_OK:
        p[2] = 0.0
    p[3] = p[3] * TAX_CARD_FACTOR if tax_card_open(firm, age, working) else 0.0
    return op._pick(BOOKKEEPING, p, u)


def draw(firm: dict, authority: dict | None = None, variant: str = "base", dependence: str = "independent") -> dict:
    """Layer-2 profile 0.5 of one firm (firm dict of the population file: firm_seed, legal_form,
    size_class, pkd_section, pkd_division). With dependence other than 'independent' the per-firm copula keeps only
    the theoretical distribution of each field; for comparing agents use draw_population (0.6.15)."""
    u = rng_for(firm["firm_seed"], "anchored").random(2)  # shared by all variants
    age, status = age_and_status(firm, u[0])
    ownership = ownership_for(firm, *rng_for(firm["firm_seed"], "ownership").random(2))  # its own stream: drawn first (0.6.15)
    working = persons_working(firm, authority, status, rng_for(firm["firm_seed"], "persons").random())  # own stream
    books = bookkeeping_for(firm, age, u[1], ownership, working)
    p_ext = P_EXTERNAL_MICRO[books] if firm["size_class"] == "0-9" else None
    b2c_only = books == "tax_card" and str(firm["pkd_division"]).zfill(2) in TAX_CARD_B2C_ONLY
    months = months_active_2024(age, rng_for(firm["firm_seed"], "founding").random())
    revenue = revenue_for(firm, working, status, float(rng_for(firm["firm_seed"], "revenue").standard_normal()), books, months)
    zi = rng_for(firm["firm_seed"], "invoices")
    z = np.append(zi.standard_normal(3), zi.random(3))
    parts: dict = {}
    out = op.draw_layer2(firm, authority, variant, suspended=status == "suspended", p_external=p_ext,
                         invoices=invoices_for(firm, revenue, working, z, parts), tool_adjust=tool_adjust_for(firm["size_class"]),
                         solo=working == "1", b2c_only=b2c_only,
                         b2c_weights=B2C_BY_DIVISION.get(str(firm["pkd_division"]).zfill(2)))
    has_erp = out["invoicing_tool"] == "erp" or rng_for(firm["firm_seed"], "erp").random() < P_ERP_EXTRA[firm["size_class"]]
    out.update({"layer2_version": VERSION, "activity_status": status, "firm_age": age, "bookkeeping_form": books,
                "persons_working": working, "revenue_annual_pln": revenue, "months_active_2024": months,
                # 0.6.14: a firm suspended now has an unknown revenue of 2024 (its history is not modelled), not 0
                "revenue_year_pln": None if status == "suspended" else revenue_2024(revenue, books, months), "has_erp": bool(has_erp),
                "invoices_b2c_month": parts.get("b2c", 0), "invoices_ksef_month": parts.get("ksef", 0),
                "b2c_invoices_in_ksef": parts.get("b2c_in_ksef", False),
                "ownership": ownership,
                "vat_status": vat_status_for(firm, rng_for(firm["firm_seed"], "vat").random()),
                **products_for(firm, working, status, rng_for(firm["firm_seed"], "products").random(10)),
                **payments_for(firm, out, status, rng_for(firm["firm_seed"], "payments").random(2)),
                "dependence": dependence, **counterparties_for(firm, out, status, parts, dependence)})
    out.update(banking_for(firm, out, age, working, revenue, out["ownership"], rng_for(firm["firm_seed"], "banking").random(5)))
    zc = rng_for(firm["firm_seed"], "channels")
    out.update(channels_for(firm, out, status, working, authority, zc.random(3), float(zc.standard_normal())))
    out["bank_segment"] = bank_segment(firm, out)
    out["bank_segment_rule"] = "combined"
    out["bank_platform"] = bank_platform(out["bank_segment"], revenue, rng_for(firm["firm_seed"], "platform").random(),
                                         out["has_separate_business_account"])
    out["null_reasons"] = null_reasons(out, status, firm)
    problems = consistency(out, firm)
    assert not problems, problems
    return out


# 0.6.9 (0.6.11: from the firm's facts, not from the field name alone): why a field is null. "not_applicable": the
# field does not apply to this firm; "unknown": the value exists but the population does not know it.
def null_reasons(p: dict, status: str, firm: dict | None = None) -> dict:
    out = {}
    suspended = status == "suspended"
    for k, v in p.items():
        if v is not None or k == "null_reasons":
            continue
        if k == "firm_age":
            r = ("unknown", "not_drawn_for_companies")
        elif k == "liquidity_declared":
            r = ("not_applicable", "suspended") if suspended else ("unknown", "not_given")
        elif k in ("b2b_clients_year", "payment_terms_purchases_days", "overdue_payables_share"):
            r = ("unknown", "suspended_history") if suspended else ("unknown", "not_given")
        elif k in ("largest_b2b_client_share", "payment_terms_sales_days", "overdue_receivables_share"):
            r = ("unknown", "suspended_history") if suspended else (
                ("not_applicable", "no_b2b_clients") if p.get("b2b_clients_year") == "0" else ("unknown", "not_given"))
        elif k == "owner_private_account_same_bank":
            by_persons = firm is not None and run_by_natural_persons(firm["legal_form"], p.get("ownership"))
            r = ("unknown", "not_measured_50plus") if by_persons else ("not_applicable", "not_run_by_natural_persons")
        elif k == "bank_platform":
            r = ("unknown", "segment_unknown")
        elif k == "revenue_year_pln":
            r = ("not_applicable", "founded_after_2024") if p.get("months_active_2024") == 0 else ("unknown", "suspended_history")
        else:
            r = ("unknown", "not_given")
        out[k] = {"kind": r[0], "why": r[1]}
    return out


def consistency(p: dict, firm: dict | None = None) -> list[str]:
    bad = op.consistency(p)
    if "invoices_ksef_month" in p and not 0 <= p["invoices_ksef_month"] <= p["invoices_issued_month"]:
        bad.append("more invoices in KSeF than issued")
    if "invoices_b2c_month" in p and p["invoices_b2c_month"] > p["invoices_issued_month"]:
        bad.append("more B2C invoices than issued")
    if p.get("invoicing_tool") == "erp" and p.get("has_erp") is False:
        bad.append("invoicing in an ERP without an ERP")
    if p.get("activity_status") == "suspended" and p["invoices_issued_month"] > 0:
        bad.append("a suspended firm issues invoices")
    if p.get("activity_status") == "suspended" and p.get("persons_working", "1") != "1":
        bad.append("a suspended firm with persons working besides the owner")
    if firm is not None and p.get("activity_status") == "suspended" and firm.get("size_class") != "0-9":
        bad.append("a suspended firm with employees")
    books, rev = p.get("bookkeeping_form"), p.get("revenue_year_pln")
    if books in SIMPLIFIED_LIMIT_PLN and rev is not None and (
            rev > SIMPLIFIED_LIMIT_PLN[books] if books == "flat_rate" else rev >= SIMPLIFIED_LIMIT_PLN[books]):
        bad.append("simplified books above the legal revenue limit")  # KPiR / tax card below EUR 2.5m, flat rate at most EUR 2m
    if p.get("persons_working") == "1" and (p.get("accounting") == "mixed" or (
            p.get("bank_operator") == "accounting_office" and p.get("accounting") == "own")):
        bad.append("a firm where only the owner works keeps its books in-house with a finance person")  # 0.6.13
    if "months_active_2024" in p:  # 0.6.13: chronology of age and revenue
        m = p["months_active_2024"]
        if p.get("firm_age") == "<1" and m != 0:
            bad.append("a firm under a year old (as of 11.11.2026) active in 2024")
        if (m == 0 and rev is not None) or (m > 0 and rev is None and p.get("activity_status") != "suspended"):
            bad.append("revenue of 2024 known for a firm founded after 2024, or missing for one active in 2024")
        if rev is not None and p.get("revenue_annual_pln") is not None and rev > p["revenue_annual_pln"] * m / 12 + 100:
            bad.append("revenue of 2024 above the annual revenue for the months active")
    if p.get("has_separate_business_account") is False and (p.get("bank_platform") == "business_platform" or p.get("primary_channel") == "h2h_api"):
        bad.append("business platform or host-to-host without a business account")
    if (p.get("bank_segment") in ("retail_family", "business")) != (p.get("bank_platform") is not None) and "bank_platform" in p:
        bad.append("bank platform and bank segment disagree on being known")
    if p.get("has_separate_business_account") is False and not p.get("mixes_private_and_business"):
        bad.append("no separate business account, yet private and business money kept apart")
    if p.get("has_separate_business_account") is False and p.get("owner_private_account_same_bank") is False:
        bad.append("business run through the private account, yet the private account at another bank")
    if firm is not None and "has_separate_business_account" in p and firm["legal_form"] != "099" and not p["has_separate_business_account"]:
        bad.append("a company or partnership without a business account")
    if p.get("primary_channel") == "mobile_app" and p.get("uses_mobile_app") is False:
        bad.append("the app as primary channel without using the app")
    if p.get("primary_channel") == "branch" and p.get("uses_branch_12m") is False:
        bad.append("the branch as primary channel without a branch visit")
    if p.get("primary_channel") == "h2h_api" and p.get("has_erp") is False:
        bad.append("host-to-host or API banking without an ERP")
    n = p.get("b2b_clients_year")
    if n is not None:
        if n == "0" and (p.get("largest_b2b_client_share") or p.get("payment_terms_sales_days") or p.get("overdue_receivables_share")):
            bad.append("concentration, sales terms or overdue receivables without B2B clients")
        if n == "1" and p.get("largest_b2b_client_share") != ">=75%":
            bad.append("one B2B client below 100% of B2B sales")
        if n == "2" and p.get("largest_b2b_client_share") in ("<25%", "25-<50%"):
            bad.append("two B2B clients, the largest below 50%")
        if n != "0" and p.get("b2c_share_of_sales", 0) >= 1:
            bad.append("B2B clients without B2B sales")
    if firm is not None and "bookkeeping_form" in p:
        form = firm["legal_form"]
        if p["bookkeeping_form"] != "full_books" and form not in SIMPLIFIED_OK:
            bad.append("simplified books for a form that must keep full books")
        if p["bookkeeping_form"] == "flat_rate" and form not in FLAT_RATE_OK:
            bad.append("flat rate for a form that may not use it")
        if p["bookkeeping_form"] == "tax_card" and not tax_card_open(firm, p.get("firm_age"), p.get("persons_working")):
            bad.append("tax card for a form, an age or an activity that may not have it")  # 0.6.15: activity (art. 23)
        if p["bookkeeping_form"] == "tax_card" and str(firm.get("pkd_division", "")).zfill(2) in TAX_CARD_B2C_ONLY \
                and p.get("b2c_share_of_sales", 1.0) < 1.0:
            bad.append("a tax card in health or veterinary services with sales to firms")  # 0.6.15
        if p.get("mixes_private_and_business") and not run_by_natural_persons(form, p.get("ownership")):
            bad.append("private and business money mixed in a firm not run by natural persons")  # 0.6.16
        if p.get("owner_private_account_same_bank") is not None and not run_by_natural_persons(form, p.get("ownership")):
            bad.append("an owner's private account in a firm not run by natural persons")  # 0.6.16
        if form == "118" and p["bookkeeping_form"] != "full_books" and not run_by_natural_persons(form, p.get("ownership")):
            bad.append("simplified books in a general partnership with a partner that is not a natural person")  # 0.6.15
    return bad
