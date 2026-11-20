"""Firm card for a track-B agent (docs/tor-b-etap1.md): a Polish text rendered from the record, never
by an LLM. Every fact carries its field id in brackets, so the premises an agent cites can be checked
against the card. The order of the facts can be permuted (a placebo intervention).
"""
from __future__ import annotations

import hashlib

import numpy as np

from .dictionaries import BUSINESS_FORMS

VERSION = "0.4.0"

PKD_SECTIONS = {
    "A": "rolnictwo, leśnictwo, łowiectwo i rybactwo", "B": "górnictwo i wydobywanie", "C": "przetwórstwo przemysłowe",
    "D": "wytwarzanie i zaopatrywanie w energię elektryczną, gaz, parę wodną i gorącą wodę",
    "E": "dostawa wody; gospodarowanie ściekami i odpadami; rekultywacja", "F": "budownictwo",
    "G": "handel hurtowy i detaliczny; naprawa pojazdów samochodowych", "H": "transport i gospodarka magazynowa",
    "I": "działalność związana z zakwaterowaniem i usługami gastronomicznymi", "J": "informacja i komunikacja",
    "L": "działalność związana z obsługą rynku nieruchomości", "M": "działalność profesjonalna, naukowa i techniczna",
    "N": "działalność w zakresie usług administrowania i działalność wspierająca", "P": "edukacja",
    "Q": "opieka zdrowotna i pomoc społeczna", "R": "działalność związana z kulturą, rozrywką i rekreacją",
    "S": "pozostała działalność usługowa",
}
SIZE_TEXT = {"0-9": "do 9 osób", "10-49": "10–49 osób", "50-249": "50–249 osób", "250+": "250 osób i więcej"}
TOOL_TEXT = {"none": "brak dedykowanego narzędzia (edytor tekstu, arkusz albo ręcznie)", "invoicing_program": "program fakturowy",
             "erp": "system ERP"}
LINK_TEXT = {"manual": "ręcznie", "files": "przez import i eksport plików", "automatic": "automatycznie z narzędzia do faktur",
             "office_from_ksef": "biuro rachunkowe samo pobiera faktury z KSeF"}
KSEF_MANUAL = "ręcznie: wpisanie albo przepisanie faktury w bezpłatnej aplikacji KSeF Ministerstwa Finansów"
ACCOUNTING_TEXT = {"own": "własna (w firmie)", "external": "zewnętrzne biuro rachunkowe", "mixed": "model mieszany: w firmie i w biurze rachunkowym"}
FUNCTION_TEXT = {"owner": "właściciel", "president": "prezes zarządu", "vice_president": "wiceprezes zarządu",
                 "member": "członek zarządu", "partner": "wspólnik", "general_partner": "komplementariusz"}
ROLE_TEXT = {"accountant": "księgowa", "cfo": "dyrektor finansowy", "assistant": "asystentka",
             "dedicated_employee": "pracownik do spraw płatności", "accounting_firm": "biuro rachunkowe (z dostępem do bankowości)",
             "accounting_office": "biuro rachunkowe (bez dostępu do bankowości)"}
SYLLABLES = ["ka", "ro", "mi", "len", "tor", "vi", "sa", "dom", "bel", "ter", "no", "pol", "gra", "max", "ri",
             "zen", "ko", "ma", "lu", "fin", "tek", "zo", "bra", "nor", "vel", "dar", "pro", "sil", "ver", "mar"]

CLIENTS_TEXT = {"0": "brak", "1": "jeden", "2": "dwóch", "3-5": "3–5", "6-20": "6–20", "21-100": "21–100", "101+": "ponad 100"}
CONCENTRATION_TEXT = {"<25%": "poniżej 25%", "25-<50%": "od 25% do poniżej 50%", "50-<75%": "od 50% do poniżej 75%",
                      ">=75%": "75% lub więcej"}  # 0.6.16 (blind check of B-K2): the bounds stated
TERMS_TEXT = {"0": "w dniu wystawienia faktury (termin 0 dni)", "1-7": "do 7 dni", "8-14": "8–14 dni", "15-30": "15–30 dni", "31-60": "31–60 dni", "61+": "ponad 60 dni"}
OVERDUE_TEXT = {"0": "0% (wszystko w terminie)", "(0,5%]": "do 5%", "(5,15%]": "5–15%", "(15,30%]": "15–30%", ">30%": "ponad 30%"}
YES_NO = lambda v, p: "tak" if v else "nie"  # noqa: E731
def _pln(r: int) -> str:
    return f"ok. {r / 1e6:.1f} mln zł" if r >= 1e6 else f"ok. {round(r / 1e3)} tys. zł"


AS_OF = "11.11.2026"  # the date of the age bands and of "the last 12 months" (layer 2 chronology, 0.6.13)
NA_TEXT = {"no_b2b_clients": "firma nie miała klientów firmowych w ostatnich 12 miesiącach", "not_run_by_natural_persons": "właścicielami nie są osoby fizyczne z Polski",
           "suspended": "działalność zawieszona"}
EXPERIMENTAL_FACTS = (
    ("b2b_clients_year", "Klienci firmowi w ostatnich 12 miesiącach", lambda v, p: CLIENTS_TEXT[v]),
    ("largest_b2b_client_share", "Największy klient firmowy: udział w sprzedaży do firm",
     lambda v, p: "jedyny klient firmowy (100%)" if p.get("b2b_clients_year") == "1" else CONCENTRATION_TEXT[v]),
    ("payment_terms_sales_days", "Typowy termin płatności dla klientów firmowych", lambda v, p: TERMS_TEXT[v]),
    ("overdue_receivables_share", "Faktury dla firm z ostatnich 12 miesięcy, których termin minął: udział wartości zapłaconej po terminie albo niezapłaconej", lambda v, p: OVERDUE_TEXT[v]),
    ("payment_terms_purchases_days", "Typowy termin płatności dostawcom", lambda v, p: TERMS_TEXT[v]),
    ("overdue_payables_share", "Faktury zakupowe z ostatnich 12 miesięcy, których termin minął: udział wartości zapłaconej przez firmę po terminie albo niezapłaconej", lambda v, p: OVERDUE_TEXT[v]),
    ("has_separate_business_account", "Osobny rachunek firmowy",
     lambda v, p: "tak" if v else "nie; firma korzysta z rachunku prywatnego właściciela"),
    ("mixes_private_and_business", "Finanse prywatne i firmowe",
     lambda v, p: ("wspólne (brak osobnego rachunku firmowego)" if p.get("has_separate_business_account") is False else "częściowo mieszane")
     if v else "rozdzielone"),
    ("banks_count", "Liczba banków z rachunkami używanymi przez firmę", lambda v, p: {"1": "jeden", "2": "dwa", "3": "trzy", "4+": "cztery lub więcej"}[v]),
    ("owner_private_account_same_bank", "Prywatne konto właściciela w banku głównym firmy", YES_NO),
    ("changed_main_bank_since_start", "Zmiana banku głównego od założenia firmy", YES_NO),
    ("bank_segment", "Model obsługi w banku", lambda v, p: {"retail_family": "detaliczny z funkcjami firmowymi",
                                                             "business": "wyspecjalizowana bankowość firmowa"}.get(v)),
    ("bank_platform", "System bankowości", lambda v, p: {"retail_app_business_profile": "aplikacja bankowości detalicznej z profilem firmowym",
                                                         "business_platform": "osobna platforma bankowości firmowej"}[v]),
    ("primary_channel", "Główny kanał bankowości firmy", lambda v, p: {"web": "bankowość internetowa na komputerze", "mobile_app": "aplikacja mobilna",
                                                                       "branch": "oddział", "h2h_api": "połączenie systemu ERP z bankiem (host-to-host lub API)"}[v]),
    ("uses_mobile_app", "Korzystanie z aplikacji mobilnej banku", YES_NO),
    ("uses_branch_12m", "Wizyta w oddziale banku w ostatnich 12 miesiącach", YES_NO),
    ("bank_operator", "Kto obsługuje bankowość firmy", lambda v, p: {"owner_or_board": "właściciel lub zarząd", "in_house_finance": "pracownicy finansów w firmie",
                                                                     "accounting_office": "biuro rachunkowe lub księgowa spoza firmy", "mixed": "biuro rachunkowe i pracownicy firmy"}[v]),
    ("transfers_out_month", "Przelewy wychodzące w typowym miesiącu", lambda v, p: f"około {v}"),
    ("banking_frequency", "Jak często firma korzysta z bankowości", lambda v, p: {"daily": "codziennie", "weekly": "co tydzień",
                                                                                 "monthly_or_less": "raz w miesiącu lub rzadziej"}[v]),
)
LIQUIDITY_TEXT = {"over_3_months": "na ponad 3 miesiące działalności", "2_3_months": "na 2–3 miesiące działalności",
                  "1_month": "na około miesiąc działalności", "insufficient_1_month": "nawet nie na miesiąc działalności"}


def firm_name(firm: dict) -> str:
    """A synthetic name made of syllables, deterministic from the firm seed (no real firm)."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"name/{firm['firm_seed']}".encode()).digest()[:8], "little"))
    word = "".join(rng.choice(SYLLABLES) for _ in range(int(rng.integers(2, 4))))
    return word.capitalize()


def person_text(pid: str, authority: dict | None) -> str:
    if pid in ROLE_TEXT:
        return ROLE_TEXT[pid]
    for p in (authority or {}).get("persons", []):
        if p["id"] == pid:
            if p.get("function"):
                return FUNCTION_TEXT.get(p["function"], p["function"])
            roles = [ROLE_TEXT.get(r, r) for r in p.get("company_roles", [])]
            return ", ".join(roles) or pid
    return "właściciel" if pid == "owner" else pid


ACTIVITY_TEXT = {"active": "prowadzona", "suspended": "zawieszona (firma nie sprzedaje; rachunek pozostaje otwarty)"}
AGE_TEXT = {"<1": "poniżej roku", "1-2": "1–2 lata", "3-5": "3–5 lat", "6-10": "6–10 lat", "11-20": "11–20 lat",
            "21+": "ponad 20 lat"}
OWNERSHIP_TEXT = {"domestic_persons": "osoby fizyczne z Polski", "domestic_entities": "inne firmy lub podmioty z Polski (zwykle grupa)",
                  "foreign": "właściciel zagraniczny", "public": "sektor publiczny", "mixed_no_majority": "własność mieszana, bez przewagi"}
PRODUCT_TEXT = {"credit_line_overdraft": "limit w rachunku lub linia kredytowa", "bank_loan": "kredyt bankowy",
                "trade_credit": "kredyt kupiecki (odroczona płatność u dostawców)", "leasing": "leasing", "factoring": "faktoring"}
BOOKKEEPING_TEXT = {"full_books": "pełne księgi rachunkowe", "kpir": "książka przychodów i rozchodów",
                    "flat_rate": "ryczałt od przychodów ewidencjonowanych", "tax_card": "karta podatkowa"}


def facts(firm: dict, layer2: dict, authority: dict | None, name: str | None = None,
          experimental: bool = False) -> list[tuple[str, str, str]]:
    """(field id, label, value) of every fact on the card, in the default order. The experimental fields of
    0.6.3-0.6.8 appear only with experimental=True: then every known value is shown (also "no" and "kept
    apart"), a field that does not apply is shown as "nie dotyczy" with its reason, and an unknown value is left
    out (null_reasons of the profile tells the two apart)."""
    form = BUSINESS_FORMS.get(firm["legal_form"])
    form_text = form.name if form else firm.get("legal_form_name", firm["legal_form"])
    abbr = form.short if form else firm.get("legal_form_abbr", "")
    nm = name or firm_name(firm)
    d0 = layer2["division_of_work"]
    ids = sorted({i for k in ("issues", "passes_on", "books") for i in d0[k]})
    label = {i: person_text(i, authority) for i in ids}
    same = {t for t in label.values() if list(label.values()).count(t) > 1}
    for t in same:  # two different persons with the same role are told apart on the card
        for n, i in enumerate(sorted(i for i in ids if label[i] == t), 1):
            label[i] = f"{t} {n}"
    who = lambda xs: ", ".join(label[i] for i in xs)  # noqa: E731
    d = layer2["division_of_work"]
    if d["passes_on"]:
        passes = who(d["passes_on"])
    elif layer2["to_accounting"] == "automatic":
        passes = "nikt: dane przechodzą automatycznie z narzędzia do faktur"
    elif layer2["to_accounting"] == "office_from_ksef":
        passes = "nikt: biuro rachunkowe samo pobiera faktury z KSeF"
    else:  # the issuer keeps the books
        passes = "nie ma osobnego przekazania: ta sama osoba wystawia faktury i księguje"
    extra = []  # layer 2 from 0.5 on; absent from 0.4 profiles, so their cards do not change
    if "activity_status" in layer2:
        # 0.6.16 (blind check): the date the card describes, and what a field left out means
        extra.append(("as_of", "Stan na dzień", AS_OF + "; nie zakładaj wartości pól, których nie podano na karcie"))
        extra.append(("activity_status", "Stan działalności", ACTIVITY_TEXT[layer2["activity_status"]]))
    if layer2.get("firm_age"):
        extra.append(("firm_age", "Wiek firmy (pełne lata)", AGE_TEXT[layer2["firm_age"]]))
    if layer2.get("persons_working") in ("1", "2-9"):
        one = "tylko właściciel" if firm.get("legal_form") == "099" else "jedna osoba"  # 0.6.16: a company has no single owner
        extra.append(("persons_working", "Kto pracuje w firmie", one if layer2["persons_working"] == "1" else "2–9 osób"))
    if "revenue_year_pln" in layer2:
        r, months = layer2["revenue_year_pln"], layer2.get("months_active_2024", 12)
        if r is None and months == 0:  # 0.6.13: founded after 2024
            txt = "nie dotyczy (firma założona po 2024 r.)"
        elif r is None:  # 0.6.14: suspended now, its revenue of 2024 is not known
            txt = "nieznany (firma obecnie zawieszona: brak danych o przychodzie za 2024 r.)"
        else:
            txt = _pln(r) + ("" if months == 12 else f" (za {months} mies. działalności w 2024 r.)")
        extra.append(("revenue_year_pln", "Przychód w 2024 r.", txt))
        annual = layer2.get("revenue_annual_pln")
        if annual and (months < 12):  # a young firm: its current yearly scale, a model estimate
            extra.append(("revenue_annual_pln", "Obecna skala przychodu (szacunek roczny)", _pln(annual)))
    if layer2.get("ownership") and layer2["ownership"] != "natural_person_owner":
        own = OWNERSHIP_TEXT[layer2["ownership"]]
        if layer2["ownership"] == "foreign" and firm.get("legal_form") == "118":
            own = "osoby fizyczne z zagranicy"  # 0.6.16: as the generator assumes for general partnerships
        extra.append(("ownership", "Właściciele", own))
    elif "ownership" in layer2 and firm.get("legal_form") == "019":  # 0.6.16: a civil partnership is natural persons here
        extra.append(("ownership", "Właściciele", "wspólnicy, osoby fizyczne"))
    elif "ownership" in layer2 and firm.get("legal_form") == "099":  # 0.6.16 (blind check of B-K2): stated, not left to infer
        extra.append(("ownership", "Właściciele", "osoba fizyczna prowadząca działalność"))
    if "accepts_cards" in layer2:
        extra.append(("accepts_cards", "Przyjmowanie płatności kartą", "tak" if layer2["accepts_cards"] else "nie"))
        extra.append(("e_payment_share_of_sales", "Część sprzedaży opłacana elektronicznie (wartość)", f"{round(100 * layer2['e_payment_share_of_sales'])}%"))
    if layer2.get("liquidity_declared"):
        extra.append(("liquidity_declared", "Na ile wystarczy obecna płynność (ocena firmy)", LIQUIDITY_TEXT[layer2["liquidity_declared"]]))
    if experimental and layer2.get("liquidity_declared") is None and \
            (layer2.get("null_reasons") or {}).get("liquidity_declared", {}).get("kind") == "not_applicable":
        extra.append(("liquidity_declared", "Na ile wystarczy obecna płynność (ocena firmy)",
                      "nie dotyczy (" + NA_TEXT[layer2["null_reasons"]["liquidity_declared"]["why"]] + ")"))
    if experimental:  # 0.6.3-0.6.11 fields: every known value shown (False too); "not applicable" said; unknown left out
        for fid, lab, fmt in EXPERIMENTAL_FACTS:
            if fid not in layer2:
                continue
            v = layer2[fid]
            if v is None:
                why = (layer2.get("null_reasons") or {}).get(fid, {})
                if why.get("kind") == "not_applicable":
                    txt = NA_TEXT.get(why.get("why"), "brak podstawy")
                    if why.get("why") == "not_run_by_natural_persons" and firm.get("legal_form") == "118":
                        txt = "wspólnikami nie są osoby fizyczne"  # a general partnership: foreign persons count (0.6.16)
                    extra.append((fid, lab, "nie dotyczy (" + txt + ")"))
                continue
            text = fmt(v, layer2)
            if text is not None:
                extra.append((fid, lab, text))
    if "products_used_6m" in layer2:
        used = [PRODUCT_TEXT[k] for k, v in layer2["products_used_6m"].items() if v]
        extra.append(("products_used_6m", "Finansowanie wykorzystane w ostatnich 6 miesiącach", ", ".join(used) if used else "żadne z wymienionych"))
        # 0.6.16 (blind check of B-K2): always shown, so that its absence never has to be read as 'no need'
        unmet = [t for k, t in (("unmet_need_bank_loan", "kredyt bankowy"), ("unmet_need_credit_line", "limit w rachunku lub linia kredytowa"))
                 if layer2.get(k)]
        extra.append(("unmet_need", "Niezaspokojona potrzeba kredytu bankowego lub limitu w rachunku w ostatnich 6 miesiącach",
                      ", ".join(unmet) if unmet else "nie było"))  # only these two are drawn (SAFE); others unknown
    if "bookkeeping_form" in layer2:  # 0.6.16: the period of the form (2025, or the first year of a firm founded later)
        later = layer2.get("months_active_2024") == 0
        lab = ("Forma ewidencji w pierwszym roku działalności, czyli w roku założenia (firma założona po 2024 r.)" if later
               else "Forma ewidencji w 2025 r.")
        extra.append(("bookkeeping_form", lab, BOOKKEEPING_TEXT[layer2["bookkeeping_form"]]))
    return [
        ("name", "Nazwa firmy", f"{nm} {abbr}".strip() if abbr != "JDG" else f"{nm} (działalność gospodarcza)"),
        ("legal_form", "Forma prawna", form_text),
        ("pkd", "Branża", f"sekcja {firm['pkd_section']} ({PKD_SECTIONS.get(firm['pkd_section'], '')}), dział {firm['pkd_division']}"),
        ("size_class", "Liczba pracujących", SIZE_TEXT[firm["size_class"]]),
        ("voivodeship", "Województwo", firm["voivodeship"]),
        *extra,
        ("invoices_issued_month", "Faktury wystawiane w typowym miesiącu", str(layer2["invoices_issued_month"])),
        ("invoices_received_month", "Faktury otrzymywane w typowym miesiącu", str(layer2["invoices_received_month"])),
        ("customers_active_month", "Odbiorcy z co najmniej jedną fakturą w typowym miesiącu", str(layer2["customers_active_month"])),
        ("suppliers_active_month", "Dostawcy z co najmniej jedną fakturą w typowym miesiącu", str(layer2["suppliers_active_month"])),
        ("b2c_share_of_sales", "Udział sprzedaży do konsumentów w wartości sprzedaży netto", f"{round(100 * layer2['b2c_share_of_sales'])}%"),
        ("invoicing_tool", "Narzędzie do wystawiania faktur", TOOL_TEXT[layer2["invoicing_tool"]]),
        ("ksef_submission", "Jak faktury sprzedaży trafiają do KSeF",
         KSEF_MANUAL if layer2["ksef_submission"] == "manual" else LINK_TEXT[layer2["ksef_submission"]]),
        ("to_accounting", "Jak dane faktur trafiają do księgowości", LINK_TEXT[layer2["to_accounting"]]),
        ("payment_reconciliation", "Jak wpłaty są łączone z fakturami", LINK_TEXT[layer2["payment_reconciliation"]]),
        ("accounting", "Księgowość", ACCOUNTING_TEXT[layer2["accounting"]]),
        ("issues", "Kto wystawia faktury", who(d["issues"])),
        ("passes_on", "Kto przekazuje dokumenty do księgowości", passes),
        ("books", "Kto księguje", who(d["books"])),
    ]


def render(firm: dict, layer2: dict, authority: dict | None = None, order_seed: int | None = None,
           name: str | None = None, experimental: bool = False) -> str:
    """The card text. order_seed permutes the facts after the name (placebo: the order of the facts)."""
    fs = facts(firm, layer2, authority, name, experimental)
    head, rest = fs[:1], fs[1:]
    if order_seed is not None:
        rest = [rest[i] for i in np.random.default_rng(order_seed).permutation(len(rest))]
    return "\n".join(f"- {label}: {value} [{fid}]" for fid, label, value in head + rest)
