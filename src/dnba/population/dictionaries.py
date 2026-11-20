"""Layer 1 dictionaries: legal forms, voivodeships, employment size classes."""

from __future__ import annotations

from dataclasses import dataclass

SIZE_CLASSES = ("0-9", "10-49", "50-249", "250+")
"""Classes of the expected number of employees in the REGON tables (in GUS order)."""


@dataclass(frozen=True)
class LegalForm:
    code: str  # specific legal form code (sfp) in REGON
    name: str
    short: str
    group: str  # natural_person | partnership | capital_company | other
    register: str  # CEIDG | KRS


BUSINESS_FORMS: dict[str, LegalForm] = {
    f.code: f
    for f in (
        LegalForm("099", "osoba fizyczna prowadząca działalność gospodarczą", "JDG", "natural_person", "CEIDG"),
        LegalForm("019", "spółka cywilna", "s.c.", "partnership", "CEIDG"),
        LegalForm("118", "spółka jawna", "sp.j.", "partnership", "KRS"),
        LegalForm("115", "spółka partnerska", "sp.p.", "partnership", "KRS"),
        LegalForm("120", "spółka komandytowa", "sp.k.", "partnership", "KRS"),
        LegalForm("121", "spółka komandytowo-akcyjna", "S.K.A.", "partnership", "KRS"),
        LegalForm("117", "spółka z ograniczoną odpowiedzialnością", "sp. z o.o.", "capital_company", "KRS"),
        LegalForm("113", "prosta spółka akcyjna", "P.S.A.", "capital_company", "KRS"),
        LegalForm("116", "spółka akcyjna", "S.A.", "capital_company", "KRS"),
        LegalForm("122", "spółka europejska", "SE", "capital_company", "KRS"),
        LegalForm("140", "spółdzielnia", "spółdzielnia", "other", "KRS"),
        LegalForm("179", "oddział zagranicznego przedsiębiorcy", "oddział", "other", "KRS"),
    )
}

OTHER_FORM = "other"
"""All remaining forms (foundations, associations, public sector, housing communities...)."""


@dataclass(frozen=True)
class Voivodeship:
    name: str
    teryt: str
    nuts2: str | None  # None: the voivodeship is split into two NUTS 2 regions


VOIVODESHIPS: tuple[Voivodeship, ...] = (
    Voivodeship("dolnośląskie", "02", "PL51"),
    Voivodeship("kujawsko-pomorskie", "04", "PL61"),
    Voivodeship("lubelskie", "06", "PL81"),
    Voivodeship("lubuskie", "08", "PL43"),
    Voivodeship("łódzkie", "10", "PL71"),
    Voivodeship("małopolskie", "12", "PL21"),
    # PL91 warszawski stołeczny and PL92 mazowiecki regionalny; the split needs subregional data.
    Voivodeship("mazowieckie", "14", None),
    Voivodeship("opolskie", "16", "PL52"),
    Voivodeship("podkarpackie", "18", "PL82"),
    Voivodeship("podlaskie", "20", "PL84"),
    Voivodeship("pomorskie", "22", "PL63"),
    Voivodeship("śląskie", "24", "PL22"),
    Voivodeship("świętokrzyskie", "26", "PL72"),
    Voivodeship("warmińsko-mazurskie", "28", "PL62"),
    Voivodeship("wielkopolskie", "30", "PL41"),
    Voivodeship("zachodniopomorskie", "32", "PL42"),
)

VOIVODESHIP_BY_NAME = {v.name: v for v in VOIVODESHIPS}


def normalize_voivodeship(raw: str) -> str | None:
    """'KUJAWSKO-\\n-POMORSKIE ' -> 'kujawsko-pomorskie'; None if it is not a voivodeship.

    GUS breaks names in headers into two lines with a hyphen at both ends,
    and in Table 6 writes them in capitals padded with spaces.
    """
    s = "".join(raw.split()).lower().replace("--", "-")
    if s == "zachodnio-pomorskie":
        s = "zachodniopomorskie"
    return s if s in VOIVODESHIP_BY_NAME else None
