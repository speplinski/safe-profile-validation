"""Definition of the stage 0 target population.

The research assumptions require a single target population fixed in the registration, because GUS
(REGON, active enterprises) and PARP count different populations. This definition is
a proposal for decision (entry A0-1 in docs/etap0-populacja.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dictionaries import BUSINESS_FORMS


@dataclass(frozen=True)
class TargetPopulation:
    name: str
    description: str
    forms: frozenset[str]
    excluded_sections: frozenset[str] = field(default_factory=frozenset)


REGON_BUSINESS = TargetPopulation(
    name="regon_nonfinancial_firms",
    description=(
        "Entities in REGON declaring business activity, in business legal forms "
        "(JDG, partnerships and companies, cooperatives, foreign branches), with PKD outside "
        "sections K (finance), O (public administration), T (households) and U "
        "(extraterritorial organisations); excluding entities without PKD or voivodeship."
    ),
    forms=frozenset(BUSINESS_FORMS),
    excluded_sections=frozenset({"K", "O", "T", "U"}),
)

TARGETS = {t.name: t for t in (REGON_BUSINESS,)}
