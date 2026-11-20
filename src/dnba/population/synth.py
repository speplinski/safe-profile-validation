"""Layer 1 synthesis: joint distribution from the REGON tables and sampling of firms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dictionaries import BUSINESS_FORMS, OTHER_FORM, SIZE_CLASSES, VOIVODESHIP_BY_NAME
from .ipf import Margin, ipf
from .regon import NO_PKD
from .target import TargetPopulation

AXES = ("legal_form", "voivodeship", "pkd_division", "size_class")


@dataclass(frozen=True)
class JointInfo:
    ipf_iterations: int
    ipf_max_abs_error: float
    scaled_out: float  # Table 11 entities without a voivodeship, removed when reconciling the margins
    population_all_forms: float
    population_target: float


def _cube(df: pd.DataFrame, dims: list[str], levels: dict[str, list[str]]) -> np.ndarray:
    idx = pd.MultiIndex.from_product([levels[d] for d in dims], names=dims)
    s = df.groupby(dims)["n"].sum().reindex(idx, fill_value=0)
    return s.to_numpy(dtype=float).reshape([len(levels[d]) for d in dims])


def build_joint(
    forms: pd.DataFrame,
    voivodeships: pd.DataFrame,
    target: TargetPopulation,
    form_voiv: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, JointInfo]:
    """Joint distribution legal form × voivodeship × PKD division × size class.

    Margins: Table 11 (form × division × class) and Table 6 (voivodeship × division × class).
    With two margins sharing axes (division, class) IPF gives the solution with
    conditional independence of form and voivodeship given division and class
    (assumption A0-3). A further margin, e.g. form × voivodeship from BDL, only needs
    to be appended to the `margins` list.

    `form_voiv` (from `krs.form_voivodeship_shares`) adds a third, partial margin
    form × voivodeship for KRS forms: form count from REGON × voivodeship share from KRS
    (A0-7). The other forms (JDG, s.c., other) are not constrained by it.

    Returns (target population, distribution for all forms and sections, fit information).
    """
    f = forms[forms["pkd_division"] != NO_PKD].copy()
    f["legal_form"] = np.where(f["legal_form"].isin(BUSINESS_FORMS), f["legal_form"], OTHER_FORM)
    w = voivodeships[voivodeships["pkd_division"] != NO_PKD]

    levels = {
        "legal_form": sorted(BUSINESS_FORMS) + [OTHER_FORM],
        "voivodeship": sorted(VOIVODESHIP_BY_NAME),
        "pkd_division": sorted(set(f["pkd_division"]) | set(w["pkd_division"])),
        "size_class": list(SIZE_CLASSES),
    }
    m_form = _cube(f, ["legal_form", "pkd_division", "size_class"], levels)
    m_voiv = _cube(w, ["voivodeship", "pkd_division", "size_class"], levels)

    # A0-2: Table 6 has no entities without a voivodeship, Table 11 has them. We reconcile Table 11
    # to the Table 6 sums in every (division, class) cell, proportionally across forms.
    tot_form, tot_voiv = m_form.sum(axis=0), m_voiv.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(tot_form > 0, tot_voiv / tot_form, 0.0)
    scaled_out = float(m_form.sum() - (m_form * scale).sum())
    m_form = m_form * scale

    shape = [len(levels[a]) for a in AXES]
    margins = [Margin((0, 2, 3), m_form), Margin((1, 2, 3), m_voiv)]
    if form_voiv is not None and len(form_voiv):
        m_fw = np.full((len(levels["legal_form"]), len(levels["voivodeship"])), np.nan)
        fi = {f: i for i, f in enumerate(levels["legal_form"])}
        wi = {w: i for i, w in enumerate(levels["voivodeship"])}
        form_totals = m_form.sum(axis=(1, 2))
        for form, g in form_voiv.groupby("legal_form"):
            share = g.set_index("voivodeship")["share"]
            if share.isna().any():
                continue  # unknown cell: the form total cannot be reconciled, form left unconstrained
            share = share / share.sum()  # A0-8: mass removed by the min_cell threshold spread proportionally
            for voiv, p in share.items():
                m_fw[fi[form], wi[voiv]] = p * form_totals[fi[form]]
        margins.append(Margin((0, 1), m_fw))
    res = ipf(np.ones(shape), margins, tol=1e-4, max_iter=5000)
    if not res.converged:
        raise RuntimeError(f"IPF did not converge: error {res.max_abs_error}")

    idx = pd.MultiIndex.from_product([levels[a] for a in AXES], names=AXES)
    joint = pd.Series(res.table.ravel(), index=idx, name="n").reset_index()
    sections = pd.concat([f[["pkd_division", "pkd_section"]], w[["pkd_division", "pkd_section"]]])
    joint = joint.merge(sections.drop_duplicates("pkd_division"), on="pkd_division", how="left")
    joint_all = joint
    all_forms = float(joint["n"].sum())

    keep = joint["legal_form"].isin(target.forms) & ~joint["pkd_section"].isin(target.excluded_sections)
    joint = joint[keep & (joint["n"] > 0)].reset_index(drop=True)
    joint = joint[["legal_form", "voivodeship", "pkd_section", "pkd_division", "size_class", "n"]]
    info = JointInfo(res.iterations, res.max_abs_error, scaled_out, all_forms, float(joint["n"].sum()))
    return joint, joint_all, info


def _allocate(sizes: np.ndarray, n: int, minimum: int) -> np.ndarray:
    """Proportional allocation with a per-stratum minimum (capped at the stratum size)."""
    floor = np.minimum(minimum, np.floor(sizes)).astype(int)
    fixed = np.zeros(len(sizes), dtype=bool)
    for _ in range(len(sizes) + 1):
        rest = n - floor[fixed].sum()
        share = np.where(fixed, 0.0, sizes)
        alloc = np.where(fixed, floor, rest * share / share.sum())
        low = ~fixed & (alloc < floor)
        if not low.any():
            break
        fixed |= low
    alloc = np.where(fixed, floor, alloc)
    out = np.floor(alloc).astype(int)
    # remainders are assigned by largest fractional parts
    short = n - out.sum()
    if short > 0:
        frac = np.where(fixed, -1.0, alloc - out)
        out[np.argsort(-frac)[:short]] += 1
    return out


def sample_firms(
    joint: pd.DataFrame,
    n: int,
    seed: int,
    *,
    strata: list[str] | None = None,
    min_per_stratum: int = 0,
) -> pd.DataFrame:
    """Samples `n` firms from the joint distribution.

    Without `strata`: proportional sampling, every firm has the same weight.
    With `strata`: stratified sampling with a per-stratum minimum (e.g. more 250+ firms),
    weights = stratum size / number sampled in the stratum.
    """
    seq_sample, seq_firm = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(seq_sample)
    total = joint["n"].sum()

    if not strata:
        groups = [(None, joint)]
        alloc = [n]
    else:
        g = joint.groupby(strata, sort=True)
        groups = list(g)
        alloc = _allocate(np.array([d["n"].sum() for _, d in groups]), n, min_per_stratum)

    parts = []
    for (_, cells), k in zip(groups, alloc):
        if k == 0:
            continue
        p = cells["n"].to_numpy() / cells["n"].sum()
        picked = cells.iloc[rng.choice(len(cells), size=k, p=p)].copy()
        picked["weight"] = cells["n"].sum() / k
        parts.append(picked)
    firms = pd.concat(parts, ignore_index=True)
    firms = firms.iloc[rng.permutation(len(firms))].reset_index(drop=True)

    firm_rng = np.random.default_rng(seq_firm)
    firms["firm_seed"] = firm_rng.integers(0, 2**63 - 1, size=len(firms), dtype=np.int64)
    firms.insert(0, "firm_id", [f"F{seed}-{i:06d}" for i in range(len(firms))])

    firms["legal_form_name"] = firms["legal_form"].map(lambda c: BUSINESS_FORMS[c].name)
    firms["legal_form_abbr"] = firms["legal_form"].map(lambda c: BUSINESS_FORMS[c].short)
    firms["form_group"] = firms["legal_form"].map(lambda c: BUSINESS_FORMS[c].group)
    firms["register"] = firms["legal_form"].map(lambda c: BUSINESS_FORMS[c].register)
    firms["voivodeship_teryt"] = firms["voivodeship"].map(lambda v: VOIVODESHIP_BY_NAME[v].teryt)
    firms["nuts2"] = firms["voivodeship"].map(lambda v: VOIVODESHIP_BY_NAME[v].nuts2)
    firms["pkd_version"] = "PKD 2007"
    cols = [
        "firm_id", "legal_form", "legal_form_name", "legal_form_abbr", "form_group", "register",
        "voivodeship", "voivodeship_teryt", "nuts2", "pkd_section", "pkd_division", "pkd_version", "size_class",
        "weight", "firm_seed",
    ]
    assert abs(firms["weight"].sum() - total) / total < 1e-9
    return firms[cols]
