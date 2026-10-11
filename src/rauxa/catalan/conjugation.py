"""Small verb paradigms (IEC, central variety) for the verbs the templates need.

This is deliberately hand-written and tiny. A full inflection engine (e.g. backed by the
Softcatalà lexicon) can replace it later behind the same ``conjugate`` function.
"""

from __future__ import annotations

from typing import Literal

from .orthography import PERSONS, Person

SimpleTense = Literal["present", "imperfect", "future", "conditional"]

_IRREGULAR: dict[str, dict[str, tuple[str, ...]]] = {
    "haver": {
        "present": ("he", "has", "ha", "hem", "heu", "han"),
        "imperfect": ("havia", "havies", "havia", "havíem", "havíeu", "havien"),
        "future": ("hauré", "hauràs", "haurà", "haurem", "haureu", "hauran"),
        "conditional": ("hauria", "hauries", "hauria", "hauríem", "hauríeu", "haurien"),
        "participle": ("hagut",),
    },
    "tenir": {
        "present": ("tinc", "tens", "té", "tenim", "teniu", "tenen"),
        "imperfect": ("tenia", "tenies", "tenia", "teníem", "teníeu", "tenien"),
        "future": ("tindré", "tindràs", "tindrà", "tindrem", "tindreu", "tindran"),
        "conditional": ("tindria", "tindries", "tindria", "tindríem", "tindríeu", "tindrien"),
        "participle": ("tingut",),
    },
    # Auxiliary of the periphrastic past ("vaig fer"), present forms only.
    "anar_aux": {
        "present": ("vaig", "vas", "va", "vam", "vau", "van"),
    },
}

_AR_ENDINGS: dict[str, tuple[str, ...]] = {
    "present": ("o", "es", "a", "em", "eu", "en"),
    "imperfect": ("ava", "aves", "ava", "àvem", "àveu", "aven"),
    "future": ("aré", "aràs", "arà", "arem", "areu", "aran"),
    "conditional": ("aria", "aries", "aria", "aríem", "aríeu", "arien"),
    "participle": ("at",),
}

# Third-person-singular-only forms of impersonal verbs.
IMPERSONAL: dict[str, dict[str, str]] = {
    "caler": {
        "present": "cal",
        "imperfect": "calia",
        "future": "caldrà",
        "conditional": "caldria",
        "participle": "calgut",
    },
}


def conjugate(lemma: str, tense: SimpleTense, person: Person) -> str:
    """Finite form of ``lemma``. Supports the irregulars above and regular -ar verbs."""
    i = PERSONS.index(person)
    if lemma in _IRREGULAR:
        return _IRREGULAR[lemma][tense][i]
    if lemma.endswith("ar"):
        return lemma[:-2] + _AR_ENDINGS[tense][i]
    raise NotImplementedError(f"no paradigm for '{lemma}'")


def participle(lemma: str) -> str:
    if lemma in _IRREGULAR:
        return _IRREGULAR[lemma]["participle"][0]
    if lemma.endswith("ar"):
        return lemma[:-2] + "at"
    raise NotImplementedError(f"no participle for '{lemma}'")
