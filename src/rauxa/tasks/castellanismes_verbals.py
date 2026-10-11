"""Verbal calques from Spanish (castellanismes sintàctics en perífrasis verbals).

Each item contrasts a normative sentence with the same sentence using a Spanish calque:

- ``tenir_que``: *tenir que* + inf. (cast. *tener que*) → *haver de* + inf.
  "Demà he d'anar al metge." / "Demà tinc que anar al metge."
- ``hi_ha_que``: *hi ha que* + inf. (cast. *hay que*) → *cal* + inf. or *s'ha de* + inf.
  "Ahir va caldre ordenar l'habitació." / "Ahir hi va haver que ordenar l'habitació."
- ``donar_se_compte``: *donar-se compte* (cast. *darse cuenta*) → *adonar-se*.
  "Ens vam adonar de l'error." / "Ens vam donar compte de l'error."

Only the IEC (central) norm is implemented so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Literal, Optional

from ..catalan.conjugation import IMPERSONAL, conjugate, participle
from ..catalan.orthography import (
    PERSONS,
    Noun,
    Person,
    elide_de,
    prepositional_np,
    reflexive_proclitic,
    sentence,
)
from ..curriculum import BaseCurriculum, ScalarAttributeDefinition
from ..dataset import ProceduralDataset
from ..factory import register_dataset
from ..formats import FORMATS, render, score

DATASET_NAME = "castellanismes_verbals"

Tense = Literal["present", "imperfect", "future", "conditional", "periphrastic_past", "perfect"]
TENSES: tuple[Tense, ...] = ("present", "imperfect", "future", "conditional", "periphrastic_past", "perfect")

PHENOMENA = ("tenir_que", "hi_ha_que", "donar_se_compte")

# Tenses each phenomenon can be realised in.
_PHENOMENON_TENSES: dict[str, tuple[Tense, ...]] = {
    "tenir_que": TENSES,
    "hi_ha_que": TENSES,
    "donar_se_compte": ("present", "future", "periphrastic_past", "perfect"),
}

# Phenomena whose negated form changes meaning ("no cal" ≠ "no s'ha de") are never negated.
_NEGATABLE = frozenset({"tenir_que", "donar_se_compte"})

# ---------------------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------------------

_ADVERBS: dict[Tense, tuple[str, ...]] = {
    "present": ("avui", "ara", "aquesta setmana"),
    "imperfect": ("abans", "cada matí", "aquell estiu"),
    "future": ("demà", "la setmana que ve", "aviat"),
    "conditional": ("potser", "segurament"),
    "periphrastic_past": ("ahir", "la setmana passada", "aquell dia"),
    "perfect": ("avui", "aquest matí", "aquesta setmana"),
}

_SUBJECT_PRONOUNS: dict[Person, tuple[str, ...]] = {
    "1sg": ("jo",),
    "2sg": ("tu",),
    "1pl": ("nosaltres",),
    "2pl": ("vosaltres",),
}

# Third persons always get an explicit subject. Personal articles (en/l') follow the
# central variety, which the IEC norm accepts.
_SUBJECT_NPS: dict[Person, tuple[str, ...]] = {
    "3sg": ("la Marta", "en Pau", "l'Anna", "l'Oriol", "el meu germà", "la directora", "ella", "ell"),
    "3pl": ("els veïns", "les meves cosines", "els alumnes", "ells", "elles"),
}

# (infinitive, complement). Mix of vowel- and consonant-initial infinitives to exercise
# elision of "de" (he d'anar / he de comprar).
_INFINITIVE_PHRASES: tuple[tuple[str, str], ...] = (
    ("anar", "al metge"),
    ("anar", "a la reunió"),
    ("acabar", "l'informe abans de divendres"),
    ("escriure", "una carta al director"),
    ("estudiar", "més hores"),
    ("arribar", "abans de les nou"),
    ("esperar", "l'autobús"),
    ("explicar", "el problema als alumnes"),
    ("omplir", "aquest formulari"),
    ("entregar", "els deures"),
    ("ordenar", "l'habitació"),
    ("enviar", "el paquet"),
    ("escoltar", "els consells del metge"),
    ("arreglar", "la bicicleta"),
    ("estalviar", "més diners"),
    ("aprendre", "a cuinar"),
    ("aprovar", "l'examen"),
    ("ajudar", "els veïns"),
    ("obrir", "la botiga"),
    ("aparcar", "el cotxe"),
    ("acompanyar", "la meva mare a l'hospital"),
    ("agafar", "el tren de les set"),
    ("ensenyar", "el pis als compradors"),
    ("trucar", "a l'àvia"),
    ("comprar", "pa i llet"),
    ("parlar", "amb la directora"),
    ("demanar", "permís al professor"),
    ("llegir", "aquest llibre"),
    ("fer", "la compra"),
    ("netejar", "la cuina"),
    ("pagar", "la factura"),
    ("preparar", "el sopar"),
    ("tornar", "a casa"),
    ("canviar", "les rodes del cotxe"),
    ("revisar", "els comptes"),
    ("vendre", "el pis"),
)

# Complements of adonar-se: "que" clauses by time frame, or "de" + NP.
_QUE_CLAUSES: dict[str, tuple[str, ...]] = {
    "nonpast": (
        "que la porta està oberta",
        "que no tenim prou temps",
        "que el pla no funciona",
        "que ningú no ho entén",
        "que falta una peça",
        "que el cotxe fa un soroll estrany",
    ),
    "past": (
        "que la porta estava oberta",
        "que no teníem prou temps",
        "que el pla no funcionava",
        "que ningú no ho entenia",
        "que faltava una peça",
        "que el cotxe feia un soroll estrany",
    ),
}

# (noun, numbers it reads naturally in after "adonar-se de").
_DE_NOUNS: tuple[tuple[Noun, tuple[str, ...]], ...] = (
    (Noun("error", "errors", "m"), ("sg", "pl")),
    (Noun("problema", "problemes", "m"), ("sg", "pl")),
    (Noun("canvi", "canvis", "m"), ("sg", "pl")),
    (Noun("detall", "detalls", "m"), ("sg", "pl")),
    (Noun("perill", "perills", "m"), ("sg", "pl")),
    (Noun("engany", "enganys", "m"), ("sg",)),
    (Noun("situació", "situacions", "f"), ("sg",)),
    (Noun("equivocació", "equivocacions", "f"), ("sg", "pl")),
    (Noun("hora", "hores", "f"), ("sg",)),
)

# ---------------------------------------------------------------------------------------
# Verb chains
# ---------------------------------------------------------------------------------------


def _finite_chain(lemma: str, tense: Tense, person: Person) -> list[str]:
    """Finite verb group: simple form, periphrastic past (vaig + inf.) or perfect (he + part.)."""
    if tense == "periphrastic_past":
        return [conjugate("anar_aux", "present", person), lemma]
    if tense == "perfect":
        return [conjugate("haver", "present", person), participle(lemma)]
    return [conjugate(lemma, tense, person)]


def _caler_chain(tense: Tense) -> list[str]:
    if tense == "periphrastic_past":
        return ["va", "caldre"]
    if tense == "perfect":
        return ["ha", IMPERSONAL["caler"]["participle"]]
    return [IMPERSONAL["caler"][tense]]


def _with_reflexive(person: Person, chain: list[str]) -> list[str]:
    return [reflexive_proclitic(person, chain[0]), *chain]


@dataclass(frozen=True)
class _Realisation:
    good: str
    bad: str
    variants: tuple[str, ...]  # every accepted correction, including ``good``
    details: dict[str, Any]


def _realise_tenir_que(rng: Random, cfg: "CastellanismesVerbalsConfig", tense: Tense) -> _Realisation:
    person = rng.choice(PERSONS)
    prefix = _prefix(rng, cfg, tense, person)
    negated = rng.random() < cfg.negation_prob
    neg = ["no"] if negated else []
    inf, compl = rng.choice(_INFINITIVE_PHRASES)

    good = sentence(prefix + neg + _finite_chain("haver", tense, person) + [elide_de(inf), compl])
    bad = sentence(prefix + neg + _finite_chain("tenir", tense, person) + ["que", inf, compl])
    return _Realisation(good, bad, (good,), {"person": person, "negated": negated, "infinitive": inf})


def _realise_hi_ha_que(rng: Random, cfg: "CastellanismesVerbalsConfig", tense: Tense) -> _Realisation:
    prefix = _adverb(rng, cfg, tense)
    inf, compl = rng.choice(_INFINITIVE_PHRASES)

    cal = sentence(prefix + _caler_chain(tense) + [inf, compl])
    s_ha_de = sentence(prefix + _with_reflexive("3sg", _finite_chain("haver", tense, "3sg")) + [elide_de(inf), compl])
    bad = sentence(prefix + ["hi"] + _finite_chain("haver", tense, "3sg") + ["que", inf, compl])
    good, alternative = (cal, s_ha_de) if rng.random() < 0.5 else (s_ha_de, cal)
    return _Realisation(good, bad, (good, alternative), {"person": None, "negated": False, "infinitive": inf})


def _realise_donar_se_compte(rng: Random, cfg: "CastellanismesVerbalsConfig", tense: Tense) -> _Realisation:
    person = rng.choice(PERSONS)
    prefix = _prefix(rng, cfg, tense, person)
    negated = rng.random() < cfg.negation_prob
    neg = ["no"] if negated else []

    if rng.random() < 0.5:
        frame = "past" if tense in ("periphrastic_past", "perfect") else "nonpast"
        compl = rng.choice(_QUE_CLAUSES[frame])
        complement_type = "que"
    else:
        noun, numbers = rng.choice(_DE_NOUNS)
        number = "pl" if "pl" in numbers and rng.random() < 0.3 else "sg"
        compl = prepositional_np("de", noun, number)
        complement_type = "de"

    good = sentence(prefix + neg + _with_reflexive(person, _finite_chain("adonar", tense, person)) + [compl])
    bad = sentence(prefix + neg + _with_reflexive(person, _finite_chain("donar", tense, person)) + ["compte", compl])
    return _Realisation(
        good, bad, (good,), {"person": person, "negated": negated, "complement_type": complement_type}
    )


_REALISERS = {
    "tenir_que": _realise_tenir_que,
    "hi_ha_que": _realise_hi_ha_que,
    "donar_se_compte": _realise_donar_se_compte,
}


def _adverb(rng: Random, cfg: "CastellanismesVerbalsConfig", tense: Tense) -> list[str]:
    return [rng.choice(_ADVERBS[tense])] if rng.random() < cfg.adverb_prob else []


def _prefix(rng: Random, cfg: "CastellanismesVerbalsConfig", tense: Tense, person: Person) -> list[str]:
    """Optional time adverb followed by an optional (1st/2nd person) or explicit (3rd) subject."""
    tokens = _adverb(rng, cfg, tense)
    if person in _SUBJECT_NPS:
        tokens.append(rng.choice(_SUBJECT_NPS[person]))
    elif rng.random() < cfg.subject_pronoun_prob:
        tokens.append(rng.choice(_SUBJECT_PRONOUNS[person]))
    return tokens


# ---------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------


@dataclass
class CastellanismesVerbalsConfig:
    phenomena: tuple[str, ...] = PHENOMENA
    tenses: tuple[Tense, ...] = ("present", "future", "periphrastic_past", "perfect")
    formats: tuple[str, ...] = ("mcq",)
    negation_prob: float = 0.2
    subject_pronoun_prob: float = 0.3
    adverb_prob: float = 0.6
    norm: str = "iec"
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.phenomena, "phenomena must not be empty"
        assert self.formats, "formats must not be empty"
        for p in self.phenomena:
            assert p in PHENOMENA, f"unknown phenomenon '{p}'; expected one of {PHENOMENA}"
            assert set(self.tenses) & set(_PHENOMENON_TENSES[p]), f"no tense in {self.tenses} supports '{p}'"
        for t in self.tenses:
            assert t in TENSES, f"unknown tense '{t}'; expected one of {TENSES}"
        for f in self.formats:
            assert f in FORMATS, f"unknown format '{f}'; expected one of {FORMATS}"
        for name in ("negation_prob", "subject_pronoun_prob", "adverb_prob"):
            assert 0.0 <= getattr(self, name) <= 1.0, f"{name} must be in [0, 1]"
        if self.norm != "iec":
            raise NotImplementedError(f"norm '{self.norm}' is not implemented yet (only 'iec')")


class CastellanismesVerbalsDataset(ProceduralDataset):
    def __init__(self, config: CastellanismesVerbalsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = Random(self.seed + idx)
        cfg = self.config

        phenomenon = rng.choice(cfg.phenomena)
        tense = rng.choice([t for t in cfg.tenses if t in _PHENOMENON_TENSES[phenomenon]])
        realisation = _REALISERS[phenomenon](rng, cfg, tense)
        fmt = rng.choice(cfg.formats)
        question, answer, extra = render(fmt, realisation.good, realisation.bad, rng)

        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "format": fmt,
                "phenomenon": phenomenon,
                "norm": cfg.norm,
                "tense": tense,
                "sentence_good": realisation.good,
                "sentence_bad": realisation.bad,
                "correct_variants": list(realisation.variants),
                **realisation.details,
                **extra,
                "difficulty": {"tenses": list(cfg.tenses), "negation_prob": cfg.negation_prob},
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        return score(answer, entry)


class CastellanismesVerbalsCurriculum(BaseCurriculum):
    def __init__(self) -> None:
        super().__init__(
            DATASET_NAME,
            CastellanismesVerbalsConfig,
            [
                ScalarAttributeDefinition(
                    name="tenses",
                    field_name="tenses",
                    levels=(
                        ("present",),
                        ("present", "future"),
                        ("present", "future", "periphrastic_past", "perfect"),
                        TENSES,
                    ),
                    description="Verb tenses the construction appears in",
                ),
                ScalarAttributeDefinition(
                    name="negation_prob",
                    field_name="negation_prob",
                    levels=(0.0, 0.2, 0.5),
                    description="Probability of a negated sentence",
                ),
            ],
        )


register_dataset(DATASET_NAME, CastellanismesVerbalsDataset, CastellanismesVerbalsConfig, CastellanismesVerbalsCurriculum)
