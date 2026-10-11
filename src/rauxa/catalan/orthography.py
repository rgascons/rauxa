"""Catalan orthographic rules needed to realise templates correctly.

Covers elision (apostrophation) of articles, ``de`` and weak pronouns, the contractions
``al/als/del/dels/pel/pels``, and joining tokens into a sentence. These are the places where
naive templating produces non-normative text (``de el``, ``la illa``, ``de anar``...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

VOWELS = frozenset("aàeèéiíïoòóuúü")
_STRESSED_IU = frozenset("íú")

Gender = Literal["m", "f"]
Number = Literal["sg", "pl"]
Person = Literal["1sg", "2sg", "3sg", "1pl", "2pl", "3pl"]
PERSONS: tuple[Person, ...] = ("1sg", "2sg", "3sg", "1pl", "2pl", "3pl")


def _strip_h(word: str) -> str:
    w = word.lower()
    return w[1:] if w.startswith("h") else w


def starts_with_vowel_sound(word: str) -> bool:
    """Whether ``word`` triggers elision of a preceding ``de``, ``el``, ``em``...

    Initial ``h`` is silent. Initial ``i``/``u`` followed by a vowel is a semivowel
    (``iogurt``, ``hiena``, ``uombat``) and does not trigger elision.
    """
    w = _strip_h(word)
    if not w or w[0] not in VOWELS:
        return False
    if w[0] in "iu" and len(w) > 1 and w[1] in VOWELS:
        return False
    return True


@dataclass(frozen=True)
class Noun:
    singular: str
    plural: str
    gender: Gender
    # Whether the singular definite article is apostrophised. Only required for feminine
    # nouns starting with unaccented i/u (stress decides: "l'illa" but "la universitat")
    # and for exceptions such as "la una" or negative a- prefixes ("la asimetria").
    elides: Optional[bool] = None

    def form(self, number: Number) -> str:
        return self.singular if number == "sg" else self.plural

    def article_elides(self) -> bool:
        if self.elides is not None:
            return self.elides
        if not starts_with_vowel_sound(self.singular):
            return False
        if self.gender == "f":
            first = _strip_h(self.singular)[0]
            if first in "iuïü":
                raise ValueError(
                    f"feminine noun '{self.singular}' starts with i/u: set `elides` explicitly"
                )
        return True


def definite_article(noun: Noun, number: Number) -> str:
    """``el``, ``la``, ``l'``, ``els`` or ``les`` for ``noun`` (plurals never elide)."""
    if number == "pl":
        return "els" if noun.gender == "m" else "les"
    if noun.article_elides():
        return "l'"
    return "el" if noun.gender == "m" else "la"


_CONTRACTIONS = {
    ("a", "el"): "al",
    ("a", "els"): "als",
    ("de", "el"): "del",
    ("de", "els"): "dels",
    ("per", "el"): "pel",
    ("per", "els"): "pels",
}


def definite_np(noun: Noun, number: Number) -> str:
    """Noun phrase with definite article: ``l'error``, ``el problema``, ``les hores``."""
    return join([definite_article(noun, number), noun.form(number)])


def prepositional_np(prep: str, noun: Noun, number: Number) -> str:
    """``prep`` + definite NP with contraction: ``del problema``, ``de l'error``, ``als alumnes``."""
    article = definite_article(noun, number)
    contracted = _CONTRACTIONS.get((prep, article))
    if contracted:
        return join([contracted, noun.form(number)])
    return join([prep, article, noun.form(number)])


def elide_de(word: str) -> str:
    """``de`` + ``word``: ``d'anar``, ``d'haver``, ``de fer``."""
    return join(["d'" if starts_with_vowel_sound(word) else "de", word])


_REFLEXIVE_PROCLITICS: dict[Person, tuple[str, str]] = {
    "1sg": ("em", "m'"),
    "2sg": ("et", "t'"),
    "3sg": ("es", "s'"),
    "1pl": ("ens", "ens"),
    "2pl": ("us", "us"),
    "3pl": ("es", "s'"),
}


def reflexive_proclitic(person: Person, next_word: str) -> str:
    """Reflexive weak pronoun before a verb: ``em dono`` / ``m'adono``, ``ens adonem``."""
    full, elided = _REFLEXIVE_PROCLITICS[person]
    return elided if starts_with_vowel_sound(next_word) else full


def join(tokens: list[str]) -> str:
    """Join tokens with spaces, attaching tokens that follow an apostrophe (``l'`` + ``error``)."""
    out = ""
    for token in tokens:
        if not token:
            continue
        if out and not out.endswith("'"):
            out += " "
        out += token
    return out


def sentence(tokens: list[str], final: str = ".") -> str:
    """Join tokens into a sentence: capitalise the first letter and add final punctuation."""
    text = join(tokens)
    return text[:1].upper() + text[1:] + final
