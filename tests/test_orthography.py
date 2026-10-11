import unittest

from rauxa.catalan.conjugation import conjugate, participle
from rauxa.catalan.orthography import (
    Noun,
    definite_np,
    elide_de,
    prepositional_np,
    reflexive_proclitic,
    sentence,
    starts_with_vowel_sound,
)


class TestVowelSound(unittest.TestCase):
    def test_vowels_and_silent_h(self):
        for word in ("anar", "error", "home", "hora", "illa", "usar", "àvia"):
            self.assertTrue(starts_with_vowel_sound(word), word)

    def test_consonants_and_semivowels(self):
        for word in ("fer", "problema", "iogurt", "hiena", "uombat"):
            self.assertFalse(starts_with_vowel_sound(word), word)


class TestArticles(unittest.TestCase):
    def test_definite_np(self):
        cases = [
            (Noun("error", "errors", "m"), "sg", "l'error"),
            (Noun("problema", "problemes", "m"), "sg", "el problema"),
            (Noun("hora", "hores", "f"), "sg", "l'hora"),
            (Noun("situació", "situacions", "f"), "sg", "la situació"),
            (Noun("error", "errors", "m"), "pl", "els errors"),
            (Noun("hora", "hores", "f"), "pl", "les hores"),
            (Noun("universitat", "universitats", "f", elides=False), "sg", "la universitat"),
            (Noun("illa", "illes", "f", elides=True), "sg", "l'illa"),
        ]
        for noun, number, expected in cases:
            self.assertEqual(definite_np(noun, number), expected)

    def test_ambiguous_feminine_requires_flag(self):
        with self.assertRaises(ValueError):
            definite_np(Noun("universitat", "universitats", "f"), "sg")

    def test_contractions(self):
        error = Noun("error", "errors", "m")
        problema = Noun("problema", "problemes", "m")
        hora = Noun("hora", "hores", "f")
        self.assertEqual(prepositional_np("de", problema, "sg"), "del problema")
        self.assertEqual(prepositional_np("de", problema, "pl"), "dels problemes")
        self.assertEqual(prepositional_np("de", error, "sg"), "de l'error")
        self.assertEqual(prepositional_np("a", problema, "sg"), "al problema")
        self.assertEqual(prepositional_np("per", error, "pl"), "pels errors")
        self.assertEqual(prepositional_np("de", hora, "sg"), "de l'hora")
        self.assertEqual(prepositional_np("de", hora, "pl"), "de les hores")


class TestElisionAndClitics(unittest.TestCase):
    def test_elide_de(self):
        self.assertEqual(elide_de("anar"), "d'anar")
        self.assertEqual(elide_de("haver"), "d'haver")
        self.assertEqual(elide_de("fer"), "de fer")

    def test_reflexive_proclitic(self):
        self.assertEqual(reflexive_proclitic("1sg", "adono"), "m'")
        self.assertEqual(reflexive_proclitic("1sg", "dono"), "em")
        self.assertEqual(reflexive_proclitic("3sg", "ha"), "s'")
        self.assertEqual(reflexive_proclitic("1pl", "adonem"), "ens")
        self.assertEqual(reflexive_proclitic("2pl", "heu"), "us")
        self.assertEqual(reflexive_proclitic("3pl", "van"), "es")

    def test_sentence(self):
        self.assertEqual(sentence(["m'", "adono", "de l'error"]), "M'adono de l'error.")
        self.assertEqual(sentence(["en Pau", "ha", "d'anar"]), "En Pau ha d'anar.")


class TestConjugation(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(conjugate("haver", "present", "1sg"), "he")
        self.assertEqual(conjugate("tenir", "present", "3sg"), "té")
        self.assertEqual(conjugate("tenir", "future", "1pl"), "tindrem")
        self.assertEqual(conjugate("adonar", "present", "1sg"), "adono")
        self.assertEqual(conjugate("donar", "present", "3sg"), "dona")  # no diacritic since 2016
        self.assertEqual(conjugate("adonar", "imperfect", "1pl"), "adonàvem")
        self.assertEqual(conjugate("anar_aux", "present", "2pl"), "vau")
        self.assertEqual(participle("haver"), "hagut")
        self.assertEqual(participle("adonar"), "adonat")


if __name__ == "__main__":
    unittest.main()
