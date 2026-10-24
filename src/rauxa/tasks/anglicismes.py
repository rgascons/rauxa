"""Anglicisms (anglicismes): lexical, semantic and syntactic interference from English.

The phenomena, their assessment and their sources live in ``data/anglicismes_catalog.toml``;
only entries with ``status = "accepted"`` are used. This module adds what the catalog doesn't
hold: sentence frames for each entry, the marker regex and the LanguageTool rules.

The catalog assessment decides the item kinds:

- ``normative_error`` → ``pair`` items: "Hem rebut un correu electrònic." / "Hem rebut un e-mail."
- ``style_preference`` → ``preference`` items (mcq only), never presented as an error:
  "El pont s'està reformant." / "El pont està sent reformat."
- ``accepted`` → ``control`` items only: "Podeu consultar el document a la pàgina web."

Entries with controls (a legitimate use of the suspect word, such as "realitzar un projecte")
also produce ``control`` items, which measure overcorrection.

Frames are written by hand so that agreement is always explicit. Frame syntax:

- ``[good/good2|bad/bad2]`` marks the one span where the sentences differ. The first good
  form is ``sentence_good``; every good form is an accepted correction; the bad form is
  picked at random.
- ``{slot}`` is filled with the same value on both sides; ``{p2}`` draws a second, different
  value from the ``p`` list. Shared slots are below; an entry can add its own.
- A frame may be a ``(frame, context)`` tuple; the context is shown before the item.

Control frames use the same slots and have no ``[...]`` span.

Only the IEC (central) norm is implemented.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib import resources
from random import Random
from typing import Any, Optional, Union

from ..curriculum import BaseCurriculum, ScalarAttributeDefinition
from ..dataset import ProceduralDataset
from ..factory import register_dataset
from ..formats import FORMATS, KIND_FORMATS, KINDS, render, score

DATASET_NAME = "anglicismes"

CATEGORIES = ("lexical", "semantic", "syntactic")


def load_catalog() -> dict[str, Any]:
    with resources.files(__package__).joinpath("data/anglicismes_catalog.toml").open("rb") as f:
        return tomllib.load(f)


CATALOG = load_catalog()
ENTRIES: dict[str, dict[str, Any]] = {e["id"]: e for e in CATALOG["phenomenon"] if e["status"] == "accepted"}

# ---------------------------------------------------------------------------------------
# Shared slots
# ---------------------------------------------------------------------------------------

# Third-person singular subjects. Frames only use them as subjects or after a verb or
# conjunction, never after "a"/"de"/"per", so no contraction or elision is needed.
_PEOPLE = (
    "la Marta", "en Pau", "l'Anna", "l'Oriol", "la Laia", "en Jordi", "l'Ernest", "la Núria",
    "la Clara", "en Marc", "la Berta", "en Quim", "l'Èlia", "en Xavier", "la Montse", "l'Aleix",
    "la directora", "el professor", "la meva veïna", "el meu germà", "la gerent", "el coordinador",
    "la periodista", "l'advocada", "el metge",
)
_WOMEN = ("la Marta", "l'Anna", "la Laia", "la Núria", "la Clara", "la Berta", "l'Èlia", "la Montse", "la Júlia", "la Carla")
_MEN = ("en Pau", "l'Oriol", "en Jordi", "l'Ernest", "en Marc", "en Quim", "en Xavier", "l'Aleix", "en Pere", "en Biel")

# Organisations, used as third-person singular subjects.
_ORGS = (
    "l'ajuntament", "l'empresa", "l'escola", "la universitat", "el departament", "la biblioteca",
    "l'associació", "el club", "la fundació", "l'hospital", "la cooperativa", "el sindicat",
    "l'agència", "la conselleria", "el museu",
)

# Optional time adverbs for periphrastic-past frames ("" leaves the slot empty).
_PAST = ("", "ahir", "la setmana passada", "dilluns", "aquest matí", "fa uns dies", "l'altre dia")

# Full names, for "de {author}": all consonant-initial, so "de" never elides.
_AUTHORS = ("Marta Puig", "Pau Vila", "Berta Sala", "Joan Camps", "Laia Font", "Jordi Roca", "Núria Pons", "Marc Riera")

_CITIES = ("Girona", "Lleida", "Tarragona", "Vic", "Reus", "Manresa", "Berlín", "Londres", "París", "Brussel·les")

_SLOTS: dict[str, tuple[str, ...]] = {"p": _PEOPLE, "pf": _WOMEN, "pm": _MEN, "org": _ORGS, "ap": _PAST, "city": _CITIES, "author": _AUTHORS}

Frame = Union[str, tuple[str, str]]


@dataclass(frozen=True)
class _Spec:
    frames: tuple[Frame, ...]
    controls: tuple[str, ...] = ()  # control frames, added to the catalog's control sentences
    slots: Optional[dict[str, tuple[str, ...]]] = None
    marker: Optional[str] = None  # default for lexical entries: the English forms
    error: Optional[str] = None  # default for lexical entries: built from the catalog
    correction: Optional[str] = None
    # LanguageTool rules that flag the incorrect sentence (default: _LANGUAGETOOL_RULES).
    languagetool_rules: tuple[str, ...] = ()
    # Rules that fire on correct sentences of this entry without making them wrong.
    languagetool_tolerated: tuple[str, ...] = ()


# ---------------------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------------------
#
# English words starting with s + consonant (smartphone, startup, spam, scanner, stress...)
# never follow a singular "el"/"la" or "de": whether to write "l'smartphone" or "el smartphone"
# is itself debatable, and an incorrect sentence must contain exactly one error. Frames use
# "un", plurals, possessives or demonstratives instead.

_LEXICAL: dict[str, _Spec] = {
    "lex.screenshot": _Spec(frames=(
        "{ap} {p} em va enviar [una captura de pantalla|un screenshot] de l'error.",
        "Adjunta [una captura de pantalla|un screenshot] al correu perquè {p} vegi el problema.",
        "{ap} {p} va fer [una captura de pantalla|un screenshot] del missatge abans que l'esborressin.",
        "Per demostrar-ho, {p} ens va ensenyar [unes captures de pantalla|uns screenshots] de la conversa.",
        "[Aquella captura de pantalla|Aquell screenshot] que {p} va publicar ha circulat per totes les xarxes.",
    )),
    "lex.email": _Spec(frames=(
        "{ap} {p} em va enviar [un correu electrònic/un missatge electrònic|un e-mail/un email] amb les instruccions.",
        "Si teniu dubtes, envieu [un correu electrònic/un missatge electrònic|un e-mail/un email] a la secretaria.",
        "{p} encara no ha respost [el meu correu electrònic/el meu missatge electrònic|el meu e-mail/el meu email].",
        "{p} em va demanar [la meva adreça electrònica|el meu e-mail/el meu email] per enviar-me la factura.",
        "{org} envia [un correu electrònic/un missatge electrònic|un e-mail/un email] cada setmana a tots els socis.",
    )),
    "lex.bullying": _Spec(frames=(
        "{org} ha engegat una campanya contra [l'assetjament escolar|el bullying].",
        "{p} va patir [assetjament escolar|bullying] durant tota la primària.",
        "{ap} {p} va explicar com es detecten els casos [d'assetjament escolar|de bullying].",
        "{p} ha escrit un llibre sobre [l'assetjament escolar|el bullying] a les aules.",
    )),
    "lex.coffee_break": _Spec(frames=(
        "Després de la ponència de {author} hi haurà [una pausa|un coffee break/un coffee-break] de mitja hora.",
        "{ap} {p} va aprofitar [la pausa|el coffee break/el coffee-break] per trucar a casa.",
        "Durant [la pausa|el coffee break/el coffee-break], {p} va saludar els ponents.",
        "El programa de la jornada inclou [una pausa|un coffee break/un coffee-break] a les {hour}.",
    ), slots={"hour": ("deu", "onze", "dotze", "cinc", "sis")}),
    "lex.feedback": _Spec(frames=(
        "{ap} {p} ens va fer [molts comentaris|molt feedback] sobre l'esborrany.",
        "Esperem [la resposta/el retorn|el feedback] dels clients abans de continuar.",
        "{p} diu que els alumnes necessiten [un retorn|un feedback] ràpid sobre els exercicis.",
        "{p} va demanar [comentaris|feedback] als companys abans de publicar l'article.",
        "{org} ha rebut [molts comentaris positius|molt feedback positiu] sobre el nou servei.",
    )),
    "lex.online": _Spec(frames=(
        "{org} ofereix un curs [en línia|online/on-line] de primers auxilis.",
        "Ara {p} fa totes les gestions [en línia|online/on-line].",
        "{p} diu que les entrades només es poden comprar [en línia|online].",
        "{ap} {p} va participar en una reunió [en línia|online/on-line] amb els clients.",
    )),
    "lex.offline": _Spec(frames=(
        "{p} diu que l'aplicació també funciona [fora de línia|offline/off-line].",
        "{ap} {p} va descarregar els mapes per consultar-los [fora de línia|offline].",
        "Quan el servidor falla, el sistema de {org_bare} queda [fora de línia|offline/off-line].",
        "{p} prefereix treballar [fora de línia|offline] per evitar distraccions.",
    ), slots={"org_bare": ("l'ajuntament", "l'escola", "la universitat", "la biblioteca", "l'hospital", "la cooperativa")}),
    "lex.smartphone": _Spec(frames=(
        "{ap} {p} es va comprar [un telèfon intel·ligent/un mòbil|un smartphone] nou.",
        "{ap} {p} va perdre [el seu telèfon intel·ligent/el seu mòbil|el seu smartphone] al tren.",
        "{p} diu que l'aplicació es pot descarregar a qualsevol [telèfon intel·ligent/mòbil|smartphone].",
        "{p} encara no té [telèfon intel·ligent/mòbil|smartphone] i no en vol.",
    )),
    "lex.website": _Spec(frames=(
        "{org} ha renovat [el seu lloc web|el seu website].",
        "{p} diu que trobareu tota la informació [al lloc web|al website] de l'entitat.",
        "{p} dissenya [llocs web|websites] per a petites empreses.",
        "{ap} {p} va trobar l'oferta en [un lloc web|un website] de viatges.",
    )),
    "lex.startup": _Spec(frames=(
        "{p} treballa en [una empresa emergent|una startup/una start-up] de {city}.",
        "{p} va fundar [una empresa emergent|una startup/una start-up] quan tenia vint-i-cinc anys.",
        "{org} dona suport a [les empreses emergents|les startups/les start-ups] del territori.",
        "{p} diu que moltes [empreses emergents|startups] tanquen abans del tercer any.",
    )),
    "lex.stakeholder": _Spec(frames=(
        "{org} es reunirà amb [totes les parts interessades|tots els stakeholders] abans de decidir.",
        "{p} diu que cal consultar [les parts interessades|els stakeholders] abans d'aprovar el pla.",
        "{ap} {p} va presentar l'informe [a les parts interessades|als stakeholders].",
        "El projecte de {org_bare} té en compte els interessos de [totes les parts interessades|tots els stakeholders].",
    ), slots={"org_bare": ("l'ajuntament", "l'escola", "la universitat", "la biblioteca", "l'hospital", "la cooperativa")}),
    "lex.coworking": _Spec(frames=(
        "{p} lloga una taula en [un espai de cotreball|un coworking/un espai de coworking] del centre de {city}.",
        "{org} ha obert [un espai de cotreball|un coworking/un espai de coworking] per a autònoms.",
        "{p} diu que [el cotreball|el coworking] és cada vegada més habitual.",
        "{p} prefereix [el cotreball|el coworking] a la feina des de casa.",
    )),
    "lex.spam": _Spec(frames=(
        "{p} diu que cada dia rep massa [correu brossa|spam].",
        "{ap} {p} va marcar el missatge com a [correu brossa|spam].",
        "{org} envia tants anuncis que sembla [correu brossa|spam].",
        "{p} no obre mai els enllaços que arriben com a [correu brossa|spam].",
    )),
    "lex.clickbait": _Spec(frames=(
        "{p} diu que aquell titular és [un pescaclics|un clickbait] de manual.",
        "{p} critica els mitjans que viuen [del pescaclics|del clickbait].",
        "Segons {p}, molts diaris digitals abusen [del pescaclics|del clickbait].",
        "{p} diu que això no és periodisme, que és [pescaclics|clickbait].",
        "{ap} {p} va publicar un article sobre [el pescaclics|el clickbait].",
    )),
    "lex.target": _Spec(frames=(
        "[El públic objectiu|El target] de la campanya de {org_bare} són els joves.",
        "{org} vol arribar a [un públic objectiu|un target] més ampli.",
        "{ap} {p} va analitzar [el públic objectiu|el target] abans de dissenyar l'anunci.",
        "{p} diu que aquest producte té [un públic objectiu|un target] molt definit.",
    # No controls: the TARGET payment system read as nonsense to a native reviewer (2026-10-24).
    ), slots={"org_bare": ("l'ajuntament", "l'empresa", "la universitat", "la biblioteca", "la cooperativa", "la fundació")}),
    "lex.know_how": _Spec(frames=(
        "{p} aporta molt de [saber fer|know-how/know how] a l'equip.",
        "[El saber fer|El know-how/El know how] de {org_bare} és el seu principal actiu.",
        "{org} vol conservar [el saber fer|el know-how/el know how] dels treballadors més veterans.",
        "{org} vol transmetre [el seu saber fer|el seu know-how] a les noves generacions.",
        "{ap} {p} va explicar que [el saber fer|el know-how/el know how] no s'aprèn als llibres.",
        "{p} diu que aquest ofici requereix [un saber fer|un know-how/un know how] que només dona l'experiència.",
    ), slots={"org_bare": ("l'empresa", "la cooperativa", "la fundació", "l'agència", "la fàbrica", "la companyia")}),
    "lex.merchandising": _Spec(frames=(
        "Al final {event}, {p} va comprar [productes de marxandatge/articles promocionals|merchandising].",
        "{org} ingressa molts diners amb la venda [de productes de marxandatge/d'articles promocionals|de merchandising].",
        "{p} és especialista en [marxandatge|merchandising] i disseny d'aparadors.",
        "{ap} {p} va fer un curs de [marxandatge|merchandising].",
        "{p} diu que les tècniques de [marxandatge|merchandising] influeixen en el que comprem.",
    ), slots={"event": ("del concert", "del partit", "del festival", "de la gira", "de l'espectacle")}),
    "lex.lockdown": _Spec(frames=(
        "Durant [el confinament|el lockdown], {p} va aprendre a fer pa.",
        "{ap} el govern va decretar [un confinament|un lockdown] de dues setmanes a {city}.",
        "{p} diu que [el confinament|el lockdown] va canviar els hàbits de molta gent.",
        "{p} va passar [el confinament|el lockdown] a casa dels pares.",
        "{org} va tancar les portes durant [el confinament|el lockdown].",
        "{ap} {p} va recordar com va viure [el confinament|el lockdown].",
    )),
    "lex.link": _Spec(frames=(
        "{ap} {p} em va enviar [l'enllaç|el link] de la reunió.",
        "{p} diu que cal fer clic [a l'enllaç|al link] per confirmar la inscripció.",
        "{p} ha penjat [un enllaç|un link] al document compartit.",
        "{p} diu que [aquest enllaç|aquest link] ja no funciona.",
    )),
    "lex.password": _Spec(frames=(
        "{p} ha oblidat [la contrasenya|el password] del correu.",
        "{p} diu que cal canviar [la contrasenya|el password] cada tres mesos.",
        "{p} diu que [la contrasenya|el password] ha de tenir almenys vuit caràcters.",
        "{ap} {p} em va demanar [la contrasenya|el password] de la xarxa.",
    )),
    "lex.deadline": _Spec(frames=(
        "{p} diu que [el termini/la data límit|el deadline] per presentar les sol·licituds acaba divendres.",
        "{p} treballa a contrarellotge perquè [el termini|el deadline] s'acaba demà.",
        "{ap} {p} va preguntar quin era [el termini|el deadline] del projecte.",
        "{org} ha ampliat [el termini|el deadline] una setmana.",
    )),
    "lex.timeline": _Spec(frames=(
        "{p} ha preparat [una cronologia/una línia de temps|un timeline] {history}.",
        "{ap} {p} va publicar [una cronologia|un timeline] {history}.",
        "L'exposició sobre {artist} comença amb [una cronologia|un timeline] de la seva vida.",
        "{p} diu que els alumnes han de completar [la línia de temps/la cronologia|el timeline] del tema.",
        "Al final del llibre de {author} hi ha [una cronologia|un timeline] dels fets.",
    ), slots={"artist": ("Miró", "Dalí", "Picasso", "Tàpies", "Gaudí", "Rusiñol"),
              "history": ("de la Guerra Civil", "de la Renaixença", "del segle XX",
                          "de la història del barri", "de la Transició")}),
    "lex.workshop": _Spec(frames=(
        "{p} farà [un taller|un workshop] de fotografia dissabte.",
        "{org} organitza [un taller/un seminari|un workshop] sobre intel·ligència artificial.",
        "{ap} {p} es va inscriure [al taller|al workshop] de cuina.",
        "{p} diu que [els tallers|els workshops] del congrés seran a la tarda.",
    )),
    "lex.adapt.scanner": _Spec(frames=(
        "{p} diu que hem de passar el contracte per [un escàner|un scanner].",
        "{ap} {p} va comprar [un escàner|un scanner] per digitalitzar fotos antigues.",
        "{p} diu que [aquest escàner|aquest scanner] no funciona.",
        "{ap} {p} va fer servir [un escàner|un scanner] de la biblioteca.",
    ), controls=(
        "{ap} {p} va passar les factures per l'escàner.",
        "{org} ha comprat un escàner nou.",
    )),
    "lex.adapt.stress": _Spec(frames=(
        "{p} pateix [estrès|stress] per culpa de la feina.",
        "{ap} {p} va agafar la baixa per [estrès|stress].",
        "{p} busca una manera de viure sense [estrès|stress].",
        "{p} diu que la calor també causa [estrès|stress] als animals.",
    ), controls=(
        "{p} pateix estrès des que va canviar de feina.",
        "{ap} {p} va parlar de l'estrès laboral a la ràdio.",
    )),
    "lex.adapt.leader": _Spec(frames=(
        "[El líder|El leader] del partit ha convocat una roda de premsa a {city}.",
        "[Els líders|Els leaders] europeus es reuneixen demà a {city}.",
        "{ap} {p} va entrevistar [el líder|el leader] de l'oposició.",
        "{org} busca [un líder|un leader] per al nou projecte.",
    ), controls=(
        "{org} ha triat un nou líder.",
        "{ap} {p} va entrevistar el líder del sindicat.",
    )),
    "lex.adapt.podcast": _Spec(frames=(
        "{p} escolta [un pòdcast|un podcast] {topic} cada matí.",
        "{org} ha estrenat [un pòdcast|un podcast] {topic}.",
        "{p} diu que podeu escoltar l'entrevista [al nostre pòdcast|al nostre podcast].",
        "{p} grava [pòdcasts|podcasts] des de casa.",
    ), slots={"topic": ("d'història", "de ciència", "de cuina", "d'economia", "de política", "de música")}),
    # An accepted loan: no error, controls only. The texts below are what LLM prompts show.
    "acc.web": _Spec(
        frames=(),
        controls=(
            "{org} ha estrenat una pàgina web nova.",
            "{p} dissenya pàgines web per a petites empreses.",
            "{p} diu que les bases es poden consultar al lloc web.",
            "{ap} {p} va publicar les fotos a la pàgina web del club.",
        ),
        error="Cap: «web» és un préstec acceptat (DIEC2); canviar-lo seria una correcció innecessària.",
        correction="Deixar «web».",
    ),
}

_SEMANTIC: dict[str, _Spec] = {
    "sem.realitzar_adonar": _Spec(
        frames=(
            "{ap} {p} [es va adonar|va realitzar] que havia perdut les claus.",
            "Quan {p} [es va adonar de|va realitzar] l'error, ja era massa tard.",
            "{p} no [s'adona de|realitza] la gravetat de la situació.",
            "De sobte, {p} [es va adonar|va realitzar] que no hi havia ningú més a l'oficina.",
        ),
        controls=(
            "{org} va realitzar una enquesta entre els veïns.",
            "{p} va realitzar el somni de viatjar a l'Índia.",
        ),
        marker=r"\brealitz\w*",
        error="«realitzar» en el sentit de l'anglès «realize» (adonar-se).",
        correction="«adonar-se (de / que)».",
    ),
    "sem.sever": _Spec(
        frames=(
            "{p} pateix una {illness} [greu|severa] des de fa {span}.",
        "{p} té unes migranyes molt [fortes/intenses|severes].",
            "{ap} la tempesta va causar danys [greus|severs] {place}.",
            "Els metges van detectar a {patient} una infecció [greu|severa] al pulmó.",
            "Aquest hivern a {city} hi ha hagut unes gelades molt [fortes/intenses|severes].",
            "{org} ha patit pèrdues [greus|severes] aquest any.",
        ),
        controls=(
            "El jutge va ser molt sever amb l'acusat.",
            "El nou director de {org_bare} té fama de ser sever.",
            "{ap} {pm} va ser molt sever amb els alumnes.",
            "{ap} {pf} va ser molt severa amb els alumnes.",
        ),
        slots={"patient": ("la Marta", "l'Anna", "la Laia", "l'Oriol", "l'Ernest", "la Núria", "la Clara", "la Berta"),
               "illness": ("malaltia", "infecció", "lesió", "pneumònia", "al·lèrgia"),
               "span": ("anys", "mesos", "molt de temps"),
               "place": ("a la collita", "a les carreteres", "al port", "als conreus", "a la xarxa elèctrica"),
               "org_bare": ("l'escola", "l'institut", "la biblioteca", "l'hospital", "la fundació", "l'empresa")},
        marker=r"\bsever(a|s|es)?\b",
        error="«sever» en el sentit de l'anglès «severe» (greu, intens).",
        correction="«greu», «intens», «fort», segons el context.",
    ),
    "sem.projectar_preveure": _Spec(
        frames=(
            "{org} [preveu|projecta] {goal}.",
            "Els economistes [preveuen|projecten] un creixement del dos per cent per a {city}.",
            "{org} [va preveure|va projectar] {sales}.",
            "{p} [calcula/preveu|projecta] que {work} estarà acabada l'any vinent.",
        ),
        controls=("{ap} {p} va projectar les fotos del viatge a la paret.",),
        slots={"goal": ("tancar l'any amb beneficis", "obrir tres botigues més", "contractar vint persones",
                        "duplicar les vendes"),
               "sales": ("unes vendes molt superiors a les reals", "uns ingressos més alts", "un creixement més ràpid"),
               "work": ("l'obra", "la reforma", "la nova escola", "la carretera")},
        marker=r"\bproject\w*",
        error="«projectar» en el sentit de l'anglès «project» (preveure).",
        correction="«preveure», «calcular», «pronosticar».",
    ),
    "sem.eventualment_finalment": _Spec(
        frames=(
            "Després de mesos de negociació, [finalment/al final|eventualment] {org} va signar l'acord.",
            "{p} va dubtar molt, però [finalment/al final|eventualment] va acceptar la feina.",
            ("Vam esperar dues hores a l'andana i [finalment/al final|eventualment] el tren va arribar.",
             "Un viatger explica com va acabar un retard."),
            "[Amb el temps/Finalment|Eventualment], {p} es va acostumar a viure a {city}.",
        ),
        controls=(
            "Si eventualment {p} no pogués venir, ens avisaria.",
            "Eventualment, si plou molt, es pot suspendre l'acte.",
            "{p} diu que, eventualment, es podria ajornar la reunió.",
            "Eventualment, {org} podria tancar alguns serveis.",
        ),
        marker=r"\beventualment\b",
        error="«eventualment» en el sentit de l'anglès «eventually» (finalment).",
        correction="«finalment», «al final», «amb el temps».",
    ),
    "sem.aplicar_sollicitar": _Spec(
        frames=(
            "{p} [ha sol·licitat|ha aplicat a] una beca per estudiar a {city}.",
            "{ap} {p} [es va presentar a/va optar a|va aplicar a] una plaça de professor.",
            "Més de cent persones [han sol·licitat|han aplicat per a] la plaça de {job}.",
            "{p} diu que si voleu [sol·licitar|aplicar a] la beca, cal omplir el formulari.",
        ),
        controls=(
            "El metge li va aplicar una pomada a la ferida.",
            "{org} aplicarà la nova normativa a partir de gener.",
            "{ap} {p} va aplicar el descompte a tots els clients.",
        ),
        slots={"job": ("conserge", "bibliotecari", "conductor", "tècnic de so", "jardiner", "cuiner")},
        marker=r"\bapli(c|qu)\w*",
        error="«aplicar a / per a» en el sentit de l'anglès «apply for» (una feina, una beca).",
        correction="«sol·licitar», «presentar-se a», «optar a».",
    ),
    "sem.condicio_malaltia": _Spec(
        frames=(
            "{p} té [una malaltia|una condició] que li impedeix {activity}.",
            "Els pacients amb [aquesta malaltia/aquesta afecció|aquesta condició] necessiten un seguiment constant.",
            "{p} pateix [una malaltia crònica|una condició crònica] des de fa {span}.",
            "{p} diu que la seva [malaltia|condició] cardíaca {duty}.",
        ),
        controls=(
            "{p} va acceptar la feina amb una condició: treballar des de casa.",
            "{ap} {p} va posar una condició per signar el contracte.",
        ),
        slots={"activity": ("fer esport", "treballar de nit", "viatjar en avió", "pujar escales"),
               "span": ("anys", "mesos", "molt de temps"),
               "duty": ("l'obliga a prendre medicació", "l'obliga a fer revisions periòdiques", "li impedeix córrer")},
        marker=r"\bcondici(ó|ons)\b",
        error="«condició» en el sentit de l'anglès «condition» (malaltia).",
        correction="«malaltia», «afecció», «trastorn».",
    ),
    "sem.allegat_presumpte": _Spec(
        frames=(
            "La policia de {city} investiga [el presumpte/el suposat|l'al·legat] {crime}.",
            "{p} ha negat [les presumptes/les suposades|les al·legades] {charges}.",
            "{ap} el jutge va interrogar [el presumpte/el suposat|l'al·legat] {culprit}.",
            "{org} desmenteix [el presumpte/el suposat|l'al·legat] {crime}.",
        ),
        controls=(
            "{ap} la defensa va presentar un al·legat a favor de l'acusat.",
            "{ap} {p} va escriure un al·legat contra la guerra.",
            "{ap} el fiscal va rebatre l'al·legat final de la defensa.",
        ),
        slots={"crime": ("frau", "suborn", "desfalc", "cas de corrupció", "robatori de dades", "tràfic d'influències"),
               "charges": ("irregularitats", "amenaces", "pressions", "filtracions"),
               "culprit": ("autor del robatori", "responsable de l'estafa", "cap de la xarxa", "autor de les pintades")},
        marker=r"\bal·lega(t|ts|da|des)\b",
        error="«al·legat» com a adjectiu, en el sentit de l'anglès «alleged» (presumpte).",
        correction="«presumpte», «suposat».",
    ),
    "sem.replicar_repetir": _Spec(
        frames=(
            "Els investigadors de {org_bare} no han pogut [repetir/reproduir|replicar] l'experiment.",
            "{org} vol [reproduir|replicar] el model en altres ciutats.",
            "{ap} {p} va [repetir|replicar] la prova tres vegades.",
            "{p} diu que és difícil [reproduir|replicar] els resultats d'aquell estudi.",
        ),
        controls=("{ap} {p} va replicar que no estava d'acord.",),
        slots={"org_bare": ("la universitat", "l'hospital", "l'institut", "la fundació")},
        marker=r"\brepli(c|qu)\w*",
        error="«replicar» en el sentit de l'anglès «replicate» (repetir, reproduir).",
        correction="«repetir», «reproduir».",
    ),
    "sem.basat_a": _Spec(
        frames=(
            "{p} treballa per a una empresa [amb seu a/establerta a|basada a] {city}.",
            "Un fotògraf [establert a/radicat a|basat a] {city} ha guanyat el premi.",
            "Les empreses [amb seu a/establertes a|basades a] {city} rebran ajuts.",
            "Una periodista [establerta a/radicada a|basada a] {city} cobreix la cimera.",
        ),
        controls=(
            "La pel·lícula està basada en fets reals.",
            "{p} té un pla basat en dades fiables.",
            "{ap} {p} va presentar un estudi basat en {data}.",
        ),
        slots={"data": ("enquestes", "dades oficials", "entrevistes")},
        marker=r"\bbasa(t|da|ts|des) a\b",
        error="«basat a» en el sentit de l'anglès «based in» (establert en un lloc).",
        correction="«establert a», «amb seu a», «radicat a».",
    ),
    "sem.evidencia_proves": _Spec(
        frames=(
            "La policia no ha trobat cap [prova|evidència] contra {p}.",
            "{ap} {p} va aportar [proves|evidències] de la seva innocència.",
            "Hi ha [indicis/proves|evidències] que {p} coneixia el sospitós.",
            "{ap} l'advocada de la família va presentar noves [proves|evidències] al jutge.",
            "{org} ha reunit prou [proves|evidències] per denunciar l'empresa.",
        ),
        controls=(
            "{p} es nega a acceptar l'evidència.",
            "{ap} {p} va posar en evidència les contradiccions del govern.",
        ),
        marker=r"\bevidènci(a|es)\b",
        error="«evidència» en el sentit de l'anglès «evidence» (prova), en un context general.",
        correction="«prova», «indici».",
    ),
}

_CAUSES = (
    "la vaga", "la tempesta", "la crisi", "la decisió del govern", "la nova llei",
    "la fusió de les dues empreses", "la manca de pluja", "l'apagada",
)
_CONSEQUENCES = (
    "el tancament de la fàbrica", "retards a tota la línia", "un augment dels preus",
    "la pèrdua de molts llocs de treball", "un acord inesperat", "una forta caiguda de les vendes",
    "la cancel·lació del concert",
)

_SYNTACTIC: dict[str, _Spec] = {
    "syn.resultar_en": _Spec(
        frames=(
            "{ap} {cause} [va provocar/va ocasionar|va resultar en] {conseq}.",
            "Segons {p}, {cause} [ha provocat/ha comportat|ha resultat en] {conseq}.",
            "{p} va cometre un error que [va provocar|va resultar en] la pèrdua de dades.",
        ),
        controls=(
            "Aquests problemes resulten de la manca d'inversió.",
            "L'examen va resultar més fàcil del que {p} esperava.",
            "{ap} {p} va dir que els problemes resultaven de {cause}.",
        ),
        slots={"cause": _CAUSES, "conseq": _CONSEQUENCES},
        marker=r"\bresult\w* en\b",
        error="«resultar en» per introduir la conseqüència (calc de l'anglès «result in»).",
        correction="«provocar», «ocasionar», «comportar», «donar com a resultat».",
    ),
    "syn.gerundi_especificatiu": _Spec(
        frames=(
            "{org} ha aprovat una normativa [que regula|regulant] {topic}.",
            "{ap} {p} va rebre una carta [que anunciava|anunciant] {news}.",
            "{ap} {p} va enviar una caixa [que contenia|contenint] {stuff}.",
            "S'ha publicat un decret [que estableix|establint] {rule}.",
            "{ap} {p} va sortir de casa a les vuit[ i va arribar|, arribant] a la feina a les nou.",
        ),
        controls=(
            "{ap} {p} va entrar cantant a l'aula.",
            "Estudiant cada dia, aprovaràs l'examen.",
        ),
        slots={
            "topic": ("l'ús dels patinets", "els horaris comercials", "el soroll nocturn", "les terrasses dels bars",
                      "l'aparcament al centre", "la recollida de residus"),
            "news": ("el tancament de l'oficina", "una pujada de preus", "canvis en l'horari",
                     "la fusió de les dues empreses", "un nou servei"),
            "stuff": ("llibres antics", "documents personals", "fotografies de família", "eines de jardineria", "roba d'hivern"),
            "rule": ("les noves taxes", "els requisits de la convocatòria", "el calendari escolar", "les ajudes per al lloguer"),
        },
        marker=r"\b(regulant|anunciant|contenint|establint|arribant)\b",
        error="Gerundi especificatiu o de posterioritat (calc de l'anglès i del castellà).",
        correction="Una oració de relatiu («que regula») o una coordinada («i va arribar»).",
    ),
    "syn.possessiu_cos": _Spec(
        frames=(
            "{ap} {p} va aixecar [la mà|la seva mà] per demanar la paraula.",
            "{ap} {p} es va trencar [el braç|el seu braç] jugant a futbol.",
            "{p} sempre es renta [les mans|les seves mans] abans de dinar.",
            "{p} va tancar [els ulls|els seus ulls] i es va adormir.",
            "{p} es raspalla [les dents|les seves dents] tres vegades al dia.",
        ),
        controls=(
            "{ap} {p} va deixar la seva bicicleta al pati.",
            "{ap} {p} va trucar a la seva germana.",
        ),
        marker=r"\b(el seu|la seva|els seus|les seves) (mà|mans|braç|braços|ulls|dents|cap|cames|peus)\b",
        error="Possessiu amb parts del cos quan el posseïdor ja és clar (calc de l'anglès «raise your hand»).",
        correction="L'article definit («aixecar la mà») o el datiu possessiu («li ploren els ulls»).",
    ),
    "syn.passiva_datiu": _Spec(
        frames=(
            "[Van concedir una beca a {pf}/Es va concedir una beca a {pf}/Una beca va ser concedida a {pf}|{pf} va ser concedida una beca] {why}.",
            "[Van concedir un premi a {pm}/Es va concedir un premi a {pm}/Un premi va ser concedit a {pm}|{pm} va ser concedit un premi] {why}.",
            "[Van oferir un contracte a {pf}/Es va oferir un contracte a {pf}|{pf} va ser oferta un contracte] {when}.",
            "[Van assignar un despatx a {pm}/Es va assignar un despatx a {pm}|{pm} va ser assignat un despatx] {when}.",
            "[Van enviar una carta a {pf}/Es va enviar una carta a {pf}|{pf} va ser enviada una carta] {when}.",
            "[Van regalar un rellotge a {pm}/Es va regalar un rellotge a {pm}|{pm} va ser regalat un rellotge] {occasion}.",
        ),
        controls=(
            "{ap} {pf} va ser elegida {role_f}.",
            "{ap} {pm} va ser elegit {role_m}.",
            "El contracte va ser signat pels dos socis.",
        ),
        slots={"why": ("", "per la seva tesi", "pel seu treball", "per estudiar a l'estranger"),
               "occasion": ("", "per la jubilació", "pel seu aniversari", "després de trenta anys a l'empresa"),
               "when": ("", "la setmana passada", "després de l'entrevista", "aquest matí"),
               "role_f": ("delegada de curs", "presidenta de l'associació", "directora del centre"),
               "role_m": ("delegat de curs", "president de l'associació", "director del centre")},
        marker=r"\bser \w+(t|da|ta) (un|una)\b",
        error="Passiva amb el complement indirecte com a subjecte (calc de l'anglès «Mary was given a prize»).",
        correction="Veu activa, passiva pronominal o passiva amb el complement directe com a subjecte.",
    ),
    "syn.estar_sent_participi": _Spec(
        frames=(
            "{msg} [s'està reformant|està sent reformat] {while}.",
            "{fsg} [s'està restaurant|està sent restaurada] {while}.",
            "{mpl} [s'estan revisant|estan sent revisats] {while}.",
            "{fpl} [s'estan tramitant|estan sent tramitades] {while}.",
        ),
        slots={
            "msg": ("el pont", "el museu", "el mercat", "el teatre", "el pavelló", "el castell", "l'estadi", "l'hospital"),
            "fsg": ("la catedral", "la muralla", "l'església", "la biblioteca", "la masia", "l'estació"),
            "mpl": ("els documents", "els pressupostos", "els contractes", "els expedients", "els exàmens"),
            "fpl": ("les sol·licituds", "les factures", "les queixes", "les reclamacions", "les inscripcions"),
            "while": ("des de fa mesos", "aquesta setmana", "en aquest moment", "ara mateix", "des del gener",
                      "durant aquests dies", "des de l'estiu passat"),
        },
        marker=r"\best\w* sent\b",
        error="«estar sent» + participi (passiva progressiva, calc de l'anglès «is being done»).",
        correction="Passiva pronominal («s'està reformant») o veu activa.",
    ),
    "syn.tambe_ho_fa": _Spec(
        frames=(
            "{p} {act} i {p2} també[| ho fa].",
            "{p} no ha vingut a la reunió i {p2} tampoc[| ho ha fet].",
            "{ap} {p} va votar a favor de la proposta, i {p2} també[| ho va fer].",
        ),
        slots={"act": ("fa classes de francès", "juga a escacs", "toca el piano", "estudia medicina", "treballa a l'hospital",
                       "viu a Girona", "fa ioga", "canta en una coral", "llegeix el diari cada matí", "va a la feina amb bicicleta")},
        marker=r"\bho (fa|fan|feia|va fer|ha fet)\b",
        error="Proverb «fer-ho» redundant després d'un verb ja expressat (calc de l'anglès «so does he»).",
        correction="Ometre el proverb: «i en Pau també».",
    ),
    "syn.prendre_accio": _Spec(
        frames=(
            "{org} [ha pres mesures/ha actuat|ha pres acció] contra {problem}.",
            "{p} diu que cal [prendre mesures/passar a l'acció|prendre acció] abans que sigui massa tard.",
            "{ap} {p} va demanar al govern que [prengués mesures|prengués acció] contra {problem}.",
            "{ap} {org} va [prendre mesures/adoptar mesures|prendre acció] després de les queixes dels veïns.",
        ),
        slots={"problem": ("la contaminació", "el soroll", "l'abandonament escolar", "el frau fiscal", "l'incivisme",
                           "la sequera")},
        marker=r"\b(prendre|pren\w*|pres|prengu\w*) acci(ó|ons)\b",
        error="«prendre acció» (calc de l'anglès «take action»).",
        correction="«prendre mesures», «adoptar mesures», «actuar».",
    ),
}

SPECS: dict[str, _Spec] = {**_LEXICAL, **_SEMANTIC, **_SYNTACTIC}

# Where the suspect form is correct, for the LLM writer and judge of control items. The
# catalog's `scope` describes the error, so it would mislead them.
_CORRECT_USES: dict[str, str] = {
    "lex.adapt.scanner": "the adapted loan 'escàner', which is in DIEC2.",
    "lex.adapt.stress": "the adapted loan 'estrès', which is in DIEC2.",
    "lex.adapt.leader": "the adapted loan 'líder', which is in DIEC2.",
    "acc.web": "'web' for the web as a system ('el web') and in 'pàgina web', 'lloc web'. It is an accepted loan (DIEC2).",
    "sem.realitzar_adonar": "'realitzar' meaning carry out or accomplish ('realitzar un projecte'), and 'realitzar-se' (self-fulfilment).",
    "sem.sever": "'sever' about people or style: strict, not indulgent, austere ('un mestre sever', 'un estil sever').",
    "sem.projectar_preveure": "'projectar' meaning cast light or images, or design a construction ('projectar una pel·lícula', 'projectar un edifici').",
    "sem.eventualment_finalment": "'eventualment' meaning occasionally or possibly, if it happens ('si eventualment no pogués venir').",
    "sem.aplicar_sollicitar": "'aplicar' meaning put on, or make a rule, treatment or procedure act on something; 'aplicar-se' meaning make an effort.",
    "sem.condicio_malaltia": "'condició' as a requirement, a circumstance or a social status ('a condició que', 'en bones condicions').",
    "sem.allegat_presumpte": "the noun 'al·legat', a plea or statement of reasons ('un al·legat de la defensa').",
    "sem.replicar_repetir": "'replicar' meaning answer back or object ('va replicar que no'); in finance, 'replicar un índex'.",
    "sem.basat_a": "'basat en' meaning founded on data, facts or a theory ('basada en fets reals').",
    "sem.evidencia_proves": "'evidència' as the quality of being evident ('negar l'evidència', 'posar en evidència'), and in specialised uses ('medicina basada en l'evidència', 'evidència científica', audit evidence).",
    "syn.resultar_en": "'resultar de' (originate from) and 'resultar' + adjective ('va resultar fàcil').",
    "syn.gerundi_especificatiu": "gerunds of manner, cause, condition or simultaneity ('va arribar corrent', 'estudiant cada dia, aprovaràs').",
    "syn.possessiu_cos": "possessives with objects and kinship nouns ('la seva bicicleta', 'la seva germana'), where they are normal.",
    "syn.passiva_datiu": "passives whose subject is the direct object ('El premi va ser lliurat a la Maria') or that take a predicative ('va ser elegida delegada').",
}

# Rules that flag the incorrect sentences, observed with LanguageTool 6.6 by
# scripts/languagetool_probe.py. Entries left out are not caught by LanguageTool: lex.email,
# sem.realitzar_adonar, sem.eventualment_finalment, sem.aplicar_sollicitar, sem.condicio_malaltia,
# sem.allegat_presumpte, syn.gerundi_especificatiu, syn.passiva_datiu, and the preferences
# sem.evidencia_proves and syn.tambe_ho_fa. SEVER, PROJECTAR, REPLICAR_REPETIR and the
# possessive rules catch only some frames.
_LANGUAGETOOL_RULES: dict[str, tuple[str, ...]] = {
    "lex.screenshot": ("CA_SIMPLE_REPLACE_ANGLICISM_SCREENSHOT", "CA_SIMPLE_REPLACE_ANGLICISM_SCREENSHOTS"),
    "lex.bullying": ("CA_SIMPLE_REPLACE_ANGLICISM_BULLYING",),
    "lex.coffee_break": ("CA_SIMPLE_REPLACE_ANGLICISM_COFFEE_BREAK",),
    "lex.feedback": ("CA_SIMPLE_REPLACE_ANGLICISM_FEEDBACK", "CA_SIMPLE_REPLACE_ANGLICISM_EL_FEEDBACK"),
    "lex.online": ("CA_SIMPLE_REPLACE_ANGLICISM_ONLINE", "CA_SIMPLE_REPLACE_ANGLICISM_ON_LINE"),
    "lex.offline": ("CA_SIMPLE_REPLACE_ANGLICISM_OFFLINE", "CA_SIMPLE_REPLACE_ANGLICISM_OFF_LINE"),
    "lex.smartphone": ("CA_SIMPLE_REPLACE_ANGLICISM_SMARTPHONE",),
    "lex.website": ("CA_SIMPLE_REPLACE_ANGLICISM_WEBSITE",),
    "lex.startup": ("CA_SIMPLE_REPLACE_ANGLICISM_STARTUP", "CA_SIMPLE_REPLACE_ANGLICISM_STARTUPS",
                    "CA_SIMPLE_REPLACE_ANGLICISM_START_UP", "CA_SIMPLE_REPLACE_ANGLICISM_START_UPS"),
    "lex.stakeholder": ("CA_SIMPLE_REPLACE_ANGLICISM_STAKEHOLDERS",),
    "lex.coworking": ("CA_SIMPLE_REPLACE_ANGLICISM_COWORKING",),
    "lex.spam": ("CA_SIMPLE_REPLACE_ANGLICISM_SPAM",),
    "lex.clickbait": ("CA_SIMPLE_REPLACE_ANGLICISM_CLICKBAIT",),
    "lex.target": ("CA_SIMPLE_REPLACE_ANGLICISM_TARGET",),
    "lex.know_how": ("CA_SIMPLE_REPLACE_ANGLICISM_KNOW_HOW",),
    "lex.merchandising": ("CA_SIMPLE_REPLACE_ANGLICISM_MERCHANDISING",),
    "lex.lockdown": ("CA_SIMPLE_REPLACE_ANGLICISM_LOCKDOWN",),
    "lex.link": ("CA_SIMPLE_REPLACE_ANGLICISM_LINK",),
    "lex.password": ("CA_SIMPLE_REPLACE_ANGLICISM_PASSWORD",),
    "lex.deadline": ("CA_SIMPLE_REPLACE_ANGLICISM_DEADLINE",),
    "lex.timeline": ("CA_SIMPLE_REPLACE_ANGLICISM_TIMELINE",),
    "lex.workshop": ("CA_SIMPLE_REPLACE_ANGLICISM_WORKSHOP", "CA_SIMPLE_REPLACE_ANGLICISM_WORKSHOPS"),
    "lex.adapt.scanner": ("CA_SIMPLE_REPLACE_ANGLICISM_SCANNER",),
    "lex.adapt.stress": ("CA_SIMPLE_REPLACE_ANGLICISM_STRESS",),
    "lex.adapt.leader": ("CA_SIMPLE_REPLACE_ANGLICISM_LEADER", "CA_SIMPLE_REPLACE_ANGLICISM_LEADERS"),
    "lex.adapt.podcast": ("CA_SIMPLE_REPLACE_ANGLICISM_PODCAST", "CA_SIMPLE_REPLACE_ANGLICISM_PODCASTS"),
    "sem.sever": ("SEVER",),
    "sem.projectar_preveure": ("PROJECTAR",),
    "sem.replicar_repetir": ("REPLICAR_REPETIR",),
    "sem.basat_a": ("BASAT_EN",),
    "syn.resultar_en": ("RESULTAR_EN",),
    "syn.estar_sent_participi": ("ESTA_SENT_FET", "ESTAR_GERUNDI"),
    "syn.prendre_accio": ("PRENDRE_ACCIÓ",),
    "syn.possessiu_cos": ("POSSESSIUS_PARTS_COS", "POSSESSIU_REDUNDANT"),
}

# ---------------------------------------------------------------------------------------
# Phenomenon info
# ---------------------------------------------------------------------------------------


def _english_marker(forms: list[str]) -> str:
    alternatives = sorted({re.escape(f).replace(r"\ ", r"[\s-]") for f in forms}, key=len, reverse=True)
    return r"\b(" + "|".join(alternatives) + r")\b"


def _kinds(entry: dict[str, Any], spec: _Spec) -> tuple[str, ...]:
    kinds = []
    if entry["assessment"] == "normative_error" and spec.frames:
        kinds.append("pair")
    if entry["assessment"] == "style_preference" and spec.frames:
        kinds.append("preference")
    if entry.get("controls") or spec.controls:
        kinds.append("control")
    return tuple(kinds)


def _info(pid: str) -> dict[str, Any]:
    entry, spec = ENTRIES[pid], SPECS[pid]
    alternatives = " / ".join(f"«{a['form']}»" for a in entry.get("alternatives", []))
    if entry["category"] == "lexical":
        adapted = pid.startswith("lex.adapt.")
        if adapted:
            default_error = f"Grafia no adaptada «{entry['english'][0]}»."
        elif entry["assessment"] == "style_preference":
            default_error = f"Anglicisme «{entry['english'][0]}», admès en registres informals i semiformals."
        else:
            default_error = f"Anglicisme «{entry['english'][0]}»."
        error = spec.error or default_error
        marker = spec.marker or _english_marker(entry["english"])
    else:
        error, marker = spec.error, spec.marker
    info = {
        "error": error,
        "correction": spec.correction or alternatives + ".",
        "languagetool_rules": list(spec.languagetool_rules or _LANGUAGETOOL_RULES.get(pid, ())),
        "languagetool_gaps": [],
        "category": entry["category"],
        "assessment": entry["assessment"],
        "evidence_level": entry["evidence_level"],
        "kinds": list(_kinds(entry, spec)),
        "scope": entry.get("scope", ""),
    }
    if pid in _CORRECT_USES:
        info["correct_uses"] = _CORRECT_USES[pid]
    if spec.languagetool_tolerated:
        info["languagetool_tolerated"] = list(spec.languagetool_tolerated)
    if marker:
        info["marker"] = marker
    return info


PHENOMENA: tuple[str, ...] = tuple(ENTRIES)
PHENOMENON_INFO: dict[str, dict[str, Any]] = {pid: _info(pid) for pid in PHENOMENA}

# ---------------------------------------------------------------------------------------
# Realisation
# ---------------------------------------------------------------------------------------

_SLOT = re.compile(r"\{([a-z_]+?)(\d?)\}")
_SPAN = re.compile(r"\[([^\]]*)\]")


def _fill(text: str, rng: Random, slots: dict[str, tuple[str, ...]]) -> str:
    """Replace every ``{slot}``; repeated slots get the same value, ``{p2}`` a different one."""
    chosen: dict[str, str] = {}
    used: dict[str, set[str]] = {}

    def value(match: re.Match[str]) -> str:
        key = match.group(0)
        if key not in chosen:
            name = match.group(1)
            options = [v for v in slots[name] if v not in used.get(name, set())]
            chosen[key] = rng.choice(options)
            used.setdefault(name, set()).add(chosen[key])
        return chosen[key]

    return _SLOT.sub(value, text)


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text[:1].upper() + text[1:]


def _slots(spec: _Spec) -> dict[str, tuple[str, ...]]:
    return {**_SLOTS, **(spec.slots or {})}


@dataclass(frozen=True)
class _Realisation:
    good: str
    bad: Optional[str]
    variants: tuple[str, ...]
    frame: int
    context: Optional[str] = None


def _realise_pair(rng: Random, spec: _Spec) -> _Realisation:
    index = rng.randrange(len(spec.frames))
    frame = spec.frames[index]
    frame, context = (frame, None) if isinstance(frame, str) else frame
    filled = _fill(frame, rng, _slots(spec))
    match = _SPAN.search(filled)
    goods, bads = (side.split("/") for side in match.group(1).split("|"))
    bad = rng.choice(bads)

    def sentence(span: str) -> str:
        return _tidy(filled[: match.start()] + span + filled[match.end():])

    variants = tuple(dict.fromkeys(sentence(g) for g in goods))
    return _Realisation(variants[0], sentence(bad), variants, index, context)


def _control_frames(pid: str) -> tuple[str, ...]:
    return tuple(ENTRIES[pid].get("controls", ())) + SPECS[pid].controls


def _realise_control(rng: Random, pid: str) -> _Realisation:
    frames = _control_frames(pid)
    index = rng.randrange(len(frames))
    good = _tidy(_fill(frames[index], rng, _slots(SPECS[pid])))
    return _Realisation(good, None, (good,), index)


# ---------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------


@dataclass
class AnglicismesConfig:
    phenomena: tuple[str, ...] = ()  # empty: every accepted catalog entry in ``categories``
    categories: tuple[str, ...] = CATEGORIES
    kinds: tuple[str, ...] = KINDS
    formats: tuple[str, ...] = FORMATS
    # Share of each kind among the items (pairs get the rest), among the kinds available.
    preference_share: float = 0.08
    control_share: float = 0.12
    norm: str = "iec"
    seed: Optional[int] = None
    size: int = 500

    def selected(self) -> tuple[str, ...]:
        chosen = self.phenomena or PHENOMENA
        return tuple(p for p in chosen if PHENOMENON_INFO[p]["category"] in self.categories)

    def kind_weights(self) -> dict[str, float]:
        """Kinds this config can produce, with their weights."""
        shares = {"pair": 1.0 - self.preference_share - self.control_share,
                  "preference": self.preference_share, "control": self.control_share}
        weights = {}
        for kind in self.kinds:
            has_format = set(self.formats) & set(KIND_FORMATS[kind])
            has_phenomenon = any(kind in PHENOMENON_INFO[p]["kinds"] for p in self.selected())
            if has_format and has_phenomenon and shares[kind] > 0:
                weights[kind] = shares[kind]
        return weights

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        for p in self.phenomena:
            assert p in PHENOMENON_INFO, f"unknown phenomenon '{p}'; expected one of {PHENOMENA}"
        for c in self.categories:
            assert c in CATEGORIES, f"unknown category '{c}'; expected one of {CATEGORIES}"
        for k in self.kinds:
            assert k in KINDS, f"unknown kind '{k}'; expected one of {KINDS}"
        for f in self.formats:
            assert f in FORMATS, f"unknown format '{f}'; expected one of {FORMATS}"
        for name in ("preference_share", "control_share"):
            assert 0.0 <= getattr(self, name) <= 1.0, f"{name} must be in [0, 1]"
        assert self.preference_share + self.control_share <= 1.0, "preference_share + control_share must be at most 1"
        assert self.selected(), "no phenomenon left after filtering by categories"
        assert self.kind_weights(), "no kind can be produced with these phenomena, kinds and formats"
        if self.norm != "iec":
            raise NotImplementedError(f"norm '{self.norm}' is not implemented yet (only 'iec')")


class AnglicismesDataset(ProceduralDataset):
    def __init__(self, config: AnglicismesConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self._weights = config.kind_weights()
        selected = config.selected()
        self._by_kind = {k: [p for p in selected if k in PHENOMENON_INFO[p]["kinds"]] for k in self._weights}
        # Every gec item of a task with controls admits an unchanged answer, so the wording
        # doesn't give away which sentences are already correct.
        self._allow_unchanged = "control" in self._weights

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = Random(self.seed + idx)
        cfg = self.config

        kinds = list(self._weights)
        kind = rng.choices(kinds, weights=[self._weights[k] for k in kinds])[0]
        phenomenon = rng.choice(self._by_kind[kind])
        spec, info = SPECS[phenomenon], PHENOMENON_INFO[phenomenon]
        realisation = _realise_control(rng, phenomenon) if kind == "control" else _realise_pair(rng, spec)
        fmt = rng.choice([f for f in cfg.formats if f in KIND_FORMATS[kind]])
        question, answer, extra = render(
            fmt, realisation.good, realisation.bad, rng,
            kind=kind, context=realisation.context, allow_unchanged=self._allow_unchanged,
        )

        meta: dict[str, Any] = {
            "source_dataset": DATASET_NAME,
            "source_index": idx,
            "format": fmt,
            "phenomenon": phenomenon,
            "kind": kind,
            "category": info["category"],
            "assessment": info["assessment"],
            "evidence_level": info["evidence_level"],
            "norm": cfg.norm,
            "sentence_good": realisation.good,
            "sentence_bad": realisation.bad,
            "correct_variants": list(realisation.variants),
            "languagetool_rules": info["languagetool_rules"],
            "frame": realisation.frame,
            **extra,
        }
        if realisation.context:
            meta["context"] = realisation.context
        if spec.languagetool_tolerated:
            meta["languagetool_tolerated"] = list(spec.languagetool_tolerated)
        return {"question": question, "answer": answer, "metadata": meta}

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        return score(answer, entry)


class AnglicismesCurriculum(BaseCurriculum):
    def __init__(self) -> None:
        super().__init__(
            DATASET_NAME,
            AnglicismesConfig,
            [
                ScalarAttributeDefinition(
                    name="categories",
                    field_name="categories",
                    levels=(("lexical",), ("lexical", "semantic"), CATEGORIES),
                    description="Kinds of interference: English words, English senses, English constructions",
                ),
            ],
        )


register_dataset(DATASET_NAME, AnglicismesDataset, AnglicismesConfig, AnglicismesCurriculum)
