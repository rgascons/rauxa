# LLM weaknesses in Catalan and seed resources for a Catalan synthetic data generator (as of Oct 2026)

Research budget: about 20 tool calls. Several primary documents could not be fetched (UAB DDD PDF blocked by bot protection; the Linguamática paper itself was not retrieved). Claims marked "background knowledge" in Inferences are from model knowledge, not sources fetched in this session.

## Q1. What studies document LLM errors in Catalan (Spanish interference, normative errors, grammar phenomena)?

### Takeaway
There is little peer-reviewed, phenomenon-level work on LLM errors in Catalan. The strongest evidence is a UPF study (Brochhagen, Almena; *Linguamática*, reported Feb 2026). In it, multilingual LLMs picked a Spanish-influenced non-normative preposition in 55% of relevant cases, against 27% for Catalan-specialised models. The IEC (Oct 2025) also documented ChatGPT inventing normative claims and DIEC citations. Most other phenomena on the user's list (pronoms febles, participle agreement, diacritics, apostrophes) have no published LLM-specific measurement I could find. They are still well codified in LanguageTool's rule categories.

### Cited Findings
- **UPF study (Brochhagen, Almena), published in *Linguamática*.** It evaluated six LLMs (four multilingual, two Catalan-specialised "monolingual" ones) on 160 sentences covering 8 preposition-doubt structures, and measured whether each model preferred the normative or non-normative variant. Results: multilingual models chose the Spanish-influenced non-normative preposition in 55% of relevant cases and erred for other reasons in about 4%. Monolingual models showed 27% Spanish-influenced errors, with about 4% from other causes. VilaWeb describes it as the first study to show a bias toward Spanish, after earlier work showed a bias toward English structures. — [VilaWeb, 26 Feb 2026](https://www.vilaweb.cat/noticies/upf-ia-formes-no-normatives-catala/)
- Examples from the same study: the prepositional direct object calqued from Spanish (*he visto **al** profesor* → non-normative *he vist **al** professor* vs normative *he vist **el** professor*), and a non-Spanish error, *propens **d'**enfadar-me* instead of *propens **a** enfadar-me*. The article names ChatGPT and Gemini among the models but does not list the 8 structures. — [Ara (EN), 26 Feb 2026](https://en.ara.cat/languages/have-seen-the-professor-ai-spreads-mistakes-in-catalan-due-to-spanish-influence_1_5659940.html)
- **IEC / Ara test of ChatGPT on normative questions (1 Oct 2025).** ChatGPT gave non-normative terms and falsely claimed they were in the DIEC: it said "quiròfan"-type calques (rendered in the English edition as "operating room", described as "a translation from Castilian") were normative and in the DIEC. It also attributed to the IEC dictionary a definition for "passepartout" that does not appear there (Termcat/GDLC give another term), and cited the DIEC for definitions not in it. Nicolau Dols, president of the IEC Philological Section, said the main problem is that the AI "sometimes spreads clearly false information" and does not distinguish normative from non-normative Catalan or registers. — [Ara (EN), 1 Oct 2025](https://en.ara.cat/languages/do-you-have-questions-about-standard-catalan-don-t-ask-them-in-the-gpt-chat_1_5513337.html)
- A master's thesis on using ChatGPT to generate Catalan teaching materials found that the tool "tendeix a respondre en castellà" (tends to answer in Spanish) even when instructions are in Catalan. This is language-switching drift. — [UVic repository](https://repositori.uvic-ucc.cat/handle/10854/8446) (via search snippet; not opened)
- Meta AI refused or failed to answer in Catalan for months in 2025 (messages replaced with "doesn't understand Catalan yet"). By Dec 2025 it conversed in Catalan, but VilaWeb noted it "a vegades encara hi ha alguns problemes ortogràfics" (still has some spelling problems at times). — [VilaWeb](https://www.vilaweb.cat/noticies/meta-ia-parla-finalment-catala-mesos-despres-problemes/)
- **Wug-test study** (Pantelidou, Leivada, Montero, Morosi; arXiv Oct 2025, v3 Apr 2026). It compared six LLMs with humans on novel-word morphology in Catalan, English, Greek and Spanish. Catalan and Greek accuracy was lower than Spanish and English, which the authors attribute to community size and data availability rather than grammatical complexity. — [arXiv 2510.12463](https://arxiv.org/abs/2510.12463)
- In compact-LLM zero-shot evaluation on IberoBench, Catalan generally scored between Spanish (best) and Galician. — [arXiv 2504.03312](https://arxiv.org/html/2504.03312v1)
- **LanguageTool categories as an error taxonomy.** The Catalan rule browser shows categories including "Pronoms febles", "Apostrofació", "Concordances entre subjecte, verb i complements", "Concordances en grups nominals" and, notably, "Errors de traductor automàtic" (machine-translation errors). — [LanguageTool community rule list (ca)](https://community.languagetool.org/rule/list?lang=ca)
- The LanguageTool Catalan rule directory contains dedicated files that map directly to the user's target phenomena:
  - `replace_diacritics_iec.txt` (2016 diacritic reform)
  - `replace_anglicism.txt`
  - `replace_verbs.txt`
  - `confusion_pairs.txt`
  - `wrongWordInContext.txt`
  - `replace_adverbs_ment.txt`
  - `verb_senseaccent_*` (accent homographs)
  - `replace_balearic.txt`
  - `replace_dnv*.txt`
  - `coherency-valencia.txt`
  - a `ca-ES-valencia/` subfolder

  — [GitHub languagetool ca rules](https://github.com/languagetool-org/languagetool/tree/master/languagetool-language-modules/ca/src/main/resources/org/languagetool/rules/ca)

### Inferences
- The UPF figures give a quantified, publishable anchor: Spanish interference in **prepositions** is a high-yield target, with roughly half of relevant cases wrong for multilingual models. A synthetic minimal-pair generator for preposition structures (DO without *a*, *propens a*, *a/en* locatives, *per/per a*, preposition dropping before *que*) is directly motivated by evidence.
- **Hallucinated normativity** is a distinct error type worth benchmarking separately from production errors: asking "is X normative / in the DIEC?" and checking answers against the DIEC/DNV/Optimot.
- Background knowledge, not verified this session: the 2016 IEC orthographic reform cut diacritic accents to about 15 pairs (e.g. *són/son*, *més/mes*, *déu/deu*), so forms like *dóna*, *bé/be* changed status. Models trained on pre-2017 web text likely mix old and new diacritic norms. LanguageTool's `replace_diacritics_iec.txt` presumably encodes these.
- Background knowledge: classic castellanismes such as *tenir que* (→ *haver de*), *donar-se compte* (→ *adonar-se*), *en quant a* (→ *pel que fa a*), *degut a* (→ *a causa de*) and *hasta* are standard targets in LanguageTool grammar.xml and in Optimot/Ésadir. I found **no published study measuring LLM rates** for these, nor for pronom feble combinations (*l'hi*, *n'hi*, *els hi*), participle agreement with clitics, the periphrastic past, ser/estar or apostrophe rules. These are gaps the user's benchmark could fill.
- The fact that LanguageTool has an "Errors de traductor automàtic" category suggests machine-translation artefacts (which LLM output resembles) were common enough to warrant dedicated rules.

### Gaps
- Could not retrieve the *Linguamática* paper itself: the exact 8 structures, model names and per-model breakdown are unknown. Search Linguamática 2025–26 issues for Almena/Brochhagen.
- No study found measuring LLM accuracy on pronoms febles, participle agreement, diacritics, apostrophation, or ser/estar.
- The UAB DDD paper "rld84_borrell_ca.pdf" (Revista de Llengua i Dret 84, 2025, author Borrell) appeared in search results related to LanguageTool/Softcatalà but was blocked by bot protection. Its content is unknown and it may be relevant.
- No Softcatalà or Plataforma per la Llengua report with LLM error analysis was found.

## Q2. How do LLMs handle dialectal variation (Valencian/AVL, Central, Balearic)?

### Takeaway
I found no published evaluation testing whether LLMs default to Central/IEC Catalan when asked for Valencian (AVL) or Balearic, or slip into Spanish. This is an open gap. Resources to build such a test exist: LanguageTool's Valencian and Balearic rule files, Softcatalà's Valencian Hunspell build, and the DNV.

### Cited Findings
- No source found directly testing LLM AVL-vs-IEC behaviour. The closest work, a 2025 UPV paper on Valencian political discourse (#DANAValencia), found LLMs default to anglophone normative frames when analysing evaluative language. — [Polytechnic/SPbSTU journal PDF](https://human.spbstu.ru/userfiles/files/articles/2025/4/159-183.pdf)
- The AVL is the official regulator for Valencian. Its norms derive from the Normes de Castelló and do not differ greatly from the IEC, but its grammar includes Valencian-specific syntax features. — [Wikipedia: AVL](https://en.wikipedia.org/wiki/Acad%C3%A8mia_Valenciana_de_la_Llengua); [UV news on DNV/GNV](https://www.uv.es/uvweb/college/en/news-release/avl-valencian-language-academy-explains-content-dnv-valencian-normative-dictionary-gramatica-normative-valenciana-valencian-normative-grammar-gandia-s-summer-university-1285846070123/Noticia.html?id=1285915349820)
- The DNV has more than 93,000 entries, has been online since Feb 2014, and was printed in Sept 2016. — (search-result summary citing AVL/El Temps sources; [AVL](https://www.avl.gva.es/welcome-en/))
- LanguageTool has a `ca-ES-valencia` variant with `coherency-valencia.txt`, `replace_dnv.txt`, `replace_dnv_colloquial.txt`, `replace_dnv_secondary.txt` and `replace_balearic.txt`. — [GitHub](https://github.com/languagetool-org/languagetool/tree/master/languagetool-language-modules/ca/src/main/resources/org/languagetool/rules/ca)
- Softcatalà's catalan-dict-tools builds both general Catalan and Valencian Hunspell dictionaries. — [GitHub Softcatala/catalan-dict-tools](https://github.com/Softcatala/catalan-dict-tools)
- Valencian-oriented models exist, e.g. GPLSI's Aitana-2B-S-Instruct (Univ. Alacant), whose model card reports LLM-as-judge comparisons with Salamandra-2B. — [Aitana model page](https://friendli.ai/models/gplsi/Aitana-2B-S-Instruct)

### Inferences
- A dialect-consistency test is feasible and novel. Prompt the model "in Valencian (AVL)" and check output for Central forms (e.g. 1st-person present *-o* vs Valencian *-e/-ø*, *aquest* vs *este*, *noi* vs *xiquet*, subjunctive *-i* vs *-e/-a*), using `coherency-valencia.txt` / `replace_dnv*` as the lexicon of variant pairs. Note these are background-knowledge examples; verify them against the DNV.
- The coherency files (mixing variants within a text is flagged) are a ready-made spec for an "intra-text dialect consistency" metric.
- The normative reference is politically contested (AVL vs IEC), so benchmark items should state which norm they follow.

### Gaps
- No evidence found on Balearic handling or on how often LLMs code-switch to Spanish in Valencian prompts.

## Q3. Cultural/local knowledge gaps

### Takeaway
Evidence is mostly advocacy and journalistic rather than measured. Accent Obert (Feb 2026) argued that chatbots "think" in English and translate, and announced plans to give 12 commercial chatbots official Catalan exams (school, PISA, university entrance, C-level certificate, driving theory) with a public ranking. I found no results from those exams yet.

### Cited Findings
- Accent Obert (successor to Fundació .cat), via VilaWeb on 19 Feb 2026: 12 main chatbots (ChatGPT, Gemini, Claude, Grok, Qwen, etc.) converse in Catalan, but the response process often relies on English, introducing cultural bias. Planned tests: primary/ESO exams, PISA, PAU, the C-level Catalan certificate and driving-licence theory, repeated after model updates, with a public ranking. Risks named: friction-driven language abandonment, dependence on foreign models, and cultural framing. — [VilaWeb](https://www.vilaweb.cat/noticies/ia-pensa-catala-accent-obert/); [Accent Obert supplementary note PDF](https://accentobert.cat/wp-content/uploads/2026/02/Informacio-complementaria-a-la-Nota-de-Premsa-dAccent-Obert.pdf)
- The IEC test above also shows a knowledge gap on Catalan normative and terminology sources: invented DIEC citations, and ignoring Termcat/GDLC terms. — [Ara](https://en.ara.cat/languages/do-you-have-questions-about-standard-catalan-don-t-ask-them-in-the-gpt-chat_1_5513337.html)
- A ScienceDirect paper evaluates LLM mathematical reasoning in Catalan (2025). — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2590123025004475) (not opened)

### Inferences
- Accent Obert's exam ranking, if published after Oct 2026, would be a key comparison point. The user should check for it.
- Template-based generation for cultural knowledge (comarques, capitals, institutions such as the Generalitat, Consell, Síndic de Greuges, Parlament, Corts Valencianes; literature) could use Wikidata/Viquipèdia, but no source documents specific failure rates.

### Gaps
- No measured evaluation of LLM knowledge of Catalan geography, law, literature or history was found.

## Q4. Evaluations of GPT/Claude/Gemini/Llama/Salamandra/Mistral writing quality in Catalan (incl. LLM-as-judge, human)

### Takeaway
Evaluation infrastructure exists: CatalanBench/IberoBench, La Leaderboard, CatCoLA, and Salamandra's Prometheus-2 judge setup. These test mostly understanding, not normative writing quality. No head-to-head human evaluation of frontier models' Catalan *writing correctness* was found. The UPF preposition study is the closest.

### Cited Findings
- The Salamandra Technical Report used Prometheus-2 8x7B as an LLM judge for open-ended tasks in es/ca/eu/gl/en. Its Catalan tasks include catcola, teca and parafraseja. — [arXiv 2502.08489](https://arxiv.org/pdf/2502.08489)
- A cross-lingual commonsense study using Prometheus-2 and JudgeLM judges found LLaMA/Qwen/Gemma skewed toward English, while EuroLLM and Salamandra were more even across languages. — [arXiv 2509.06401](https://arxiv.org/pdf/2509.06401)
- IberoBench covers Basque, Catalan, Galician, Spanish and Portuguese on the LM Evaluation Harness. — [ACL Anthology COLING 2025](https://preview.aclanthology.org/credits/2025.coling-main.699)
- IberBench integrates 101 datasets, including Catalan. — [arXiv 2504.16921](https://web3.arxiv.org/pdf/2504.16921)
- La Leaderboard covers Catalan, Basque, Galician and Spanish varieties with 66 datasets and 50 models (search summary). — [Salamandra/related search result](https://arxiv.org/pdf/2502.08489)
- **CatCoLA** (Bel, Punsola et al.; *Procesamiento del Lenguaje Natural* 73, 2024): 10,443 sentences with acceptability judgments from Catalan reference grammars, each tagged with the linguistic phenomenon. The test split is withheld. The licence is reported inconsistently: CC BY-NC-SA 4.0 in one record, CC BY-SA 4.0 in CORA, CC BY 4.0 on GitHub. — [B2FIND record](https://b2find.eudat.eu/dataset/4f103bd1-2905-55eb-9f2d-998cdc30bf7c)
- Caveat from Basque work: state-of-the-art judge models correlate poorly with humans in lower-resource languages (search summary). — [arXiv 2509.06401 context](https://arxiv.org/pdf/2509.06401)

### Inferences
- CatCoLA's phenomenon tags are a strong seed for grammar minimal pairs, but its licence ambiguity, possibly NC, needs resolving before derivatives are published.
- Using an LLM judge for Catalan normativity is risky given the IEC evidence that models hallucinate normativity. Rule-based checking (LanguageTool) plus human spot-checks is a safer judge for a correctness benchmark.

### Gaps
- No public human evaluation comparing Claude/GPT/Gemini Catalan prose quality (castellanismes per 1k words, etc.) was found.
- No Softcatalà, Plataforma per la Llengua or Optimot evaluation of chatbots was found.

## Q5. Seed resources: formats, licences, and suitability for publishing derived datasets

### Takeaway
The most useful openly licensed seeds are:
- **LanguageTool Catalan rules** (LGPL 2.1+; XML plus replace/confusion TXT files covering castellanismes, diacritics, pronoms febles, Valencian/Balearic variants).
- **Softcatalà catalan-dict-tools** (LGPL 2.1/GPL 2; a full inflected lexicon of about 10M forms, general plus Valencian).
- **UD Catalan-AnCora** (CC BY 4.0 per UD, though it notes an inherited GNU licence from AnCora).
- **Projecte Aina parallel corpora** (licences vary by pair).
- **CaWaC / Catalan Textual Corpus** (CC BY-SA).

Normative dictionaries (DIEC2, DNV) and IEC sites are copyrighted, free to consult, and have no found open licence or API. Use them for reference and validation, not bulk extraction.

### Cited Findings
- **LanguageTool**: LGPL v2.1+ overall. — [Wikipedia](https://en.wikipedia.org/wiki/LanguageTool)
  - Catalan rules are XML (`grammar.xml`) plus many TXT replacement/confusion lists. — [GitHub dir](https://github.com/languagetool-org/languagetool/tree/master/languagetool-language-modules/ca/src/main/resources/org/languagetool/rules/ca)
  - The community rule browser shows 9,488 "matches" for Catalan; this is not equal to distinct rules. — [LT community](https://community.languagetool.org/rule/list?lang=ca)
  - An older Softcatalà announcement cited more than 1,700 rules. — [UB Retrucs](https://www.ub.edu/retrucs/?p=5140)
  - Maintained by Jaume Ortolà (Softcatalà). — [Meta-Wiki IEG](https://meta.wikimedia.org/wiki/Grants:IEG/Proofreading_semiautomatically_the_Catalan_Wikipedia_with_LanguageTool)
- **Softcatalà catalan-dict-tools**: root dictionary (`diccionari-arrel`) by POS. `-fdic.txt` files hold inflection data (verb model, feminine, irregular plurals); `-lt.txt` uses LanguageTool/FreeLing format. Outputs: Hunspell (general plus Valencian), a LanguageTool tagged dictionary, Morfologik, and a full wordlist of about 10M forms (about 166 MB) including apostrophes and attached pronouns. Dual-licensed LGPL 2.1 / GPL 2. — [GitHub](https://github.com/Softcatala/catalan-dict-tools)
- **Softcatalà English–Catalan dictionary**: data under CC BY 4.0 (server code GPL v2). A `softcatala/catalan-dictionary` dataset exists on Hugging Face; its licence was not confirmed. Softcatalà also publishes translation memories from 180+ open-source projects; the licence field was blank in the snippet. — [UB Multicercadors](https://www.ub.edu/multicercadors/diccionari-angles-catala-softcatala/); [HF](https://huggingface.co/datasets/softcatala/catalan-dictionary/discussions/2); [Softcatalà](https://www.softcatala.org/?p=16612)
- **UD Catalan-AnCora**: 16,678 sentences, 531,390 tokens, news genre, licence CC BY 4.0. The page also notes a GNU licence inherited from AnCora and says to contact UB with questions. — [UD](https://universaldependencies.org/treebanks/ca_ancora/index.html)
- **CaWaC**: 780M-token web corpus from the .cat domain (2013), CC BY-SA 3.0. — [HF card](https://huggingface.co/datasets/community-datasets/cawac/blob/d80d4a7a20a333e7d1d70aa61b27a84c790bb658/README.md)
- **Catalan Textual Corpus** (BSC, Zenodo 2021): about 1.76B tokens including CaWaC; packaging is CC BY-SA 4.0, and the underlying sources may carry their own terms. — [Zenodo 4519349](https://zenodo.org/record/4519349)
- **Projecte Aina parallel corpora**: CA-EN has 14,967,979 sentence pairs (CC BY 4.0); CA-DE is CC BY-NC-SA 4.0; CA-FR was changed to CC BY-SA 4.0. — [HF CA-EN](https://huggingface.co/datasets/projecte-aina/CA-EN_Parallel_Corpus/blob/4d1041c3cb3bd1916d2821e9d8e710dd2278d478/README.md); [HF CA-FR](https://huggingface.co/datasets/projecte-aina/CA-FR_Parallel_Corpus/blob/813b54a1d5f5a7170f2d331acabeab7102a4cbb1/README.md); [HF CA-DE](https://huggingface.co/datasets/projecte-aina/CA-DE_Parallel_Corpus/commits/8161091e14318435e3aa6986e537c018c3499d06)
- **CatCoLA**: see Q4 (licence inconsistent, possibly NC). — [B2FIND](https://b2find.eudat.eu/dataset/4f103bd1-2905-55eb-9f2d-998cdc30bf7c)
- **DIEC2 / IEC**: IEC pages (e.g. DECAT) carry "© 2024 FPC IEC". The IEC states its online resources are free to consult, and DIEC2 is updated digitally twice a year. No open licence or API was found. — [DECAT example](https://decat.iec.cat/veuredoc.asp?id=22284); [Ara](https://en.ara.cat/languages/do-you-have-questions-about-standard-catalan-don-t-ask-them-in-the-gpt-chat_1_5513337.html)
- **DNV (AVL)**: more than 93k entries, online since 2014. No licence or API information was found. — [AVL](https://www.avl.gva.es/welcome-en/)
- **MULTITuDE** (machine-generated text detection) includes Catalan among 11 languages: about 74k machine and 7k human texts. — [Computational Linguistics survey](https://direct.mit.edu/coli/article/51/1/275/127462/A-Survey-on-LLM-Generated-Text-Detection-Necessity)

### Inferences
- **Best seed pipeline (inference):**
  1. Use the LanguageTool `grammar.xml` patterns and `replace*.txt` lists as an inventory of (incorrect → correct) pairs and patterns, to inject errors into clean sentences and to auto-check model output.
  2. Use the Softcatalà lexicon for morphologically correct inflection in templates (gender/number, verb conjugation, clitic attachment).
  3. Use AnCora-UD for syntactic templates and real sentence frames.
  4. Use ca-es parallel data for calque mining.
- **Licensing for a published derived dataset (inference, not legal advice):**
  - CC BY 4.0 sources (AnCora-UD per UD, Aina CA-EN, Softcatalà en-ca dictionary data) are the most permissive.
  - CC BY-SA sources (CaWaC, Catalan Textual Corpus, Aina CA-FR, Wikipedia/Viquipèdia) require share-alike on the derived data.
  - LGPL/GPL lexicons and rule files: whether generated sentences are a "derivative work" of LGPL data is unclear. The safest course is to release the derived dataset under a compatible licence and attribute.
  - Avoid NC sources (CA-DE, possibly CatCoLA) if commercial use is intended.
  - DIEC2/DNV/Optimot/GIEC/Ésadir: use for manual validation only, unless permission is obtained.
- Background knowledge, not verified this session:
  - Wiktionary is CC BY-SA.
  - Common Voice text is CC0.
  - Viquipèdia is CC BY-SA 4.0.
  - Optimot, Ésadir (CCMA style guide) and Termcat are free to consult but copyright-reserved.
  - FreeLing data is under a GPL/AGPL-family licence. Verify the exact licence.

### Gaps
- No licence or API terms were verified for DIEC2, DNV, Optimot, GIEC, Ésadir, Termcat or Plataforma per la Llengua lists. A direct check of legal notices is needed.
- No dedicated, Aina-labelled ca-es parallel corpus licence was verified this session. Aina likely has a CA-ES corpus; check `huggingface.co/projecte-aina`.
- No public Catalan grammatical-error-correction corpus (Softcatalà or other) was found. The GitHub Typo Corpus may or may not include Catalan.
- The exact count of distinct Catalan LanguageTool rules was not verified (the old figure is 1,700+ rules; the current browser shows 9,488 matches).
