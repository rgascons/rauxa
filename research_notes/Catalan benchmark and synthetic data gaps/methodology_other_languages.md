# Methodologies of comparable mid-resource-language synthetic/generated benchmarks (templates for a Catalan generator + benchmark), as of Oct 2026

Note on verification: items marked [VERIFIED] were confirmed in sources fetched/searched in this session (Oct 2026). Items marked [UNVERIFIED-PRIOR] come from background knowledge and were not re-confirmed here; the report writer should treat them as lower confidence or omit.

## 1. Minimal-pair grammar benchmarks: how built, validated, scaled, evaluated

### Takeaway
Three construction families exist: (a) linguist-written templates + feature-annotated vocabulary (BLiMP, ZhoBLiMP), (b) automatic perturbation of real treebank/parsed sentences with rules (MultiBLiMP, RuBLiMP), and (c) manual extraction/editing from authoritative grammar resources (QFrBLiMP, EsCoLA, CatCoLA, Icelandic NoDaLiDa 2025). All evaluate by log-probability comparison of the pair. Catalan IS in MultiBLiMP 1.0 (2,284 pairs, agreement only), so a Catalan BLiMP must go beyond subject-verb agreement to add value.

### Cited Findings
**BLiMP (English, TACL 2020) — the template archetype**
- 67 datasets x 1,000 pairs, 12 phenomenon categories, covering syntax, morphology, semantics; automatically generated from linguist-crafted grammar templates; aggregate human agreement with labels 96.4% (crowdworker validation); scoring = model assigns higher probability to acceptable sentence; models good on agreement morphology, weak on NPIs and extraction islands [VERIFIED] — [ACL Anthology 2020.tacl-1.25](https://aclanthology.org/2020.tacl-1.25); [arXiv 1912.00582](https://export.arxiv.org/abs/1912.00582v1)

**MultiBLiMP 1.0 (Jumelet, Weissweiler, Nivre, Bisazza; arXiv Apr 2025; TACL 2026) — automated, UD-derived**
- 101 languages, 128,321 pairs; phenomena restricted to subject–verb and subject–participle agreement in number, person, gender (v1 arXiv described "6 phenomena", >125k pairs; scope described differently across versions) [VERIFIED] — [arXiv 2504.02768v2](https://arxiv.org/html/2504.02768v2); [TACL 2026.tacl-1.10](https://aclanthology.org/2026.tacl-1.10)
- Pipeline: UD v2.15 + UniMorph v4.0; take nsubj edges between noun and finite verb/participle, re-inflect verb to contrasting feature value via UniMorph to create ungrammatical member; "fully automated" [VERIFIED] — [arXiv 2504.02768v2](https://arxiv.org/html/2504.02768v2)
- Filters: drop sentences with reparandum, Style/Foreign/Typo features, conjoined subjects, verbs with expl/csubj:outer/nsubj:outer; binomial test (p0=0.9, Bonferroni) to confirm a language actually shows agreement in a condition; cap 100 items per condition with weighted downsampling over subject form, verb form, distance, attractors (1.4M candidate pairs before balancing) [VERIFIED] — same source
- Iberian coverage (Table 5 pair counts): Catalan 2,284; Spanish 2,541; Portuguese 3,048; Galician 753; Basque 273 [VERIFIED; column label "nn" interpreted as pair count] — same source
- Human validation: only "extensive spot checks" and feedback from language experts; full manual validation called cost-prohibitive [VERIFIED] — same source
- Results: Llama3-70B and Gemma3-27B base ~90.2% average; 125M monolingual Goldfish models beat Llama3-70B on Catalan (97.7 vs 96.1), Galician (98.3 vs 92.3), Basque (98.9 vs 95.2) but not Spanish (96.1 vs 98.2) or Portuguese (94.4 vs 98.0); size helps, post-training (chat) hurts; subword over-segmentation of the correct form hurts accuracy [VERIFIED] — same source
- Code: github.com/jumelet/multiblimp; data: huggingface.co/datasets/jumelet/multiblimp; pipeline described as reusable for other phenomena [VERIFIED] — [search summary of arXiv 2504.02768](https://arxiv.org/pdf/2504.02768)

**RuBLiMP (Russian, EMNLP 2024) — parsed-corpus perturbation + decontamination**
- 45k pairs isolating morphological, syntactic, semantic phenomena; 25 LMs evaluated [VERIFIED] — [ACL 2024.emnlp-main.522](https://aclanthology.org/2024.emnlp-main.522/)
- Sentences from open corpora (Wikipedia, news, books), parsed into UD; expert rules over dependency trees apply addition/replacement/swap/movement edits (keeps length matched); pymorphy2 for morphological errors; rules written by three native-speaker authors and peer-reviewed [VERIFIED via search summary of paper] — [arXiv 2406.19232](https://arxiv.org/pdf/2406.19232)
- Decontamination: reportedly MIN-K% PROB computed over a set of LMs; grammatical sentences above threshold (likely seen in pretraining) excluded [PARTLY VERIFIED: mechanism described only by a third-party review] — [Liner review](https://liner.com/review/rublimp-russian-benchmark-linguistic-minimal-pairs); [arXiv 2406.19232](https://arxiv.org/pdf/2406.19232)
- Findings: models sensitive to morphology/agreement, lag humans on structural relations, negation, transitivity, tense [VERIFIED] — [ACL 2024.emnlp-main.522](https://aclanthology.org/2024.emnlp-main.522/)

**ZhoBLiMP (Chinese; arXiv Nov 2024; TACL 2026) — templates at scale with a no-code GUI**
- 118 paradigms, 15 phenomena, 35k pairs; Chinese linguists wrote grammar templates from theoretical literature plus a feature-annotated vocabulary (BLiMP method); built a GUI so non-coding linguists can author templates; human validation step (§3.3, details not retrieved) [VERIFIED] — [arXiv 2411.06096](https://arxiv.org/pdf/2411.06096); [TACL 2026.tacl-1.34](https://aclanthology.org/2026.tacl-1.34.pdf)
- 95 paradigms easy/medium; 13 hard even for models up to 32B; TACL version introduces SLLN-LP (sub-linear length-normalized log-prob) to reduce length bias between pair members [VERIFIED] — same sources

**QFrBLiMP (Quebec French; arXiv Sep 2025; Findings EACL 2026) — manual pairs from prescriptive resource + dense human judgments**
- 1,761 pairs, 20 phenomena, created by manually modifying sentences from the Banque de dépannage linguistique (Office québécois de la langue française); each pair judged by 12 native Quebec-French speakers choosing the grammatical sentence [VERIFIED] — [arXiv 2509.25664](https://arxiv.org/abs/2509.25664v2); [EACL Findings 2026.findings-eacl.224](https://preview.aclanthology.org/credits/2026.findings-eacl.224)
- Findings: competence scales with size; clear difficulty hierarchy; all models fail on phenomena needing deep semantics; most models significantly worse on QFrBLiMP than on MultiBLiMP French [VERIFIED] — same
- Inter-annotator agreement number: not retrieved (aggregator claims "high" without numbers) — [Emergent Mind](https://www.emergentmind.com/papers/2509.25664)
- Repo: github.com/davebulaval/QFrBLiMP [VERIFIED from PDF link annotations] — [arXiv PDF](https://arxiv.org/pdf/2509.25664v2)

**Icelandic Linguistic Benchmark (NoDaLiDa 2025; Ármannsson, Ingimundarson, Sigurðsson)**
- Described as first linguistic benchmark for Icelandic LLMs manually constructed by native speakers; reports SOTA models leave room for improvement; discusses theoretical problems in building such a benchmark and scoring models [VERIFIED abstract-level]; format (minimal pairs vs other) not confirmed — [ACL 2025.nodalida-1.5](https://preview.aclanthology.org/setup/2025.nodalida-1.5); [Landsbókasafn record](https://iris.landsbokasafn.is/en/publications/an-icelandic-linguistic-benchmark-for-large-language-models/)

**Iberian acceptability / minimal-pair resources**
- EsCoLA (Bel, Punsola, Ruíz-Fernández; LREC-COLING 2024): 11,174 labelled sentences from Spanish reference grammars, each tagged with phenomenon; includes human annotation experiment; CC BY 4.0 [VERIFIED] — [ACL 2024.lrec-main.554](https://aclanthology.org/2024.lrec-main.554)
- CatCoLA (CLUB, Procesamiento del Lenguaje Natural 73, 2024): 10,443 Catalan sentences from reference grammars, phenomenon-annotated; test set reportedly withheld to avoid contamination; license conflicting across records (CC BY 4.0 vs CC BY-NC-SA 4.0 vs CC BY-SA 4.0) [VERIFIED, license conflict noted] — [B2FIND CatCoLA](https://b2find.eudat.eu/dataset/4f103bd1-2905-55eb-9f2d-998cdc30bf7c)
- GalCoLA: no evidence found it exists [VERIFIED absence in search] — [search](https://b2find.eudat.eu/dataset/c194cbe0-869a-57dd-9cdd-22c781dbee0b)
- BL2MP (Basque): hosted by Orai NLP on HF (orai-nlp/bl2mp), 1K–10K items, CC BY-NC-SA 4.0; used by HiTZ to evaluate Latxa models (e.g., Latxa Qwen3.5 2B card reports gains of up to ~40 points over base Qwen3.5-2B, base ~49.11) — construction paper not found [VERIFIED existence; method UNKNOWN] — [HF orai-nlp/bl2mp](https://huggingface.co/datasets/orai-nlp/bl2mp/tree/main); [Featherless Latxa card](https://featherless.ai/models/HiTZ/Latxa-Qwen3.5-2B)

**Others [UNVERIFIED-PRIOR]**
- CLAMS (Mueller et al., ACL 2020): cross-lingual syntactic agreement minimal pairs for EN/FR/DE/HE/RU generated from attribute-varying grammars (Marvin & Linzen-style).
- JBLiMP (Someya & Oseki, Findings EACL 2023): ~331 Japanese pairs extracted manually from theoretical linguistics journal articles, validated by native speakers.
- SLING (Song et al., EMNLP 2022): ~38k Chinese pairs built by perturbing Chinese Treebank sentences; reported that templates in earlier CLiMP produced unnatural sentences.

### Inferences
- A Catalan BLiMP should not duplicate MultiBLiMP (agreement only, 2,284 Catalan pairs, already ~96–98% solved); value lies in Catalan-specific phenomena (clitic combinations and pronominal "en/hi", auxiliary selection/participle agreement, periphrastic past "va + inf", "ser/estar", prepositions before "que", gender/number in determiners, pronominal verbs, Spanish-calque contrasts) — these are suggestions, not drawn from a source.
- Best hybrid template for Catalan: RuBLiMP/MultiBLiMP-style perturbation of real Catalan UD/parsed corpus sentences (naturalness, scale) + ZhoBLiMP-style linguist templates for phenomena rare in corpora + QFrBLiMP/EsCoLA/CatCoLA-style seed from normative sources (IEC grammar, Optimot), with dense native validation on a sample.
- The MultiBLiMP result that tiny monolingual models beat Llama3-70B on Catalan/Galician/Basque implies grammar minimal pairs mostly test pretraining exposure; ceilings are high, so hard/rare phenomena are needed for discrimination.

### Gaps
- Exact human validation rates for ZhoBLiMP §3.3 and QFrBLiMP agreement statistics not retrieved (PDF text not extractable).
- BL2MP construction methodology and paper not found.
- Icelandic NoDaLiDa 2025 benchmark format details not retrieved.
- Portuguese efforts (Tucano, PoETa, Napolab) not researched in this pass (tool budget).

## 2. Generative/writing-quality evaluation and language fidelity (language confusion, interference)

### Takeaway
The standard reusable tool is Cohere's Language Confusion Benchmark (LPR/WPR via fastText line-level LID plus word-level checks); it includes Spanish but not Catalan, and its detectors cannot catch Spanish interference in Catalan (same script, closely related). No benchmark found that measures Spanish interference (castellanismos) in generated Galician/Basque/Catalan — this is a genuine gap.

### Cited Findings
- Language Confusion Benchmark (Marchisio et al., EMNLP 2024): 15 languages (EN, FR, DE, ES, PT, IT, JA, KO, ZH, AR, TR, HI, RU, ID, VI) — no Catalan; ~2,600 monolingual + ~4,500 cross-lingual prompts; prompts from Aya (human-written), Dolly (MT + post-edited), Okapi (ChatGPT-translated), ShareGPT, plus commissioned native-written prompts (JA, KO, ES, FR) and complex English prompts [VERIFIED] — [arXiv 2406.20052v3](https://arxiv.org/html/2406.20052v3)
- Metrics: line-level pass rate (fastText LID on lines >4 words), word-level pass rate (English dictionary words for non-Latin scripts; out-of-Unicode-range characters for Latin scripts), harmonic mean LCPR [VERIFIED] — same
- Results: Llama 2/3 70B Instruct monolingual LPR ~46–48%; Command R and GPT ~98–99%; cross-lingual best only low 90s; higher temperature increases confusion; few-shot raises Command R Base cross-lingual LPR 1.1% -> 95.0%; English-only SFT worsens confusion, adding 10% multilingual SFT nearly eliminates line-level confusion; English-only preference tuning hurts word-level; W4 quantization hurts [VERIFIED] — same
- Galician: a CiTIUS case study proposes separating task-based from language-based evaluation for Galician continued pretraining; whether it measures Spanish interference is unconfirmed [VERIFIED existence] — [CiTIUS publication page](https://citius.gal/en/research/publications/continued-pretraining-and-interpretability-based-evaluation-for-low-resource-languages-a-galician-case-study/)
- IberoBench (COLING 2025): Basque, Catalan, Galician, Spanish, Portuguese; 62 tasks / 179 subtasks; IberBench: 101 datasets, 22 task categories — neither described as measuring interference [VERIFIED] — [IberoBench ACL 2025.coling-main.699](https://preview.aclanthology.org/credits/2025.coling-main.699); [IberBench arXiv 2504.16921](https://web3.arxiv.org/pdf/2504.16921)

### Inferences
- For Catalan, LID-based confusion metrics will under-detect the real failure mode (Spanish lexical/syntactic calques inside otherwise-Catalan text). A Catalan fidelity metric likely needs: (i) a lexicon of known castellanismos/barbarismes (e.g., from normative sources), (ii) a grammar checker such as LanguageTool's Catalan rules (which, from prior knowledge, are extensive — UNVERIFIED-PRIOR), and (iii) a native-validated LLM/classifier judge for subtler calques — reported per 1k words.
- LCB's finding that temperature and English-only post-training increase confusion suggests a Catalan generation benchmark should fix decoding settings and report them.

### Gaps
- No published Spanish-interference metric for Galician/Basque/Catalan model outputs found.
- No study found on grammatical error rates of LLM outputs in Catalan specifically (may be covered by other researchers).

## 3. Culturally native benchmarks built from scratch

### Takeaway
Native-authored benchmarks (BertaQA, INCLUDE, BLEnD) consistently show large local-vs-global gaps and that translated English benchmarks miss regional knowledge; BertaQA is the closest template for Catalan (written in the local language, professionally translated to English for parallel comparison, local vs global split).

### Cited Findings
- BertaQA (HiTZ/UPV-EHU, NeurIPS 2024 D&B): 4,756 MCQs, authored in Basque, professionally translated to English (fully parallel); split into local (Basque culture) vs global subsets; 8 categories; easy/medium/hard labels [VERIFIED] — [arXiv 2406.07302](https://arxiv.org/html/2406.07302v2); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3bb42f6bb1b1ab6809afd6c90865b087-Abstract.html)
- BertaQA results: SOTA LLMs struggle on local, excel on global; continued pretraining in Basque improves local knowledge even when queried in English (Llama 2 70B +13.46 on English local subset) but costs global performance; knowledge not language-agnostic — models do better in the language through which the knowledge was acquired [VERIFIED] — same; a 26.66-point average local–global gap reported by a third-party listing, not confirmed in paper — [alphaXiv](https://alphaxiv.org/benchmarks/university-of-the-basque-country-upvehu/bertaqa)
- INCLUDE (ICLR 2025): 197,243 MCQs from 1,926 local exams (academic, professional, licensing), 52 countries, 44 languages, 15 scripts; collected with native speakers and regional associations; motivation avoids translationese; GPT-4o ~77% avg; models good on French/Spanish, weaker on Armenian, Greek, Urdu, and on languages not intentionally trained [VERIFIED] — [arXiv 2411.19799](https://arxiv.org/pdf/2411.19799); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ced46a50befedcb884ccf0cbe8c3ad23-Abstract-Conference.html)
- BLEnD (NeurIPS 2024 D&B): ~52.6k hand-crafted QA pairs on everyday cultural knowledge (food, holidays, sports, etc.), 16 countries/regions, 13 languages; short-answer and MCQ formats; up to 57.34% performance difference across cultures for GPT-4 (short-answer); for low-resource languages, LLMs answer better in English [VERIFIED] — [arXiv 2406.09948](https://arxiv.org/pdf/2406.09948)
- [UNVERIFIED-PRIOR] KoBEST (Korean, 2022): 5 tasks built by Korean linguists from scratch rather than translated; often cited as template for native task design.

### Inferences
- Catalan analogue: a "local Catalan culture vs global" MCQ set authored in Catalan (Catalan-speaking territories: Catalonia, Valencia, Balearics, Andorra, Northern Catalonia), translated to English for parallel testing, with difficulty labels. INCLUDE-style harvesting of Catalan exams (e.g., Catalan proficiency/oposiciones exams) is a second source, but licensing and contamination (exams are on the web) need checking.
- Everyday-knowledge items (BLEnD style) are less likely to be on Wikipedia and thus more discriminative.

### Gaps
- Galician culturally native QA benchmark not confirmed in this pass.
- Whether INCLUDE contains Catalan or Basque exams not checked.

## 4. Dynamic / procedurally generated, contamination-resistant benchmarks in non-English languages

### Takeaway
Very little exists for natural-language (non-code) multilingual dynamic benchmarks; the closest are game-based (TCG-Bench, EN+Arabic) and the decontamination filtering used by RuBLiMP and CatCoLA's withheld test set. Template/grammar-driven generators (BLiMP, ZhoBLiMP GUI, MultiBLiMP pipeline) are the natural basis for regenerable, contamination-resistant Catalan test sets.

### Cited Findings
- TCG-Bench (KAUST, Findings EACL 2026): trading-card-game benchmark separating public engine from hidden card implementations; procedurally generates difficulty via Monte Carlo; parallel English and Arabic; reports a 47.4% English–Arabic gap at 32B [VERIFIED via search summary] — [EACL Findings 2026.findings-eacl.353](https://preview.aclanthology.org/ingest-eacl/2026.findings-eacl.353.pdf)
- Most dynamic benchmarks found (DyCodeEval, Code2Bench, Multi-LCB) target programming languages, not natural languages [VERIFIED] — [DyCodeEval arXiv 2503.04149](https://arxiv.org/html/2503.04149v2); [Code2Bench arXiv 2508.07180](https://arxiv.org/abs/2508.07180v1)
- clembench (May 2024): dynamic, game-based, multilingual framework [VERIFIED existence] — [arXiv 2405.20859](https://arxiv.org/abs/2405.20859v1)
- RuBLiMP: MIN-K% PROB-based decontamination of source sentences (see §1) — [arXiv 2406.19232](https://arxiv.org/pdf/2406.19232)
- CatCoLA test set withheld to avoid contamination (per Dataverse record) — [B2FIND](https://b2find.eudat.eu/dataset/4f103bd1-2905-55eb-9f2d-998cdc30bf7c)
- MultiBLiMP pipeline explicitly reusable for new UD releases/phenomena — [arXiv 2504.02768](https://arxiv.org/pdf/2504.02768)

### Inferences
- A Catalan generator that re-samples vocabulary/templates (BLiMP/ZhoBLiMP) or re-perturbs fresh corpus sentences (RuBLiMP/MultiBLiMP, e.g., from post-cutoff Catalan news) can produce fresh test instances per release, giving contamination resistance; keep a private held-out seed set (CatCoLA practice).

### Gaps
- No dynamic, procedurally generated natural-language benchmark specifically for Catalan, Spanish, Basque or Galician was found.

## 5. Known pitfalls and reusable best practices

### Takeaway
Main documented pitfalls: unverified synthetic/MT benchmarks contain severely flawed items (Icelandic LREC 2026), LLM judges are less reliable and biased in low-resource languages (including toward translationese and against certain languages), length/tokenization bias in log-prob comparison (ZhoBLiMP SLLN-LP; MultiBLiMP over-segmentation), and post-training/chat models score lower on log-prob tests.

### Cited Findings
- "Who Benchmarks the Benchmarks? A Case Study of LLM Evaluation in Icelandic" (Ingimundarson et al., arXiv Mar 2026, LREC 2026): unverified synthetic or machine-translated benchmarks "commonly contain severely flawed test examples" likely to skew results; quantitative error analysis shows clear gap between human-authored/translated and synthetic/MT benchmarks; MT benchmark quality capped by MT quality for that language; recommends avoiding unverified synthetic/MT benchmarks in low/mid-resource settings [VERIFIED abstract] — [arXiv 2603.16406](https://arxiv.org/abs/2603.16406)
- Multilingual LLM-as-judge is inconsistent across parallel inputs in different languages, worse for low-resource languages; builds on self-enhancement bias (judges favour own outputs) [VERIFIED] — [arXiv 2505.12201](https://arxiv.org/pdf/2505.12201)
- Judges show language bias not reducible to perplexity/fluency, especially for low-resource languages [VERIFIED, preprint] — [arXiv 2601.13649](https://www.arxiv.org/pdf/2601.13649)
- Translationese bias: LLM judges favour machine-translated content over human-authored references even when semantically flawed; worse in low-resource languages; DIBJUDGE mitigation proposed [VERIFIED, preprint] — [arXiv 2603.10351](https://arxiv.org/pdf/2603.10351)
- Survey: of 650 ACL papers mentioning LLM-as-judge, only 33 focus on low-resource/multilingual; tendency to overtrust judges and use a single judge model [VERIFIED, preprint] — [arXiv 2607.02235](https://www.alphaxiv.org/abs/2607.02235.md)
- Judge-bias claims are sensitive to benchmark construction quality; pairwise judging more fragile than pointwise [VERIFIED] — [ACL SRW 2026.acl-srw.33](https://aclanthology.org/2026.acl-srw.33.pdf)
- Length bias in pair log-probs addressed with SLLN-LP (ZhoBLiMP); RuBLiMP keeps pairs length-matched by edit design — [arXiv 2411.06096](https://arxiv.org/pdf/2411.06096); [arXiv 2406.19232](https://arxiv.org/pdf/2406.19232)
- Post-training negatively affects minimal-pair accuracy; evaluate base models — [arXiv 2504.02768v2](https://arxiv.org/html/2504.02768v2)
- Translated benchmarks miss regional knowledge / suffer translationese (INCLUDE motivation) — [arXiv 2411.19799](https://arxiv.org/pdf/2411.19799)
- [UNVERIFIED-PRIOR] Template artifacts: BLiMP-style templates yield semantically odd sentences; SLING authors argued CLiMP templates produced unnatural Chinese, motivating corpus-based perturbation.

### Inferences — reusable best practices for a Catalan generator + benchmark
1. Construction: hybrid — corpus perturbation of real Catalan sentences (UD Catalan-AnCora / parsed news) for naturalness + linguist templates for rare phenomena + seeds from normative sources (IEC, Optimot) — with every paradigm tagged by phenomenon (BLiMP/EsCoLA/CatCoLA practice).
2. Edit design: single-locus edits, length/token-matched where possible; report length-normalized and raw log-prob accuracy (SLLN-LP); report tokenization-sensitivity.
3. Validation: dense native judgment on a sample per paradigm (QFrBLiMP used 12 annotators per pair; BLiMP reported 96.4% human agreement) and drop paradigms below an agreement threshold; publish agreement stats (gap noted in QFrBLiMP retrieval). Include Valencian/Balearic variant awareness so dialectally acceptable forms are not labelled ungrammatical (inference).
4. Balancing: cap per-condition items and downsample over distance/attractors (MultiBLiMP).
5. Contamination: decontaminate seeds (MIN-K% PROB à la RuBLiMP), keep a private test split (CatCoLA), and make the generator regenerable for fresh versions.
6. Baselines: include monolingual small models (Goldfish-style) and base vs instruct variants; MultiBLiMP shows small monolingual models can beat 70B on Catalan.
7. Generative evaluation: never rely on a single LLM judge, especially from the same family as the generator; calibrate judges against native raters per language; prefer pointwise rubrics; check translationese bias; complement with deterministic metrics (LID line pass rate, castellanismo lexicon hits, grammar-checker error rate per 1k words); fix and report decoding temperature.
8. Synthetic/MT data: never ship unverified; human-verify at least a stratified sample and report item error rates (Icelandic LREC 2026).
9. Culture: author natively in Catalan, translate to English for parallel local-vs-global analysis (BertaQA), include everyday knowledge (BLEnD).

### Gaps
- No quantitative error rates retrieved from the Icelandic 2026 paper (abstract only).
- No direct study of judge bias toward the generator model specifically in Catalan/Iberian languages found.
- Portuguese (Tucano, PoETa v2, Napolab) and Galician native benchmark methodologies not covered in this pass.
