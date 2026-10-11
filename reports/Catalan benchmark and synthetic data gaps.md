# Target Catalan normativity, not more chat data

As of October 2026, the biggest open gap in Catalan LLM evaluation is not knowledge or reasoning. It is **normative correctness and language fidelity in generated Catalan**: Spanish interference (castellanismes), weak pronouns, prepositions, diacritics, invented normative claims, and whether a model stays in Valencian or Balearic when asked to. No public benchmark measures any of these. The Catalan evaluation pool has about 35–40 sets, and most of them are human translations of old English benchmarks (12 of the 18 Catalan sets in La Leaderboard). They are mostly multiple-choice or classification, almost all target Central Catalan, and none has had a published contamination analysis. On the synthetic-data side, the Barcelona Supercomputing Center (BSC) already ships a large general-purpose Catalan post-training suite: ALIA-2606, with 87k Catalan SFT conversations plus DPO data, generated mainly with DeepSeek-V3. Building another general Catalan instruction or chat generator would largely be double work. The evidence points to one recommended scope. Build an open, procedural, verifier-backed generator focused on Catalan linguistic phenomena and Spanish-interference contrasts. Use LanguageTool's Catalan rules, the Softcatalà lexicon and UD AnCora as seeds. On top of it, build a regenerable benchmark with four tracks: normative minimal pairs beyond agreement, interference rate in free generation, dialect fidelity, and "hallucinated normativity" QA. Package it as lm-evaluation-harness tasks next to the existing `catalan_bench`. The main risks are seed licences (LGPL rule files, a CatCoLA licence that is possibly non-commercial, closed IEC/AVL dictionaries), LLM judges that are unreliable for Catalan normativity, and the cost of native validation.

## Catalan already has about 40 eval sets, most of them translated multiple-choice in Central Catalan

Coverage is broad on classic NLU and thin almost everywhere else. Projecte AINA/BSC built most of the pool, and it lives in lm-evaluation-harness as `catalan_bench` ([lm-eval catalan_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/catalan_bench)). La Leaderboard's Table 5 lists 18 Catalan sets with sizes. **Twelve are human translations of English benchmarks**: ARC, Belebele, COPA, MGSM, OpenBookQA, PAWS, PIQA, SIQA, WNLI, XNLI, XQuAD and xStoryCloze. Only six are native: caBREU, CatalanQA, CatCoLA, CoQCat, Parafraseja and TE-ca ([La Leaderboard, arXiv 2507.00999](https://arxiv.org/pdf/2507.00999)). Several sets are small. **MGSM_ca has 250 items, WNLI_ca 146 and caBREU 301**, and caBREU is the only native summarization set. The older encoder-era CLUB suite adds native NER, POS, topic classification, STS and the ViquiQuAD/VilaQuAD extractive QA sets ([CLUB, arXiv 2112.01894](https://arxiv.org/pdf/2112.01894)).

| Area | What exists | Origin | Gap |
|---|---|---|---|
| NLI, paraphrase, extractive QA, commonsense MCQ | TE-ca, XNLI_ca, PAWS_ca, Parafraseja, CatalanQA, CoQCat, XQuAD_ca, COPA/PIQA/SIQA/ARC/OpenBookQA_ca | Mixed; mostly translated | Saturated, contamination-prone |
| Grammar/acceptability | CatCoLA (1,020 eval items; test split withheld), MultiBLiMP-ca (2,284 agreement pairs) | Native | MultiBLiMP covers agreement only and is near ceiling (97.7%) |
| Math | MGSM_ca (250) | Translated | Tiny |
| Truthfulness | VeritasQA (353, written natively for es/ca/gl/en) | Native | Small |
| Instruction following, bias, alignment | IFEval_ca, hhh_alignment_ca, CaBBQ, EQ-bench_ca, CrowS-Pairs-ca | Translated or adapted | No native Catalan instruction-following set |
| Valencian | CIEACOVA (native proficiency exams), TerretaQA, XNLI_va, TruthfulQA_va, COCOTEROS_va, phrases_va | Mostly translated | Only GPLSI, the NLP group at the Universitat d'Alacant, covers Valencian |
| Balearic, North-Western | None found | — | Fully missing |
| Normative writing quality, interference, cultural knowledge, chat/arena | None found | — | Fully missing |

The Valencian additions come from GPLSI and joined the harness after IberoBench ([catalan_bench README](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/catalan_bench/README.md)). BSC's Hugging Face listings also show Catalan translations of IFEval and HHH alignment, plus CaBBQ and EQ-bench_ca ([HF API projecte-aina](https://huggingface.co/api/datasets?author=projecte-aina&search=ca&limit=300); [HF API BSC-LT](https://huggingface.co/api/datasets?author=BSC-LT&limit=300)). So the instruction-following gap is one of **native** design, not total absence. VeritasQA is the one recent native addition. It was written for transferability, replacing the dated, English-centric TruthfulQA ([VeritasQA, COLING 2025](https://aclanthology.org/2025.coling-main.366/)).

No Catalan-only leaderboard exists. Catalan is scored inside La Leaderboard (66 datasets, 50 models) and the IberBench leaderboard (12 Catalan entries, 23 models up to 14B parameters, no Valencian set) ([IberBench, arXiv 2504.16921](https://arxiv.org/html/2504.16921); [IberBench leaderboard](https://huggingface.co/spaces/iberbench/leaderboard)). Both draw on the same AINA/IberoBench pool, so they are not independent measurements. La Leaderboard itself lists contamination analysis as **"pending work"** ([arXiv 2507.00999](https://arxiv.org/pdf/2507.00999)). The contamination risk is concrete. The English sources of the translated sets have been public since 2016–2020, so models can answer through cross-lingual memorisation. The native sets were built from Viquipèdia, Vilaweb and ACN text, the same sources that feed Catalan pretraining corpora. That is an inference from the notes, not a measured result, but no one has published evidence against it. Most tasks are scored by log-likelihood MCQ or by BLEU/ROUGE/F1, and **no rubric or judge for Catalan fluency or grammar exists**.

Neighbouring widely used multilingual sets mostly skip Catalan. Global-MMLU's 42 languages do not include it ([Global-MMLU card](https://huggingface.co/datasets/CohereLabs/Global-MMLU)). The Language Confusion Benchmark covers 15 languages, again without Catalan ([arXiv 2406.20052](https://arxiv.org/html/2406.20052v3)). Two items remain unconfirmed: whether INCLUDE, MMLU-ProX or EuroEval include Catalan, and whether Accent Obert (formerly Fundació .cat) has published its planned chatbot exam ranking. Check both before finalising scope.

## BSC's ALIA-2606 suite already covers general Catalan instruction data

The synthetic-data landscape is dominated by one producer with a mature pipeline. The ALIA-2606 post-training suite, published June–July 2026, has these parts:

- **ALIA-2606-SFT:** 713,619 conversations, 87,011 of them Catalan.
- **DPO-helpfulness:** 52,081 prompts, 7,535 Catalan.
- **DPO-safety:** 18,033 prompts, 5,397 Catalan.
- **A function-calling SFT set.**

Most synthetic subsets were generated, and the preferences rated, by **DeepSeek-V3-0324** ([ALIA-2606-SFT](https://huggingface.co/datasets/BSC-LT/ALIA-2606-SFT); [ALIA-40b-instruct-2606](https://huggingface.co/BSC-LT/ALIA-40b-instruct-2606)). The recipe combines several sources:

- WildChat prompts with DeepSeek answers.
- FineWeb-Edu-grounded QA.
- Instruction-following augmentation filtered with verifiable constraints.
- Multi-turn augmentation.
- 205k translated OpenR1 math items.

BSC also released **m-personas**: about 510k PersonaHub-style instances in 15 languages, about 41.5k of them Catalan, seeded from FineWeb-2. They were filtered with GlotLID (language identification) and MinHash/LaBSE deduplication, which removed **4–35% of the generated content** depending on language ([m-personas](https://huggingface.co/datasets/BSC-LT/m-personas)). Older BSC/AINA releases are InstruCAT (216k template-converted task instructions, CC BY-NC-ND), RAG_Multilingual (56k items, Mixtral-generated) and translated Dolly and OASST sets ([InstruCAT](https://huggingface.co/datasets/projecte-aina/InstruCAT); [RAG_Multilingual](https://huggingface.co/datasets/projecte-aina/RAG_Multilingual)). The community CataLlama project adds Catalan-Instruct-V2, about 710k rows of which about 250k are machine-translated, and Catalan-DPO-V2 with 23.5k pairs ([Catalan-Instruct-V2](https://huggingface.co/datasets/catallama/Catalan-Instruct-V2)). Catalan-Instruct-V2 carries a licence conflict: it inherits from InstruCAT's non-commercial, no-derivatives licence but advertises CC-BY/CC-BY-SA.

The conclusion for the user's first goal is direct. **General chat SFT, persona-conditioned conversations, RAG QA, translated math, and LLM-judged helpfulness or safety preferences are already done for Catalan.** Reproducing them adds little. The open niches are what BSC has not shipped:

- **Public generator code.** No prompts or pipelines were found for any BSC dataset.
- **Pretraining-scale synthetic Catalan text** (Cosmopedia-style textbooks) and a Catalan Magpie set. Neither exists, and Catalan is absent from the Aya dataset ([aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset)).
- **Human-annotated Catalan preferences.** All existing Catalan preference data is AI-judged.
- **Variety-controlled Valencian or Balearic instructions.** GPLSI's Aitana models rely on domain corpora (DOGV, Corts Valencianes) and on BSC's internal "ALIA Instruction v12". The Aitana-7B-S-Instruct card says it was not instruction-tuned on Catalan data ([Aitana-7B-S-Instruct mirror](https://friendli.ai/models/gplsi/Aitana-7B-S-Instruct)).
- **Catalan grammatical error correction (GEC).** No public Catalan GEC dataset or synthetic-error generator was found. Softcatalà's corrector is rule-based LanguageTool with more than 1,700 rules ([UB Retrucs](https://www.ub.edu/retrucs/?p=5140)). A Basque project showed the template: rule-based corruption of clean news text, then a trained corrector.

Existing pipelines also share a quality blind spot. GlotLID-style filters catch whole off-language outputs but not Spanish calques inside otherwise-Catalan text. A 2026 study found Spanish LLM generations averaged **45.7% classified as "translated"** ([arXiv 2608.17399](https://arxiv.org/pdf/2608.17399)). Much of the Catalan SFT pool is itself machine-translated, so translationese and interference are plausibly built into it. No one has measured how much.

## LLMs fail at Catalan normativity, and almost nobody measures it

The best-quantified weakness is Spanish interference. A UPF study by Brochhagen and Almena, published in *Linguamática*, tested six LLMs on 160 sentences covering eight preposition structures. **Multilingual models chose the Spanish-influenced non-normative form in 55% of relevant cases, and Catalan-specialised models in 27%** ([VilaWeb](https://www.vilaweb.cat/noticies/upf-ia-formes-no-normatives-catala/)). A typical error is the Spanish-calqued prepositional direct object, *he vist **al** professor* for normative *he vist **el** professor* ([Ara](https://en.ara.cat/languages/have-seen-the-professor-ai-spreads-mistakes-in-catalan-due-to-spanish-influence_1_5659940.html)). The second documented failure is **hallucinated normativity**. In an IEC/Ara test, ChatGPT called Castilian calques normative and invented DIEC (the IEC's normative dictionary) entries and definitions. The president of the IEC's Philological Section warned that it "sometimes spreads clearly false information" ([Ara, Oct 2025](https://en.ara.cat/languages/do-you-have-questions-about-standard-catalan-don-t-ask-them-in-the-gpt-chat_1_5513337.html)). Other studies document:

- **Language drift.** ChatGPT tends to answer in Spanish even when instructed in Catalan ([UVic thesis](https://repositori.uvic-ucc.cat/handle/10854/8446)).
- **Spelling slips in deployed assistants.** Meta AI still made occasional spelling errors after it finally began answering in Catalan in late 2025 ([VilaWeb](https://www.vilaweb.cat/noticies/meta-ia-parla-finalment-catala-mesos-despres-problemes/)).
- **Weaker morphological generalisation.** Six LLMs scored lower on Catalan wug tests (inflecting invented words) than on Spanish or English, which the authors attribute to data scarcity ([arXiv 2510.12463](https://arxiv.org/abs/2510.12463)).

Most phenomena on a Catalan teacher's error list have **no published LLM-specific measurement**: weak-pronoun combinations, participle agreement with clitics, the 2016 diacritic reform, apostrophation, ser/estar, and the classic castellanismes (*tenir que*, *donar-se compte*). Dialect behaviour is equally unmeasured. No study tests whether models drift to Central/IEC forms or to Spanish when asked for Valencian (AVL norms) or Balearic. Cultural knowledge is similar. Accent Obert argues that chatbots "think" in English and has announced exam-based testing of 12 chatbots, but no results were found ([VilaWeb](https://www.vilaweb.cat/noticies/ia-pensa-catala-accent-obert/)). BertaQA's Basque results suggest the size of a possible cultural gap: models do well on global questions, struggle on local ones, and answer better in the language through which they acquired the knowledge ([arXiv 2406.07302](https://arxiv.org/html/2406.07302v2)).

This creates a methodological trap for the benchmark. If models hallucinate Catalan normativity, then an LLM judge scoring normative correctness carries the same bias. Multilingual judges are less consistent in low-resource languages ([arXiv 2505.12201](https://arxiv.org/pdf/2505.12201)) and **favour machine-translated text over human-authored text** ([arXiv 2603.10351](https://arxiv.org/pdf/2603.10351)). A survey of 650 ACL papers that use LLM judges found only 33 focused on multilingual or low-resource settings, and a tendency to rely on a single judge ([arXiv 2607.02235](https://www.alphaxiv.org/abs/2607.02235.md)). For a correctness benchmark, the verifier must be deterministic code or rules, with an LLM judge used at most for naturalness.

## Open seed resources cover the hard part, but the dictionaries do not

The seeds needed for a normativity-focused generator exist under open licences. **LanguageTool's Catalan module** (LGPL 2.1+) encodes the error taxonomy directly. Its rule categories include weak pronouns, apostrophation, agreement and a dedicated "Errors de traductor automàtic" (machine-translation errors) group. It also ships plain-text pair lists: `replace_diacritics_iec.txt`, `confusion_pairs.txt`, `replace_anglicism.txt`, `replace_balearic.txt`, `replace_dnv*.txt` and `coherency-valencia.txt` ([LanguageTool ca rules](https://github.com/languagetool-org/languagetool/tree/master/languagetool-language-modules/ca/src/main/resources/org/languagetool/rules/ca); [rule browser](https://community.languagetool.org/rule/list?lang=ca)). It runs as a local server with `ca-ES` and `ca-ES-valencia` variants ([language_tool_python](https://pypi.org/project/language-tool-python/2.7.1)). One caveat: some rules run only on the cloud endpoint ([LT forum](https://forum.languagetool.org/t/locally-server-behaves-different-from-browser-extension/8511)).

| Seed | Use in the project | Licence | Publishing risk |
|---|---|---|---|
| LanguageTool ca rules + replace/confusion lists | Error injection, interference lexicon, automatic checker | LGPL 2.1+ | Unclear whether generated sentences are derivative; attribute and keep compatible |
| Softcatalà catalan-dict-tools (~10M inflected forms, general + Valencian) | Correct inflection in templates | LGPL 2.1 / GPL 2 | Same as above |
| UD Catalan-AnCora (16,678 sentences, 531k tokens) | Real sentence frames for perturbation | CC BY 4.0 (notes inherited AnCora terms) | Low |
| Aina CA-EN parallel (15M pairs) | Calque mining, parallel items | CC BY 4.0 | Low; CA-DE is NC, CA-FR is CC BY-SA |
| CaWaC (780M tokens) / Catalan Textual Corpus (1.76B) | Clean source text | CC BY-SA 3.0 / 4.0 | Share-alike applies |
| FineWeb-2 `cat_Latn` (15.5M documents) | Grounding documents, fresh text | Per dataset card | Check card |
| CatCoLA (10,443 phenomenon-tagged sentences) | Phenomenon inventory and seeds | Conflicting: CC BY, CC BY-SA or CC BY-NC-SA | Resolve before deriving |
| DIEC2, DNV (93k+ entries), Optimot, Termcat | Manual validation only | Copyright-reserved, no open API found | Do not bulk-extract |

Sources: [Softcatalà catalan-dict-tools](https://github.com/Softcatala/catalan-dict-tools); [UD Catalan-AnCora](https://universaldependencies.org/treebanks/ca_ancora/index.html); [Aina CA-EN](https://huggingface.co/datasets/projecte-aina/CA-EN_Parallel_Corpus/blob/4d1041c3cb3bd1916d2821e9d8e710dd2278d478/README.md); [CaWaC](https://huggingface.co/datasets/community-datasets/cawac/blob/d80d4a7a20a333e7d1d70aa61b27a84c790bb658/README.md); [Catalan Textual Corpus](https://zenodo.org/record/4519349); [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/blob/cab26bb3cab1ca7c26bb13a48f80fc3d2def4620/README.md); [CatCoLA B2FIND](https://b2find.eudat.eu/dataset/4f103bd1-2905-55eb-9f2d-998cdc30bf7c); [AVL](https://www.avl.gva.es/welcome-en/). For morphology-aware realisation, the GF Resource Grammar Library includes Catalan ([RGL synopsis](https://www.grammaticalframework.org/lib/doc/synopsis/index.html)), and Apertium's apertium-cat provides a GPL-2.0 morphological generator ([Debian metadata](https://metadata.ftp-master.debian.org/changelogs/main/a/apertium-spa-cat/stable_copyright)).

## Other mid-resource languages show how to build it without the known traps

Minimal-pair benchmarks come in three construction families, and a Catalan benchmark should combine them:

- **Linguist-written templates with a feature-annotated vocabulary.** BLiMP has 67 paradigms of 1,000 pairs with 96.4% human agreement ([BLiMP](https://aclanthology.org/2020.tacl-1.25)). ZhoBLiMP has 118 paradigms and a no-code GUI for linguists ([arXiv 2411.06096](https://arxiv.org/pdf/2411.06096)).
- **Rule-based perturbation of parsed corpus sentences.** MultiBLiMP is fully automated from UD and UniMorph ([arXiv 2504.02768](https://arxiv.org/html/2504.02768v2)). RuBLiMP has 45k pairs from expert dependency-tree rules and decontaminates its source sentences with MIN-K% PROB ([arXiv 2406.19232](https://arxiv.org/pdf/2406.19232)).
- **Manual pairs from prescriptive sources with dense native judgment.** QFrBLiMP has 1,761 pairs from the OQLF's (Office québécois de la langue française) usage guide, each judged by 12 speakers. It found models score **markedly worse on it than on MultiBLiMP French** ([arXiv 2509.25664](https://arxiv.org/abs/2509.25664v2)).

That last result matters most for Catalan. MultiBLiMP's 2,284 Catalan agreement pairs are already solved at 97.7% by a 125M monolingual model, which beats Llama3-70B. Agreement mostly tests pretraining exposure. **Normative and interference-sensitive phenomena are where headroom lies.** The same papers flag technical pitfalls:

- Length bias between pair members, which ZhoBLiMP counters with sub-linear length normalisation.
- Subword over-segmentation of the correct form.
- Lower minimal-pair scores after chat post-training, so evaluate base and instruct variants separately.

For generative fidelity, Cohere's Language Confusion Benchmark supplies the metric shape: line-level and word-level pass rates, with fixed decoding. Its findings also matter here. Higher temperature and English-only post-training increase confusion ([arXiv 2406.20052](https://arxiv.org/html/2406.20052v3)). But its fastText language ID cannot see Spanish calques in Catalan, so a Catalan version needs a castellanisme lexicon and grammar-checker hits per 1,000 words, not only language ID. No interference metric exists for Catalan, Galician or Basque, so this would be new. For cultural items, BertaQA's design is the template: author natively, translate professionally to English, and split local from global questions. BLEnD shows that everyday-knowledge items are more discriminative than Wikipedia facts ([arXiv 2406.09948](https://arxiv.org/pdf/2406.09948)). The Icelandic LREC 2026 audit gives the overriding warning: **unverified synthetic or machine-translated benchmarks "commonly contain severely flawed test examples"** ([arXiv 2603.16406](https://arxiv.org/abs/2603.16406)). Native validation of a stratified sample, with published per-item error rates, is mandatory.

## Recommended scope: an open phenomenon generator feeding a four-track benchmark

**Goal 1, the generator.** Scope it as an open-source, procedural-first Catalan generator, not a chat-data factory. Model the architecture on reasoning-gym (Apache-2.0, more than 100 tasks). Each task is a config dataclass plus a `ProceduralDataset` subclass seeded per item with `Random(seed+idx)`, returning `question`/`answer`/`metadata`, with a `score_answer` verifier and a difficulty curriculum ([reasoning-gym](https://github.com/open-thought/reasoning-gym); [chain_sum.py](https://raw.githubusercontent.com/open-thought/reasoning-gym/main/reasoning_gym/arithmetic/chain_sum.py)). The generator should have four layers:

1. **Seed layer.** Converts LanguageTool replace/confusion lists and grammar patterns, Softcatalà inflections, AnCora-UD sentences and the Valencian/Balearic variant files into typed (correct, incorrect, phenomenon, norm, variety) records.
2. **Realisation layer.** Builds pairs in two ways: by perturbing real AnCora or post-cutoff news sentences (single-locus, length-matched edits), and by morphology-aware templates for rare phenomena (Softcatalà or Apertium lexicon, optionally GF RGL Cat).
3. **Optional LLM surface layer.** Uses an open-weight model to paraphrase or wrap items, then re-checks that the gold label survived. The YourBench (document-to-QA) and distilabel or NeMo Data Designer pipeline frameworks fit here, and all are Apache-2.0 ([YourBench](https://github.com/huggingface/yourbench); [distilabel](https://github.com/argilla-io/distilabel)).
4. **QC layer.** Runs GlotLID or OpenLID-v3 for ca/es/oc leakage ([OpenLID-v3](https://huggingface.co/HPLT/OpenLID-v3/blob/main/README.md)), a local LanguageTool `ca-ES`/`ca-ES-valencia` check, MinHash deduplication against itself and against public Catalan benchmarks, and native spot-checks.

The same pipeline yields synthetic GEC pairs as a by-product: inject a rule's error into clean FineWeb-2 or CaWaC text. That fills the confirmed Catalan GEC gap at almost no extra cost.

**Goal 2, the benchmark.** Build four tracks on that output, each scored without relying on an LLM judge:

- **Track A, normative minimal pairs beyond agreement.** Prepositions (anchored on the UPF 55% result), weak-pronoun combinations, participle agreement with clitics, post-2016 diacritics, apostrophation and contractions, ser/estar, and castellanisme contrasts. Score by log-likelihood on base models, with raw and length-normalised accuracy.
- **Track B, interference in free generation.** Fixed prompts at fixed temperature. Report LanguageTool matches and castellanisme-lexicon hits per 1,000 words, and line-level language ID pass rate.
- **Track C, dialect fidelity.** Prompt for Valencian (AVL) or Balearic output and measure off-variety forms with the `coherency-valencia`/`replace_dnv`/`replace_balearic` lists. Every item records which norm (IEC or AVL) it follows.
- **Track D, hallucinated normativity.** Questions such as "is X normative / in the DIEC?", with gold answers verified manually against DIEC/DNV/Optimot.

A fifth track, BertaQA-style local-versus-global cultural MCQ, is valuable but labour-intensive. Treat it as phase 2 unless Accent Obert or BSC has already published one. Package the tracks as lm-eval YAMLs (`multiple_choice` for Track A; `generate_until` with a `process_results` function for B–D) on a Hugging Face dataset, aiming for an upstream PR next to `catalan_bench` ([lm-eval task guide](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/docs/task_guide.md)). For contamination resistance, publish the generator and a seed-A test split, and keep a private split with held-out seeds, templates and lexicon entries, as CatCoLA withholds its test set. Release versioned refreshes, in the spirit of LiveBench's monthly objective-scored updates ([arXiv 2509.24210](https://arxiv.org/pdf/2509.24210)). Include small monolingual models (Goldfish-style), Salamandra/ALIA and Aitana as baselines.

**First steps,** in order:

1. Resolve CatCoLA's licence with its authors, and decide on a data licence compatible with LGPL and CC BY-SA seeds.
2. Retrieve the *Linguamática* paper to reuse its eight preposition structures as the first Track A paradigms.
3. Parse the LanguageTool Catalan resources into a typed phenomenon inventory, and pick about 10 high-yield paradigms.
4. Build two or three reasoning-gym-style task classes end to end, with LanguageTool and language-ID QC.
5. Run a pilot: 2 native annotators on about 50 items per paradigm. Drop paradigms below an agreement threshold.
6. Evaluate 6–8 models (base and instruct) to confirm the paradigms show headroom before scaling.

**Open risks.** Five risks remain:

- **Licensing.** It is unsettled whether sentences generated from LGPL rule files or a GPL lexicon are derivative works. CatCoLA may be non-commercial, and DIEC2/DNV/Optimot have no open terms. Commercial LLM providers' output-use terms were not checked and should be read before any closed model is used as generator.
- **Norm politics.** Labelling a Valencian or Balearic form "ungrammatical" against IEC norms invalidates items, so every item needs an explicit norm field.
- **Template artefacts and LanguageTool coverage.** Templates produce unnatural sentences, and LanguageTool false positives on templated text require rule whitelisting. The cloud-only rules mean local checking under-detects.
- **Cost.** Native validation is the true bottleneck: QFrBLiMP used 12 judges per pair.
- **Possible overlap.** BSC, GPLSI or Accent Obert could release overlapping work. A quick check of the BSC-LT and GPLSI Hugging Face organisations and of Accent Obert's ranking should precede the build.

## Conclusion

The Catalan ecosystem has advanced in an unusual order. It has abundant translated NLU benchmarks and a state-of-the-art synthetic post-training pipeline, but no measurement of the thing Catalan speakers and institutions complain about most: models writing Spanish-shaped Catalan and asserting false norms. That inverts the usual mid-resource playbook. The scarce asset is not more generated text but **trustworthy verifiers of Catalan correctness**. Those verifiers already exist as community-maintained LanguageTool rules and Softcatalà lexicons, and they have never been wired into an LLM evaluation pipeline. A generator built around them serves both goals at once: its error-injection layer produces training and GEC data, and its verifiers make the benchmark judge-free.

The strategic payoff is complementarity. BSC can use such a benchmark to measure whether ALIA's DeepSeek-generated and translated SFT data carries interference, a question no one has quantified. GPLSI gains a dialect-fidelity metric that its Valencian models currently lack. Positioned this way, the project fills a gap the main Catalan actors have reasons to adopt.
