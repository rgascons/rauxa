# Existing Catalan Evaluation Benchmarks (inventory as of Oct 2026)

Research date: 2026-10-11. About 20 tool calls. Primary sources: the lm-evaluation-harness catalan_bench README, the IberBench arXiv HTML, the La Leaderboard ACL 2025 paper (full text, Table 5), the CLUB arXiv paper (full text), the VeritasQA paper and HF card, and the Global-MMLU HF card. Items marked **[unverified]** come from background knowledge and were not confirmed with a source in this session.

---

## Q1. Master inventory table (Catalan evaluation datasets)

### Takeaway
About 35–40 distinct Catalan or Valencian evaluation sets exist. Most are BSC/Projecte AINA classification, QA and NLI sets (2021–2024). A large share are human translations of English benchmarks (ARC, PIQA, SIQA, OpenBookQA, COPA, XStoryCloze, MGSM, WNLI, XNLI, XQuAD, PAWS). Native Catalan generative, cultural, instruction-following and chat-style evaluation remains thin.

### Cited Findings

**Sizes, origin and #shots (verified):** these come from La Leaderboard Table 5, "Details of the evaluation datasets in Catalan" — [La Leaderboard, ACL 2025 / arXiv 2507.00999](https://arxiv.org/pdf/2507.00999). All are run through LM Evaluation Harness (La Leaderboard uses the fork [somosnlp/lm-evaluation-harness](https://github.com/somosnlp/lm-evaluation-harness)).

| Dataset (harness task id) | Maintainer | Year | Task | # eval items | Native / translated | Dialects | HF link | Frameworks |
|---|---|---|---|---|---|---|---|---|
| ARC_ca (`arc_ca_easy`, `arc_ca_challenge`) | Projecte AINA / BSC | 2024 (IberoBench) | Science MCQA | 869 (as listed by La Leaderboard; likely the Challenge test split) | Human translation from EN | Std. Catalan | [projecte-aina/arc_ca](https://huggingface.co/datasets/projecte-aina/arc_ca) | lm-eval catalan_bench, La Leaderboard, IberBench |
| Belebele Cat (`belebele_cat_Latn`) | Meta | 2023 | Reading comprehension MCQ | 900 | Human translation (FLORES passages) | Std. | [facebook/belebele](https://huggingface.co/datasets/facebook/belebele) | lm-eval, La Leaderboard, IberBench |
| caBREU (`cabreu`: extractive, abstractive, extreme) | AINA / BSC | 2024 (LREC-COLING) | Summarization | 301 | Native (news) | Std. | [projecte-aina/caBreu](https://huggingface.co/datasets/projecte-aina/caBreu) | lm-eval, La Leaderboard, IberBench |
| CatalanQA (`catalanqa`) | AINA / BSC | 2024 | Extractive QA | 2,135 | Native | Std. | [projecte-aina/catalanqa](https://huggingface.co/datasets/projecte-aina/catalanqa) | lm-eval, La Leaderboard |
| CatCoLA (`catcola`) | Bel et al. (UPF), hosted by nbel | 2024 | Linguistic acceptability | 1,020 | Native (from linguistics literature) | Std. | [nbel/CatCoLA](https://huggingface.co/datasets/nbel/CatCoLA) | lm-eval, La Leaderboard, IberBench |
| COPA_ca (`copa_ca`) | AINA / BSC | 2024 | Commonsense causal reasoning | 500 | Human translation from EN | Std. | [projecte-aina/COPA-ca](https://huggingface.co/datasets/projecte-aina/COPA-ca) | lm-eval, La Leaderboard, IberBench |
| CoQCat (`coqcat`) | AINA / BSC | 2024 | Conversational extractive QA | 8,986 | Native | Std. | [projecte-aina/CoQCat](https://huggingface.co/datasets/projecte-aina/CoQCat) | lm-eval, La Leaderboard |
| MGSM_ca (`mgsm_direct_ca`) | AINA / BSC | 2024 | Grade-school math | 250 | Human translation from EN | Std. | [projecte-aina/mgsm_ca](https://huggingface.co/datasets/projecte-aina/mgsm_ca) | lm-eval, La Leaderboard |
| OpenBookQA_ca (`openbookqa_ca`) | AINA / BSC | 2024 | Science MCQA | 500 | Human translation from EN | Std. | [projecte-aina/openbookqa_ca](https://huggingface.co/datasets/projecte-aina/openbookqa_ca) | lm-eval, La Leaderboard, IberBench |
| Parafraseja (`parafraseja`) | AINA / BSC | 2022 | Paraphrase detection | 21,984 | Native | Std. | [projecte-aina/Parafraseja](https://huggingface.co/datasets/projecte-aina/Parafraseja) | lm-eval, La Leaderboard, IberBench |
| PAWS_ca (`paws_ca`) | AINA / BSC | 2024 | Paraphrase detection (adversarial) | 2,000 | Human translation from EN (PAWS-X) | Std. | [projecte-aina/PAWS-ca](https://huggingface.co/datasets/projecte-aina/PAWS-ca) | lm-eval, La Leaderboard, IberBench |
| PIQA_ca (`piqa_ca`) | AINA / BSC | 2024 | Physical commonsense MCQA | 1,838 | Human translation from EN | Std. | [projecte-aina/piqa_ca](https://huggingface.co/datasets/projecte-aina/piqa_ca) | lm-eval, La Leaderboard |
| SIQA_ca (`siqa_ca`) | AINA / BSC | 2024 | Social commonsense MCQA | 1,954 | Human translation from EN | Std. | [projecte-aina/siqa_ca](https://huggingface.co/datasets/projecte-aina/siqa_ca) | lm-eval, La Leaderboard |
| TE-ca (`teca`) | AINA / BSC | 2021 | NLI (3-way) | 2,117 test (more than 20k total) | Native (Vilaweb plus the Catalan Textual Corpus, hypotheses written by annotators) | Std. | [projecte-aina/teca](https://huggingface.co/datasets/projecte-aina/teca) | lm-eval, La Leaderboard, IberBench, CLUB |
| WNLI_ca (`wnli_ca`) | AINA / BSC | 2021–24 | NLI / Winograd | 146 | Human translation from EN | Std. | [projecte-aina/wnli-ca](https://huggingface.co/datasets/projecte-aina/wnli-ca) | lm-eval, La Leaderboard |
| XNLI_ca (`xnli_ca`) | AINA / BSC | 2021–24 | NLI | 5,010 | Human translation from EN | Std. | [projecte-aina/xnli-ca](https://huggingface.co/datasets/projecte-aina/xnli-ca) | lm-eval, La Leaderboard, IberBench |
| XQuAD_ca (`xquad_ca`) | AINA / BSC | 2021 | Extractive QA | 1,190 (1,189 by the CLUB count) | Professional translation from EN | Std. | [projecte-aina/xquad-ca](https://huggingface.co/datasets/projecte-aina/xquad-ca) | lm-eval, La Leaderboard, CLUB |
| xStoryCloze_ca (`xstorycloze_ca`) | AINA / BSC | 2024 | Story completion | 1,510 | Human translation from EN | Std. | [projecte-aina/xstorycloze_ca](https://huggingface.co/datasets/projecte-aina/xstorycloze_ca) | lm-eval, La Leaderboard, IberBench |

**Other harness tasks without La Leaderboard sizes.** Source is the [catalan_bench README](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/catalan_bench/README.md) unless noted.

| Dataset (task id) | Maintainer | Task | Size | Origin | Dialect | Link | Notes |
|---|---|---|---|---|---|---|---|
| FLORES_ca (`flores_ca`, 16 directions: ca↔de/en/es/eu/fr/gl/it/pt) | Meta (FLORES-200) | MT | ~1,012 devtest sentences per direction **[unverified count]** | Human translation from EN | Std. | [facebook/flores](https://huggingface.co/datasets/facebook/flores) | Belebele reuses these passages |
| VeritasQA_ca (`veritasqa_gen_ca`, `veritasqa_mc1_ca`, `veritasqa_mc2_ca`) | BSC / AINA plus CiTIUS | Truthfulness (generation and MC) | 353 Q&A | Written fresh for es/ca/gl/en, avoiding country-specific or time-bound items | Std. | [projecte-aina/veritasQA](https://huggingface.co/datasets/projecte-aina/veritasQA), [code](https://github.com/langtech-bsc/veritasQA) | COLING 2025; CC BY-SA 4.0 ([ACL](https://aclanthology.org/2025.coling-main.366/), [HF card](https://huggingface.co/datasets/projecte-aina/veritasQA/blob/10a9d1b322d364293c68833056456b3c770c8d87/README.md)) |
| TerretaQA (`terretaqa`) | GPLSI (Univ. Alicante) | QA (Valencian / local knowledge, judging by the name) | not found | not stated | **Valencian** (inferred from org and name) | [gplsi/TerretaQA](https://huggingface.co/datasets/gplsi/TerretaQA) | Newer addition to catalan_bench |
| CIEACOVA (`cieacova`) | GPLSI | MC / generation from Valencian proficiency exams | not found | **Native** (official CIEACOVA Valencian proficiency exams) | **Valencian** | [gplsi/cieaCOVA](https://huggingface.co/datasets/gplsi/cieaCOVA) | Native exam-style set; candidate for contamination checks if the exams are public online |
| COCOTEROS_va (`cocoteros_va`) | GPLSI | Commonsense generation | not found | **Translated from Spanish** (COCOTEROS) | **Valencian** | [gplsi/cocoteros_va](https://huggingface.co/datasets/gplsi/cocoteros_va) | |
| TruthfulQA_va (`truthfulqa_va`) | GPLSI | Truthfulness | ~817 if complete **[unverified]** | Translated / adapted from EN | **Valencian** | [gplsi/truthfulqa_va](https://huggingface.co/datasets/gplsi/truthfulqa_va) | |
| XNLI_va (`xnli_va`) | GPLSI | NLI | not found | Translated (XNLI) | **Valencian** | [gplsi/xnli_va](https://huggingface.co/datasets/gplsi/xnli_va) | Cited as "Building a Data Infrastructure for a Mid-Resource Language: The Case of Valencian" |
| Phrases_ca / phrases_va (`phrases_ca`, tag `phrases_va`) | BSC / GPLSI | Catalan↔Valencian adaptation (variety conversion) | not found | not stated | **Valencian vs. central Catalan** | No source listed in the README | The only dialect-conversion task found; La Leaderboard lists "phrases" (Catalan, Spanish, Valencian) as donated |

**La Leaderboard also uses these Catalan tasks** ([arXiv 2507.00999](https://arxiv.org/pdf/2507.00999), Table 1):
- `crows_pairs_ca`, a stereotype/bias task (CrowS-Pairs Catalan).
- `catalonia_independence`, stance and text classification from the Catalonia Independence Corpus (Zotova et al., 2021), in Catalan and Spanish.
- Donated but not yet scored: `americasnlp_mt`, `truthfulqa` (es/ca/eu/gl) and `veritasqa` (es/ca/gl).

**CLUB, the original 2021 encoder-era benchmark** ([Rodriguez-Penagos et al., arXiv 2112.01894](https://arxiv.org/pdf/2112.01894)):

| Dataset | Task | Size | Origin | License / host |
|---|---|---|---|---|
| AnCora-ca POS (UD_Catalan-AnCora) | POS tagging | full UD treebank (~500k words **[unverified]**) | Native | CC-BY; [UD_Catalan-AnCora](https://github.com/UniversalDependencies/UD_Catalan-AnCora) |
| AnCora-ca-NER | NERC | not given in paper | Native (from AnCora) | CC-BY; [BSC-TeMU/ancora-ca-ner](https://huggingface.co/datasets/BSC-TeMU/ancora-ca-ner) |
| TECa / TE-ca | NLI | more than 20,000 pairs (83.57% inter-annotator agreement on 600) | Native | CC-BY; [Zenodo 4761458](https://doi.org/10.5281/zenodo.4761458) |
| TeCla | News topic classification | 153,265 articles, 30 categories (ACN newswire) | Native | [Zenodo 4627197](https://doi.org/10.5281/zenodo.4627197) |
| ViquiQuAD | Extractive QA (Viquipèdia) | 15,153 Q-A pairs, 3,111 contexts | Native | [Zenodo 4562344](https://doi.org/10.5281/zenodo.4562344) |
| VilaQuAD | Extractive QA (Vilaweb news) | 6,282 Q-A pairs | Native | [Zenodo 4562337](https://doi.org/10.5281/zenodo.4562337) |
| XQuAD-ca | Extractive QA | 1,189 Q, 240 contexts | Professional translation from EN | via HF projecte-aina |
| STS-ca | Semantic textual similarity (0–5) | more than 3,000 pairs | Native (CaText corpus) | [Zenodo 4529183](https://doi.org/10.5281/zenodo.4529183) |

The CLUB paper says its datasets carry permissive licences and are on Zenodo and HF ([arXiv 2112.01894](https://arxiv.org/pdf/2112.01894)). CatalanQA and CoQCat came later; the [LREC-COLING 2024 data-infrastructure paper](https://aclanthology.org/2024.lrec-main.231/) is cited for most AINA sets.

**IberBench ([Gonzalez et al., arXiv 2504.16921](https://arxiv.org/html/2504.16921)) includes 12 Catalan entries:**
- MultiStanceCat (stance, IberEval 2018)
- PAWS (ca)
- Parafraseja
- caBreu
- TE-ca
- OpenBookQA (ca)
- ARC (ca)
- CoLA (CatCoLA)
- COPA (ca)
- Belebele (ca)
- XNLI (ca)
- XStoryCloze (ca)

Of these, 4 are "industry-relevant" (MultiStanceCat, PAWS, Parafraseja, caBreu) and 8 are "fundamental." No Valencian-specific dataset is included. The leaderboard is at [huggingface.co/spaces/iberbench/leaderboard](https://huggingface.co/spaces/iberbench/leaderboard). Across all languages IberBench integrates 101 datasets in 22 task categories. The paper's per-language sample totals are ambiguous in the extracted table, so no reliable Catalan sample count was found.

**Other multilingual sets with Catalan:**

| Benchmark | Catalan? | Details |
|---|---|---|
| MultiBLiMP 1.0 (Jumelet et al.; TACL 2026 / EMNLP 2025 presentation) | **Yes** | 101 languages. Minimal pairs for subject–verb and subject–participle agreement in number, person and gender, generated automatically from Universal Dependencies plus UniMorph. The best Catalan model is Goldfish at 97.7%. [arXiv 2504.02768](https://arxiv.org/pdf/2504.02768), [TACL](https://aclanthology.org/2026.tacl-1.10), [HF jumelet/multiblimp](https://huggingface.co/datasets/jumelet/multiblimp). Catalan pair count not found. |
| Global-MMLU (Cohere) | **No** | Its 42 language codes do not include ca. [HF card](https://huggingface.co/datasets/CohereLabs/Global-MMLU) |
| MMLU-ProX | Not confirmed | 29 languages; the list was not retrieved. [arXiv 2503.10497](https://arxiv.org/html/2503.10497v1) |
| INCLUDE (44 languages, regional exams) | Not confirmed | Language list not retrieved |
| EuroEval | Not confirmed | The leaderboard covers about 18 European languages ([PyPI](https://pypi.org/project/EuroEval/)); the datasets page did not render, so Catalan is unconfirmed |
| SIB-200 (topic classification over FLORES-200 sentences) | Likely yes (`cat_Latn`) **[unverified]** | Derived from FLORES-200, which includes Catalan |
| Okapi translated ARC / HellaSwag / MMLU (Lai et al. 2023) | Believed to include ca **[unverified]** | ChatGPT machine translation; available as `arc_multilingual` / `hellaswag_multilingual` / `m_mmlu` in lm-eval |
| BLEnD / SemEval-2026 Task 7 (everyday cultural knowledge) | No evidence of Catalan | [SemEval-2026 T7](https://arxiv.org/pdf/2605.02601) |
| IroSvA irony detection extended to Catalan | Yes, machine translation with Gemini 2 Flash | Built for a compact-LLM study. [arXiv 2504.03312](https://arxiv.org/html/2504.03312v2) |

### Inferences
- Every catalan_bench task above exists in the current harness README. These are verified: arc_ca, belebele_cat_Latn, cabreu, catalanqa, catcola, copa_ca, coqcat, flores_ca, mgsm_direct_ca, openbookqa_ca, parafraseja, paws_ca, piqa_ca, siqa_ca, teca, wnli_ca, xnli_ca, xquad_ca, xstorycloze_ca and veritasqa_*_ca.
- Valencian additions from GPLSI (cieacova, cocoteros_va, terretaqa, truthfulqa_va, xnli_va, phrases_va) have been added since IberoBench (COLING 2025). The README lists EsCoLA under CatalanBench, which looks like a README artifact because EsCoLA is Spanish.
- IberoBench's Catalan portion (Baucells et al., COLING 2025, pp. 10491–10519) corresponds to the original catalan_bench task set. The paper's own Catalan table was not fetched.

### Gaps
- Exact sizes for the GPLSI Valencian sets, FLORES and MultiBLiMP-ca, and per-dataset licenses for most AINA sets. AINA sets are generally CC BY 4.0 / CC BY-SA per their HF cards **[unverified per set]**.
- Whether EuroEval, INCLUDE and MMLU-ProX include Catalan is unconfirmed.

---

## Q2. Leaderboards measuring Catalan

### Takeaway
No standalone Catalan-only LLM leaderboard was found. Catalan is scored inside two multi-language Iberian leaderboards: La Leaderboard and the IberBench leaderboard.

### Cited Findings
- **La Leaderboard** (ACL 2025, community-driven via SomosNLP plus partners) is described as the first open-source leaderboard for generative LLMs in the languages of Spain and Latin America. Its initial version had 66 datasets across ca/eu/gl/es varieties and 50 models. It includes 18 Catalan datasets (sizes and origin in Table 5 above), uses few few-shot examples, and reports energy use. — [arXiv 2507.00999](https://arxiv.org/pdf/2507.00999); [ACL](https://preview.aclanthology.org/setup/2025.acl-long.1561)
- **IberBench leaderboard** ranks models across tasks, languages and varieties; 23 LLMs from 100M to 14B parameters were evaluated. — [HF space](https://huggingface.co/spaces/iberbench/leaderboard); [arXiv 2504.16921](https://arxiv.org/html/2504.16921)
- The Projecte AINA HF org has Spaces, but no leaderboard Space was confirmed. — [projecte-aina spaces](https://huggingface.co/projecte-aina/spaces)
- The EuroLLM-9B technical report gives Catalan benchmark scores (51.54 on one pretrained Catalan column), showing model developers report Catalan results directly. — [arXiv 2506.04079](https://arxiv.org/pdf/2506.04079)

### Inferences
- Both leaderboards mostly reuse the same AINA/IberoBench pool, so they are not independent measurements.
- Neither Catalan in HELM nor Open LLM Leaderboard Catalan coverage was found; Catalan is very likely absent from both.

### Gaps
- Whether La Leaderboard has added the GPLSI Valencian tasks by late 2026.
- Whether BSC/ALIA publishes an official ALIA model eval dashboard.

---

## Q3. New 2025–2026 Catalan benchmarks (generation, culture, instruction-following, chat, judges)

### Takeaway
2025–2026 work added VeritasQA (truthfulness, native multilingual), the GPLSI Valencian sets, MultiBLiMP Catalan agreement pairs and a few MT-derived sets. No Catalan-specific instruction-following, chat/arena, LLM-as-judge or cultural-knowledge benchmark was found.

### Cited Findings
- VeritasQA (COLING 2025): 353 items for es/ca/gl/en, designed for transferability and replacing TruthfulQA. — [ACL](https://aclanthology.org/2025.coling-main.366/)
- Valencian additions in the harness: CIEACOVA (native exams), COCOTEROS_va (translated from Spanish), TruthfulQA_va, XNLI_va, TerretaQA, phrases_va. — [catalan_bench README](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/catalan_bench/README.md)
- A targeted search for AINA instruction-following, judge or cultural benchmarks returned only general English or multilingual judge benchmarks (RubricEval, IF-RewardBench, JuICE). A 2026 survey of multilingual LLM-as-judge notes Catalan only within a multi-language summarization evaluation (Kocmi et al., 2025). — [arXiv 2607.02235](https://arxiv.org/pdf/2607.02235)
- The 2025 European benchmarking survey gives a taxonomy of language-specific benchmarks, which is useful context. — [arXiv 2510.24450](https://arxiv.org/abs/2510.24450)
- Comparable efforts exist for neighbouring languages but not Catalan: ALBA for European Portuguese generative linguistics ([arXiv 2603.26516](https://arxiv.org/pdf/2603.26516)) and QFrBLiMP for Quebec French minimal pairs ([arXiv 2509.25664](https://arxiv.org/abs/2509.25664)).

### Inferences
- No native-Catalan open-ended generation quality, instruction-following (IFEval-style), MT-Bench/arena, or Catalan culture/history/law knowledge benchmark appears to be public as of Oct 2026. These look like genuine gaps, given two extended searches with no hits.

### Gaps
- An absence of search hits is not proof. AINA/ALIA could have internal or very recent sets (for example a Catalan IFEval or MMLU translation) not indexed.
- BSC/AINA HF dataset listings were not exhaustively crawled.

---

## Q4. Coverage patterns and contamination

### Takeaway
Classic NLU is well covered: NLI, paraphrase, extractive QA, and commonsense MCQ. Coverage is thin for math (250 translated MGSM items), summarization (1 native set), truthfulness, and bias (CrowS-Pairs-ca). Native knowledge exams, cultural knowledge, long-form generation, instruction following, dialogue, safety and dialect-sensitive evaluation are effectively missing. Among the 18 La Leaderboard Catalan sets, 12 are human translations of English benchmarks.

### Cited Findings
- **Origin split in La Leaderboard Table 5** ([arXiv 2507.00999](https://arxiv.org/pdf/2507.00999)):
  - Translated (12): ARC, Belebele, COPA, MGSM, OpenBookQA, PAWS, PIQA, SIQA, WNLI, XNLI, XQuAD, xStoryCloze.
  - Original (6): caBREU, CatalanQA, CatCoLA, CoQCat, Parafraseja, TE-ca.
- La Leaderboard names contamination as pending work: "Another pending task is to analyse potential contamination (Sainz et al., 2023)". It also prefers human-translated over MT datasets. — [arXiv 2507.00999](https://arxiv.org/pdf/2507.00999)
- VeritasQA was created partly because TruthfulQA is English-centric and dated. — [ACL](https://aclanthology.org/2025.coling-main.366/)
- Native linguistic competence is covered by CatCoLA (1,020 items) and MultiBLiMP (agreement only). — [La Leaderboard](https://arxiv.org/pdf/2507.00999); [MultiBLiMP](https://arxiv.org/pdf/2504.02768)

### Inferences
- **Contamination risk is high** for translated sets whose English sources (ARC, PIQA, SIQA, OpenBookQA, COPA, StoryCloze, GSM-derived MGSM, XNLI, XQuAD/SQuAD) have been public since 2016–2020. Models can solve them through cross-lingual memorisation of answers. FLORES and Belebele are also widely used in training and tuning. Native sets built from Wikipedia or news (ViquiQuAD, CatalanQA, CoQCat, TeCla, caBreu) come from text very likely included in Catalan pretraining corpora such as CaText and CommonCrawl. The questions or labels may still be unseen. AINA's own pretraining corpora and its benchmarks share sources (Vilaweb, ACN), which creates risk for BSC/ALIA models specifically.
- **Saturation:** MultiBLiMP Catalan sits at 97.7% for a small Goldfish model, so it has little headroom.
- **Dialects:** Only Valencian is targeted (GPLSI). No Balearic or North-Western sets were found; standard Central Catalan dominates.
- **Format skew:** Most tasks are log-likelihood MCQ or classification. Generative metrics rely on BLEU, ROUGE or F1 (caBREU, FLORES, CatalanQA), and no LLM-judge rubric exists for Catalan fluency or grammar.

### Gaps
- No published contamination analysis specific to Catalan benchmarks was found.
- No quantitative dialect breakdown exists for any of these sets.
