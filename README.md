# Semantic Field Detection

**Course:** DS357 – Explainable AI (XAI)
**Author:** Manyam Jagadeeswar Reddy (Roll No: 23BDS033)
**Program:** B.Tech in Data Science and Artificial Intelligence
**Institution:** Indian Institute of Information Technology, Dharwad

---

## 1. Project Overview

This project presents a system for the automatic detection of the **semantic field** of an English sentence. Given an arbitrary input sentence, the system classifies it into one of five predefined domains: **Medical**, **Technology**, **Finance**, **Sports**, or **Food**.

The classification leverages the **Open English WordNet 2023** database as its primary knowledge source, exploiting synset relationships and semantic hierarchies to resolve word-level domain associations.

The central challenge addressed by this work is **polysemy** — the phenomenon whereby a single lexical item carries multiple distinct meanings. For instance, the word *"Apple"* may refer to a fruit (Food domain) or a multinational technology corporation (Technology domain). Similarly, *"bank"* admits 18 distinct synsets across noun and verb categories, spanning Finance, Geography, and other fields. Naive lookup-based approaches fail systematically on such cases. This project implements and contrasts two approaches — a baseline dictionary lookup and a context-aware sliding window algorithm — to demonstrate how contextual analysis resolves polysemous ambiguity. Additionally, the system incorporates an Explainable AI (XAI) component that outputs confidence scores and flags genuinely ambiguous inputs.

---

## 2. Research Foundation

The methodology of this project is grounded in three peer-reviewed publications, summarized below.

### 2.1 IJCAI 2021 — Word Sense Disambiguation Survey

**"Recent Trends in Word Sense Disambiguation: A Survey"**
Bevilacqua, Pasini, Raganato & Navigli
[PDF](https://www.ijcai.org/proceedings/2021/0593.pdf) | [Proceedings](https://www.ijcai.org/proceedings/2021/593)

This survey provides a comprehensive review of four decades of research in Word Sense Disambiguation (WSD). It demonstrates that older knowledge-based systems — which resolve word senses via direct dictionary lookup and most-frequent-sense heuristics — fail to account for sentential context and thus exhibit poor performance on polysemous inputs. The paper further establishes that modern supervised approaches leveraging contextual embeddings (e.g., BERT) surpass the long-standing "80% glass ceiling" in WSD accuracy by conditioning sense predictions on full-sentence context.

**Relevance to this project:** This survey provides the theoretical justification for moving beyond the baseline dictionary lookup (Approach 1) toward a context-aware methodology (Approach 2). The failure modes catalogued in this survey directly correspond to the failure cases observed in Approach 1.

### 2.2 NeurIPS 2022 — FIRE: Semantic Fields as Non-Linear Functions

**"FIRE: Semantic Field of Words Represented as Non-Linear Functions"**
Du & Tanaka-Ishii, University of Tokyo
[PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/f08223bc8d177df6807811c32f5acfed-Paper-Conference.pdf) | [OpenReview](https://openreview.net/forum?id=3AxaYRmJ2KY)

This work challenges the conventional representation of word meaning as a single point in a vector space. The authors argue that such point-based representations are fundamentally inadequate for polysemous words — a word with 18 senses cannot be faithfully represented by a single vector. FIRE proposes that each word be modeled as a **semantic field**, analogous to a physical field (e.g., electromagnetic), where the realized meaning of a word is a function of its interaction with surrounding words. The word *"Apple"* surrounded by *"screen"* and *"battery"* produces a different field response than *"Apple"* surrounded by *"fruit"* and *"meal"*. Notably, FIRE outperformed BERT on the task of predicting the number of senses a word possesses, validating that field-based representations capture polysemy more faithfully than point-based alternatives.

**Relevance to this project:** FIRE provides the mathematical foundation for the sliding window algorithm in Approach 2. Rather than assigning a domain based on a word in isolation, Approach 2 evaluates the aggregate semantic "pull" exerted by neighboring words within a context window — directly implementing the field-theoretic intuition introduced by this paper.

### 2.3 ACL 2023 — Uncertainty Estimation for WSD

**"Ambiguity Meets Uncertainty: Investigating Uncertainty Estimation for Word Sense Disambiguation"**
Zhu Liu & Ying Liu
[PDF](https://aclanthology.org/2023.findings-acl.245.pdf) | [ACL Anthology](https://aclanthology.org/2023.findings-acl.245/) | [arXiv](https://arxiv.org/abs/2305.13119)

This paper investigates a neglected dimension of WSD systems: the calibration of confidence estimates. The authors demonstrate that standard WSD models are frequently **overconfident**, assigning high probability to incorrect sense predictions rather than acknowledging genuine ambiguity. This finding reveals that raw probability outputs from disambiguation models are unreliable indicators of prediction quality without explicit uncertainty quantification.

**Relevance to this project:** This paper directly motivated the confidence scoring and ambiguity detection mechanism. Rather than always committing to a single domain classification — even when the evidence is divided — the system outputs a percentage-based confidence breakdown. When the top-scoring domain falls below a 60% confidence threshold, the result is explicitly flagged as **AMBIGUOUS**. For example, the sentence *"Apple is both a fruit and a technology company"* yields a 72.7% Food score, honestly reflecting the non-trivial presence of the Technology domain rather than concealing model uncertainty.

---

## 3. Methodology

The system implements two classification approaches within a single Jupyter Notebook, followed by an Explainable AI extension. This progression from a naive baseline to a context-aware system with uncertainty quantification mirrors the evolution described in the research literature.

### 3.1 Approach 1: Baseline Dictionary Lookup

The baseline approach performs direct keyword matching against a predefined domain dictionary. Each word in the input sentence is independently looked up, and the domain with the highest aggregate match count is selected.

**Limitations:** This approach is context-blind. It treats each word in isolation and defaults to the most common domain association, failing systematically on polysemous inputs. For example, it cannot distinguish whether *"Apple"* refers to the fruit or the technology company, nor can it resolve *"bank"* as a financial institution versus a riverbank.

### 3.2 Approach 2: Context-Aware Sliding Window

The second approach evaluates each word within its **sentential context** using a sliding window mechanism. For each candidate word, the algorithm examines the surrounding words within a fixed window and computes a weighted domain score based on the semantic field contributions of neighboring terms. This directly operationalizes the field-theoretic insight from the FIRE paper (NeurIPS 2022): the realized domain of a polysemous word is determined by the aggregate semantic pull of its local context.

The domain keywords are auto-expanded using the WordNet API, leveraging synset relationships (hypernymy, hyponymy, meronymy) to broaden domain coverage beyond manually curated seed lists.

### 3.3 Innovation: Explainable AI and Confidence Scoring

Building on the uncertainty estimation findings from ACL 2023, the system incorporates an **XAI tracer** that provides:

1. **Feature Importance Breakdown:** A percentage-based decomposition showing how much each detected domain contributed to the final classification. This allows the user to inspect *why* a particular domain was selected and which words drove the decision.

2. **Ambiguity Detection:** If the confidence score for the top-ranked domain falls below **60%**, the system flags the output as **AMBIGUOUS**, signaling that the sentence contains significant cross-domain semantic content and that the classification should be interpreted with caution.

This design ensures that the system does not produce overconfident predictions on genuinely ambiguous inputs — a failure mode that the ACL 2023 paper identifies as prevalent in conventional WSD systems.

---

## 4. Dataset

| Property       | Value     |
|----------------|-----------|
| **Dataset**    | Open English WordNet 2023 Edition |
| **Total Words**    | 161,338   |
| **Total Synsets**  | 120,135   |
| **Nouns**          | 123,612   |
| **Verbs**          | 11,615    |
| **Adjectives**     | 21,619    |
| **Adverbs**        | 4,481     |
| **Relations**      | 415,905   |

**Source:** [GitHub – Global WordNet](https://github.com/globalwordnet/english-wordnet) | [Princeton WordNet](https://wordnet.princeton.edu)

WordNet is a large-scale lexical database originally developed at Princeton University. Unlike conventional dictionaries, WordNet organizes English words into groups of cognitive synonyms called **synsets**, each representing a distinct concept. These synsets are interlinked through semantic relations — including **hypernymy** (is-a), **hyponymy** (type-of), **meronymy** (part-of), and **antonymy** (opposite-of) — forming a rich semantic graph.

This project utilizes the WordNet API for **auto-expansion** of domain keyword lists. Starting from manually curated seed terms for each domain, the system traverses synset relationships to automatically discover and include related terms, substantially increasing domain coverage without manual enumeration. This automated expansion is particularly important for achieving robust classification across diverse vocabulary.

---

## 5. Repository Structure

```
NLP-semantic-detection/
├── NLP_Semantic_detection.ipynb   # Complete implementation notebook
└── README.md                     # Project documentation
```

---

## 6. References

1. Bevilacqua, M., Pasini, T., Raganato, A., & Navigli, R. (2021). Recent Trends in Word Sense Disambiguation: A Survey. *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21)*, 4330–4338.

2. Du, Y., & Tanaka-Ishii, K. (2022). FIRE: Semantic Field of Words Represented as Non-Linear Functions. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*.

3. Liu, Z., & Liu, Y. (2023). Ambiguity Meets Uncertainty: Investigating Uncertainty Estimation for Word Sense Disambiguation. *Findings of the Association for Computational Linguistics: ACL 2023*, 3940–3951.

4. Fellbaum, C. (1998). *WordNet: An Electronic Lexical Database*. MIT Press.

5. McCrae, J. P., et al. (2023). Open English WordNet 2023 Edition. Global WordNet Association.

---

*Indian Institute of Information Technology, Dharwad — Department of Data Science and Artificial Intelligence*