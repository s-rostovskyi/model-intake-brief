# Model Intake Brief — public evidence checklist v0.1

**Information, not legal advice.** This is a scoped intake checklist. It records what a reviewer can substantiate from public documentation for a specified model version. It does not determine legal applicability or compliance.

## Reference material

- [AI Act, consolidated text](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX%3A02024R1689-20260727), Article 53 and Annexes XI and XII.
- [European Commission GPAI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai), Transparency chapter, Measures 1.1 and 1.2.
- [Model Documentation Form, readable PDF copy](https://www.editionmultimedia.fr/wp-content/uploads/2025/07/Code-of-Practice-for-GPAI-Transparency-10-07-25.pdf).

## Checks

The Form section and field labels below locate the source material. The acceptance criteria are specific to this intake project. A field marked **Context** may be useful internally but should not automatically produce a provider request. The full form contains additional fields and recipient-specific detail beyond this MVP.

| ID | Check | Evidence sufficient for this intake | Reference |
|---|---|---|---|
| ID-01 | Provider and model identity | Official documentation identifies the provider and the exact model or version. A family name alone is partial. | Form, General information |
| ID-02 | Release date | Provider identifies when the target model first became available. A repository modification date alone is insufficient. | Annex XII §1(c), Form, General information |
| ID-03 | Base model dependencies | Exact upstream model or checkpoint is named, or the provider explicitly states that the model is not a modification or fine-tune of a prior model. | Form, General information |
| DS-01 | Distribution and access | Official repository, API documentation or equivalent identifies how the exact model is accessed and whether access is to weights or a hosted service. | Annex XII §1(c), Form, Methods of distribution and licenses |
| DS-02 | Licence or access terms | Applicable licence text or service terms are linked from official documentation and can be tied to this distribution channel. | Annex XII §1(h), Form, Methods of distribution and licenses |
| TP-01 | Architecture | Architecture type is documented for the target model. A family description covering several architectures is partial without a mapping. | Annex XII §1(f), Form, Model properties |
| TP-02 | Parameter size | Total parameter count or a clearly documented size range for the target model. Active parameter count alone is partial. | Annex XII §1(f), Form, Model properties |
| TP-03 | Input and output modalities | Provider states supported input and output types for the target model. | Annex XII §1(g), Form, Model properties |
| TP-04 | Input and output limits | Official documentation gives relevant limits and their units. A single shared context limit is partial if the output limit is unspecified. | Annex XII §2(b), Form, Model properties |
| US-01 | Intended tasks and uses | Provider describes representative uses or gives integration examples explicitly associated with the target model. | Annex XII §1(a), Form, Intended uses |
| US-02 | Restrictions on use | Applicable restrictions or acceptable-use terms are linked and their relationship to this model or service is documented. | Annex XII §1(b), Form, Intended uses |
| US-03 | Integration instructions | Official instructions identify a usable API call or local setup and its relevant prerequisites. | Annex XII §2(a) |
| DT-01 | Training data types and provenance | Provider identifies the data types and source categories for the model or a family explicitly documented as sharing them. | Annex XII §2(c), Form, Data |
| DT-02 | Training data curation | Model-specific documentation explains selection, filtering or other relevant curation methods. Family-level evidence without explicit applicability is partial. | Annex XII §2(c), Form, Data |
| DT-03 | Public training-content summary | A provider-published public summary explicitly covers the target model. A technical report may support other data checks and should be distinguished from a published summary. | Article 53(1)(d), separate AI Office template |
| CT-01 | Training duration — Context | Provider gives a duration or documented duration range and identifies the training phase covered. | Annex XI §1(2)(d), Form, Computational resources |
| CT-02 | Training compute — Context | Provider gives a model-specific compute amount or range with units and phase. Hardware type alone is partial. | Annex XI §1(2)(d), Form, Computational resources |
| CT-03 | Training energy — Context | Provider gives a model-specific energy amount with units and scope. A documented reason why it cannot estimate energy is recorded separately as partial evidence. | Annex XI §1(2)(e), Form, Energy consumption |

## Status and report contract

- **Evidence found** — an attributable excerpt satisfies the row criterion and applies to the target. Record URL, section, retrieval date and link-check outcome.
- **Partial** — an attributable excerpt supports part of the criterion. Record the exact gap.
- **Not found** — the configured searches completed without supporting evidence. Show which sources were checked.
- **Not checked** — an essential source or tool failed or was blocked before the check could be completed. Show the reason.

The English Markdown report includes the target and run date, an executive summary, findings, prioritised provider questions, and method and limitations. Include a short excerpt for positive findings and verify it against retrieved text. Check whether cited links resolve publicly and flag blocked or unverified destinations. Keep operational provider questions separate from regulator-facing context. The same checklist version is used for both sample reports.
