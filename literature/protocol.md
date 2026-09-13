# the project's design notes

The review design for this stage's literature escalation. Written with the
`evidence-synthesis` skill, whose entry in the project's design notes had been conditional
since the pack was built: it applies "only if the related-work section has to
become a formal review with a screening log". this stage is that condition arriving.

the project's design notes has always described itself as "a targeted review,
not a systematic one. No PRISMA protocol, no screening log, no exhaustive
database coverage." That was an accurate label on a fifteen-source review whose
job was to establish that a gap existed. It is not adequate for a submitted
manuscript whose central positive claim is an absence: that no ECG prognosis
benchmark reports an acquisition-context arm. An absence claim is only as strong
as the search behind it, and until this stage the search behind it was about a dozen web
queries recorded as a list of phrases.

This protocol was written before this stage's searches ran. It is not registered, and
the reason is stated in section 9 rather than elided.

## 1. Review type, and what it may therefore claim

**Type.** Scoping review, conducted with rapid-review shortcuts.

**Reporting.** PRISMA-ScR for the review, PRISMA-S for the search. The search
record is `sources/search_record.md`, generated from the committed concept-block
specs; the screening log is the project's design notes.

A scoping review is the right type because the question is what exists and where
the gaps are, not whether an intervention works and how well. That choice
constrains what the review may conclude. It may report that a body of work does
or does not contain a given design. It may not pool an effect, rate certainty
with GRADE, or claim that the acquisition-context baseline is absent from the
literature as a whole. What it can support is narrower and checkable: absent
from the benchmarks and prognosis studies this search reached.

**No PRISMA flow diagram appears in the manuscript.** That is a deliberate
choice, recorded as a decision rather than an omission. The paper is an audit
with a registered comparison, not a systematic review, and a flow diagram in its
related-work section would invite reviewers to hold the whole manuscript to
systematic-review reporting standards it does not meet and does not claim. The
reconciling counts exist in the project's design notes for anyone who wants them.

## 2. Question and sub-questions

**Question.** Has the share of a deep prognostic model's discrimination that is
attributable to the decision to acquire the signal, rather than to the signal,
been quantified anywhere in medical machine learning?

**Sub-questions**, each mapped to one search block.

| ID | Sub-question | Block |
|---|---|---|
| SQ1 | Has informative presence been carried outside tabular records, and has the act of measuring been modelled as a predictor in its own right? | `b2_informative_presence.yaml` |
| SQ2 | Does the shortcut-learning literature anywhere run a metadata-only control arm rather than perturbing the signal? | `b1_acquisition_shortcut.yaml` |
| SQ3 | What comparator arms do ECG-AI prognosis studies report, and does any report a non-waveform arm on the same cohort and labels? | `b3_ecg_prognosis.yaml` |
| SQ4 | Does reporting or appraisal guidance for prediction models name a minimum comparator arm? | `b4_reporting_standards.yaml` |

SQ4 is new at this stage and is the one that changes the paper rather than the review.
The manuscript's surviving positive claim is a reporting recommendation, and a
reporting recommendation that does not engage the reporting-standards literature
is an opinion. SQ1 to SQ3 restate the three sub-questions the 2026-08-23 review
already carried, so this stage's search is a superset of it rather than a replacement.

## 3. Eligibility

Written so a second screener would apply them the same way. There was no second
screener; see section 9.

| Element | Include | Exclude |
|---|---|---|
| Population | Human clinical or routinely collected health data, any modality | Animal, in vitro, synthetic-only data with no clinical cohort |
| Index model | Any prognostic or diagnostic model fit on a recorded signal, image, waveform or health record | Anything with no predictive component |
| Comparator | Any reported comparator arm, and specifically any metadata-only, demographics-only, presence-only or process-only arm | none |
| Outcomes | Discrimination, calibration or clinical utility reported for a comparator arm; a methodological or bias-structure result about acquisition or presence; reporting guidance naming a comparator | Diagnostic accuracy where the label is contemporaneous with and defined by the recording |
| Designs | Primary studies, benchmark and dataset papers, methodological and simulation studies, systematic reviews, reporting guidelines | Editorials, commentary and correspondence without new content |
| Setting | Any, with emergency and acute care prioritised for SQ3 | none |
| Publication types | Peer-reviewed version of record. Preprints only as below | Anything unlocatable in an index or on a publisher page |
| Language | English, because no non-English search was run. A limitation, not a criterion | none |
| Dates | Per block: 2015 or 2016 to 2026 for the methodological blocks, 2018 to 2026 for shortcuts, 2019 to 2026 for ECG-AI, chosen to bracket the deep-learning literature | none |

**Preprints.** Eligible only when no peer-reviewed version exists and the work
is the best available evidence for the point it is cited for. Every preprint is
typed `preprint`, graded `tier_3`, labelled as unreviewed at the point it is
cited in the manuscript prose, and named in the limitations. A preprint whose
published version exists is not the thing to cite: the numbers move during
review. `sources/sources.json` carries preprints inherited from the 2026-08-23
review and every one is re-checked for a published version at this stage.

**The primary target of the search** is fixed here, before searching: a study,
in any modality, that fits a model on acquisition or presence metadata alone and
reports its discrimination against a signal model on the same cohort and the
same labels. Finding one would falsify the manuscript's novelty claim. Looking
for it is therefore the most important thing this search does, and stating it as
the primary target is what stops a null result being presented afterwards as
though absence had been the hypothesis all along.

## 4. Information sources

- **PubMed**, through E-utilities.
- **Europe PMC**, free REST.
- **Crossref**, for scripted supplementary searching and for every verification.
- **DataCite**, for arXiv DOIs, which Crossref does not hold.
- **Publisher platforms** reached with site-restricted web search rather than
  their own interfaces: SpringerLink and the Nature portfolio, ScienceDirect,
  IEEE Xplore. This is not equivalent to running each platform's own search and
  the search record says so.
- **arXiv**, deliberately and last, for work too recent to have been reviewed.
- **Backward citation chasing** from the informative-presence papers the earlier
  review already carried.

No Scopus and no Web of Science. Neither credential was available in the
session. This is the single largest coverage gap and it is recorded in section 9
rather than left for a reader to infer.

## 5. Selection process

- One screener. No dual independent screening, no measured agreement, no
  adjudication procedure, because there was one screener.
- Candidate discovery was fanned out across four concurrent AI agents, one per
  block, each instructed to verify every source against a publisher page or an
  index record and to report what it could not confirm rather than soften it.
  Screening decisions, eligibility judgments and the final inclusion of every
  source were made in one place against this protocol.
- Every full-text exclusion carries a specific reason in
  the project's design notes. "Did not meet inclusion criteria" is not a reason and
  the log contains none.
- A preprint and its published version are one study, merged at deduplication
  and counted once.

## 6. Extraction

For every included source: bibliographic metadata as structured fields in
`sources/sources.json`, the verification route, the peer-review status, the
tier, what the source shows, and which of the manuscript's claims it bears on
and in which direction. That last field is the one the review is actually for,
and it is why the related-work summary table has a column for it.

Any number quoted from a source is read from that source and carries a
`<!--lit:key-->` pointer in the manuscript, which the corresponding check resolves
against `sources/sources.json` and refuses if the key is unverified.

## 7. Appraisal

No formal risk-of-bias instrument is applied, and that is a real limitation
rather than a scoping-review exemption.

The reason it is not applied: the included studies span simulation studies,
bias-structure analyses, benchmark releases, reporting guidelines and prognostic
model developments. PROBAST+AI is the right tool for the last of those and the
wrong tool for the other four, and applying one generic checklist across all of
them would measure the wrong domains while producing a table that looks like
appraisal. What is done instead is the tier grading already in
`sources/sources.json`, which records how a source was verified and not how good
it is. The distinction is stated in the project's design notes and stays stated.

Where an included source is a preprint, that fact is carried to the point of
citation rather than to a table the reader has to cross-reference.

## 8. Synthesis

Narrative, structured by sub-question, reported against SWiM's applicable
elements. No meta-analysis: the studies answer four different questions in five
designs across at least four modalities, and pooling them would produce a
precise answer to no question. No GRADE, because GRADE rates certainty about an
effect and this review reports which designs exist.

The synthesis has to do one thing the earlier review did not, and it is why
SQ2's primary target is stated in section 3: it must present the competing
explanation for the manuscript's phenomenon at least as strongly as the
manuscript's own. the project's design notes already does this for [croon2025].
this stage extends it to whatever the new blocks surface.

## 9. Limitations of this protocol, stated before the searches ran

1. **Not registered, and the substitute is weaker than it looks.** PROSPERO
   does not accept scoping reviews and no OSF registration was created, so the
   timestamp evidence that would distinguish this protocol from a post-hoc
   description does not exist. What does exist is that this file was written
   before the searches ran and that it lands in the same commit as the sources
   it governs, never after them. That is an ordering claim git can falsify but
   not confirm: a single commit shows the two were written together, not which
   came first within the session. It is offered as exactly that much, and a
   reader who wants preregistration evidence should read this as having none.
2. **One screener.** No agreement statistic and no adjudication.
   Single-screener scoping reviews miss records, and the rate is unmeasured here.
3. **No Scopus, no Web of Science.** The two largest multidisciplinary indexes
   are absent. Coverage rests on PubMed, Europe PMC and Crossref, which
   under-cover computer-science conference proceedings in particular. For a
   question about deep-learning benchmark practice that is a coverage gap
   pointing in the worst possible direction, and it bounds the absence claim.
4. **English only**, with no justification beyond that no non-English search was
   run.
5. **No formal appraisal instrument**, for the reason in section 7.
6. **AI assistance throughout**, disclosed in the project's design notes and in
   the manuscript. Candidate discovery, drafting and the first pass of metadata
   extraction were AI-assisted. Every citation was confirmed against a publisher
   page, an index record or a registry in-session, and the two pinned tools in
   `scripts/tool_pins.json` re-derive that offline. Screening and eligibility
   judgments were made against this protocol in one place, and responsibility
   for them is not delegated to the tooling.
7. **The absence claim cannot be made stronger by searching more.** It can only
   be made weaker by someone finding the paper this search missed. The
   manuscript states the claim in the form that is checkable, which is that the
   baseline is missing from the benchmarks the field uses, and the monthly scoop
   watch in the project's design notes A7 remains the mitigation.

## 10. Deviations

Appended, never edited into the section they affect.

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-09-06 | 4 | Publisher platforms reached by site-restricted web search rather than their own interfaces | No institutional credential in the session. Recorded in Part B of the search record rather than presented as a platform-native search |
