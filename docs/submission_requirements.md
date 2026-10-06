# Group 9 / Topic 27: examination submission

Authoritative course source: `4_Exam Requirements.pdf`, Deep Learning Final Exam Projects 2026–2027. The supplied bachelor thesis guides academic appearance only. Its topic, supervisors, approval page and thesis-specific front matter are not reused as project facts. The planning document guides the technical protocol; its suggested 12-minute overview is superseded by the exam's 3-minute overview.

| Course requirement | Deliverable / evidence |
|---|---|
| PDF named GroupID_ProjectID_Report.pdf | `reports/9_27_Report.pdf` |
| Repository named DL2026-GroupID-ProjectID | `nguyends23ba14219-code/DL2026-9-27` |
| 10–15 report pages excluding references/appendix | 13 main pages, numbered 1–13; cover and contents add 2, so 15 physical pages before references/appendix |
| Abstract: 150–200 words | 175 words, checked by report builder |
| Introduction and research question | Section I, approximately half a page following the abstract |
| Related work: 0.5–1 page | Section II, one page |
| Dataset URL/version, split/preprocessing/scripts | Section III and root `DATA.md` |
| Downloadable processed experiment data | Public release `processed-data.zip`: all 70,000 normalized images, labels, split, nine partitions, manifests and license; SHA-256 in `artifacts/release_manifest.json` |
| Demo link, if applicable | Local Streamlit demo with public model bundle; no hosted demo URL; this item is optional |
| Baseline, main method, comparison strategy | Section V: centralized CNN and manually implemented FedAvg |
| Setup 1: baseline vs main | Centralized versus FedAvg IID |
| Setup 2: main research | IID/mild/strong balanced label skew |
| Setup 3: analysis/robustness/ablation | Strong E1R30 vs E3R10; all configurations use three seeds |
| Results with interpretation | Section VII, tables/curves, optimization discussion and limits |
| Error cases and reasons | Section VIII, recorded test indices and class-level analysis |
| Conclusion and limitations | Section IX; separate conclusions, limitations and future directions |
| Approximately 5–10 references | Seven cited primary sources/documentation entries |
| Appendix: member contribution table | Six named members with IDs and user-confirmed final responsibilities; Nguyễn Duy Dũng is leader |
| Complete source code | `src/`, `main.py`, `experiments/`, `demo/`, configs and tests |
| Reproduction README | Install, prepare data, run/resume one configuration or all 15, analyze/audit and infer |
| Overview: exactly 4 slides / max 3 minutes | `reports/9_27_Overview.pptx`, 175-second script and notes |
| Q/A: 12 minutes, any member | `reports/defense_notes_vi.md` |

## Submission action

The course document sets the deadline at **08:00 on 7 October 2026**, through the course Google Classroom, submitted by the group leader. Nguyễn Duy Dũng should submit `9_27_Report.pdf` and the repository link using the course's actual Classroom form. The deadline's timezone is not stated in the source; the local course/user context is Vietnam time (UTC+7).

The report and repository are prepared. The repository and release assets are public and can be accessed without a GitHub account. Uploading to Google Classroom is a separate submission action and has not been performed here.

The group leader should attach the latest `9_27_Report.pdf` and include these links in the Classroom submission:

- Repository: https://github.com/nguyends23ba14219-code/DL2026-9-27
- Dataset documentation and official source: https://github.com/nguyends23ba14219-code/DL2026-9-27/blob/main/DATA.md
- Complete processed dataset: https://github.com/nguyends23ba14219-code/DL2026-9-27/releases/download/v1.0.0/processed-data.zip
- Optional local demo instructions: the README section "Kết quả và demo có sẵn". No hosted demo is claimed.

Use the English report and four-slide overview. `report_vi.pdf` and `project27-defense.pptx` in the original release are older supplemental material. Updates after the submission deadline are not considered by the course.

## Evidence and artifact lifecycle

The fifteen pre-existing official runs are retained. An audit recomputes metrics from saved predictions and verifies split/partition coverage and exposure budgets. Reporting edits do not create new experimental measurements. Each original manifest retains its original training source commit and environment.

`experiments/prepare_data.py` prepares official files and indices without training. `experiments/package_artifacts.py --release-dir PATH --output-dir outputs` packages recorded checkpoints and fails if any intermediate round is missing. After creating a new package, upload its archives and matching generated manifest/checksum file together. Existing published `v1.0.0` downloads are verified against their actual SHA-256 values in `artifacts/release_manifest.json`.

`experiments/package_processed_data.py --release-dir PATH` exports the complete normalized study dataset after verifying source MD5 checksums and comparing every regenerated split/partition array with the committed study. `experiments/fetch_artifacts.py --data` downloads this public archive, verifies SHA-256/size, and extracts it under `artifacts/processed-data/` without GitHub CLI or authentication.

`experiments/build_submission.py` derives result tables and numeric discussion from committed CSV/JSON outputs, generates print-readable distribution/recall figures and builds the PDF. It rejects abstract-length or PDF-pagination violations. Edit the report template or `members.json`, then rerun the builder to keep PDF and source consistent.
