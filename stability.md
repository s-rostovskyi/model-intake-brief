# Stability check: Qwen/Qwen3-8B

- Runs: 5 completed of 5
- Core checks with the same status in every run: 6 of 15

## Per run (core checks)

| Run | Evidence found | Partial | Not found | Not checked | Time |
|---|---|---|---|---|---|
| 1 | 6 | 7 | 2 | 0 | 10.1 s |
| 2 | 5 | 6 | 4 | 0 | 9.3 s |
| 3 | 4 | 11 | 0 | 0 | 8.7 s |
| 4 | 3 | 7 | 5 | 0 | 8.5 s |
| 5 | 5 | 7 | 3 | 0 | 8.8 s |

## Per check

| ID | Check | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Stable |
|---|---|---|---|---|---|---|---|
| ID-01 | Provider and model identity | ✓ | ✓ | ✓ | ✓ | ✓ | yes |
| ID-02 | Release date | ◐ | × | ◐ | × | ◐ | no: ◐ 3, × 2 |
| ID-03 | Base model dependencies | ◐ | ◐ | ◐ | × | × | no: ◐ 3, × 2 |
| DS-01 | Distribution and access | ◐ | ✓ | ◐ | ◐ | ✓ | no: ◐ 3, ✓ 2 |
| DS-02 | Licence or access terms | ◐ | ◐ | ◐ | ◐ | ◐ | yes |
| TP-01 | Architecture | ✓ | ✓ | ✓ | ✓ | ✓ | yes |
| TP-02 | Parameter size | ✓ | ✓ | ✓ | ✓ | ✓ | yes |
| TP-03 | Input and output modalities | ✓ | ✓ | ◐ | ◐ | ◐ | no: ◐ 3, ✓ 2 |
| TP-04 | Input and output limits | ◐ | ◐ | ◐ | ◐ | ◐ | yes |
| US-01 | Intended tasks and uses | ✓ | ◐ | ✓ | ◐ | ✓ | no: ✓ 3, ◐ 2 |
| US-02 | Restrictions on use | ✓ | × | ◐ | × | ◐ | no: × 2, ◐ 2, ✓ 1 |
| US-03 | Integration instructions | ◐ | ◐ | ◐ | ◐ | ◐ | yes |
| DT-01 | Training data types and provenance | ◐ | × | ◐ | × | ◐ | no: ◐ 3, × 2 |
| DT-02 | Training data curation | × | × | ◐ | × | × | no: × 4, ◐ 1 |
| DT-03 | Public training-content summary | × | ◐ | ◐ | ◐ | × | no: ◐ 3, × 2 |
| CT-01 | Training duration — Context (context) | × | × | × | × | × | yes |
| CT-02 | Training compute — Context (context) | × | × | × | × | × | yes |
| CT-03 | Training energy — Context (context) | × | × | × | × | × | yes |

Legend: ✓ evidence found, ◐ partial, × not found, – not checked.

## Where the difference comes from

- Pages read by Jina: 3 in every run, 3 in at least one run.
- The same pages were read every time, so any difference in status comes from the assessment step.

## Unstable core checks

- ID-02 Release date: ◐ / × / ◐ / × / ◐
- ID-03 Base model dependencies: ◐ / ◐ / ◐ / × / ×
- DS-01 Distribution and access: ◐ / ✓ / ◐ / ◐ / ✓
- TP-03 Input and output modalities: ✓ / ✓ / ◐ / ◐ / ◐
- US-01 Intended tasks and uses: ✓ / ◐ / ✓ / ◐ / ✓
- US-02 Restrictions on use: ✓ / × / ◐ / × / ◐
- DT-01 Training data types and provenance: ◐ / × / ◐ / × / ◐
- DT-02 Training data curation: × / × / ◐ / × / ×
- DT-03 Public training-content summary: × / ◐ / ◐ / ◐ / ×
