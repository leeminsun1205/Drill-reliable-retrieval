> **Cập nhật mới nhất:** Tóm tắt dùng “Kết quả đánh giá cuối cùng”, nêu các thành phần hỏi đáp sau truy xuất chỉ phục vụ demo. Mục 4.4 và hướng phát triển đã bổ sung giả thuyết/hướng thử làm giàu tiêu đề–tóm tắt, đặc trưng đánh giá và lọc bằng chứng bằng LLM như FPT IS/EDM, kèm kiểm chứng và chi phí. Slides có thêm biểu đồ epoch cùng nhận xét ở trang 25, tổng 32 trang.

> **Report: rà soát mạch bài đã hoàn tất (06/11/2026):** PDF 15 trang,
> 10 bảng, 3 hình, 12 nguồn. Đã giải thích pipeline trước/sau ảnh, nối đầu
> vào–đầu ra từng bước, dẫn/nhận xét mọi bảng hình, thống nhất đoạn/điều
> luật/tập ứng viên và mã aid. Số liệu đã chốt không đổi; 22 hàng P/R/F2
> được kiểm tra, không cảnh báo LaTeX. Xem `final_report/review_notes.md`
> và README đã được hợp nhất, thay cho các ghi chú chừa chỗ cũ phía dưới.

> **Cập nhật mới nhất 06/11:** Theo số top-2 chủ đồ án vừa cung cấp, enhanced v2 epoch 1 dùng P/R/F2 = 0.426941/0.701696/0.621680; report/slides hiển thị 0.4269/0.7017/0.6217. Hình epoch so sánh top-2 ở cả hai đường, thay đường threshold trước đó. Ghi chú số cũ phía dưới là lịch sử. Nguồn chi tiết ở docs/experiments.md.

> **Mới nhất:** Đã điền đủ hai biểu đồ và hai ca phân tích lỗi vào report,
> vẫn 15 trang. Gói checkpoint-1234 mới được đối chiếu: ensemble F2=0.6910;
> 206 gold bỏ sót chia 60/97/49. Ba kiểm thử qua. Biểu đồ epoch dùng số đã
> ghi trước, không huấn luyện lại. Có chênh lệch cần chủ đồ án đối chiếu:
> LightGBM top-2 trong gói mới 0.6114, bảng cũ 0.6203. Đã hỏi, chưa tự đổi.
> Xem docs/evaluation.md. Các ghi chú chừa trống bên dưới đã hết hiệu lực.

> **Báo cáo — cập nhật theo phản hồi chủ đồ án (2026-11-06):**
> `final_report/main.pdf` hiện 15 trang, dùng đúng Times New Roman 12 pt
> (phông cài riêng ở `final_report/fonts/`, được Git bỏ qua). Bảng đặc trưng
> dùng 15/37/57/65 như slides; bỏ cột số lịch sử và các ghi chú biên tập
> khỏi bài. Provenance thí nghiệm vẫn giữ trong tài liệu kỹ thuật, không
> có lần huấn luyện mới. Mã nguồn đặt sau tóm tắt; pipeline dùng nguyên ảnh
> chủ đồ án; bảng 9 sao đủ 10 hàng trên slides đã được chủ đồ án xác minh.
> Đã chừa chỗ biểu đồ ở trang 12 và phân tích lỗi ở trang 14, nguồn nằm
> trong `final_report/sections/`. Cần gói complete-results của checkpoint
> 1234 để điền; gói local được kiểm tra thuộc checkpoint 1208 cũ. Xem
> `final_report/README.md` để biết chính xác tệp cần nhận. Bản nhóm gửi ban
> đầu vẫn ở `final_report/original/`.

# Project handoff for a new assistant

> **Metric correction resolved:** the owner confirmed the enhanced_v2 F2
> transcription error. The current reported P/R/F2 is **0.4223/0.7027/0.6203**,
> with F2 recomputed from reported P/R. All tables/slides were updated; no new
> model fitting or prediction evaluation was run. See docs/evaluation.md.

> **Latest code contract:** LightGBM now excludes four reference columns by
> default (two-dense enhanced_v2 = 57 inputs). Explicit legacy mode preserves
> 61-column artifact compatibility, including the demo. Historical metric
> records are unchanged and no new DRiLL fit was run. The earlier statement
> that code still always uses 61 columns is superseded. See docs/experiments.md.


> **Slide feature-count update (2026-11-05):** Display 57 for enhanced_v2,
> excluding four inactive reference-provenance columns. Display schema counts
> 15/37/57/65. Original artifacts/code remain 15/41/61/69; metrics are not
> recomputed. See the explicit counting note in tables_experiment.md R6.


> **Điều chỉnh riêng phần kết luận (2026-11-05):** Theo yêu cầu mới nhất,
> gộp ba slide kết luận, giới hạn và hướng phát triển thành một slide.
> Giữ các phần khác của bản gốc; tổng PDF còn 38 trang. Slide kết luận dùng
> sơ đồ bốn tầng, các mốc dev/OOF 0.5252 → 0.5967 → 0.6203 → 0.6910 và
> kết quả Codabench 0.6978/0.6825. Các mốc không phải ablation một biến.
> Khoảng cách phương pháp: chưa dùng tiêu đề/đặc trưng LLM như EDM và
> tăng cường dữ liệu LLM như FPT IS; không khẳng định đó là nguyên nhân đã
> được xác nhận của chênh lệch điểm. Các ghi chú khôi phục 40 trang bên dưới
> mô tả trạng thái trước thay đổi riêng này.


> **Trạng thái mới nhất (2026-11-05):** Chủ dự án bác bỏ toàn bộ bản thiết kế lại.
> Đã xóa thư mục slide mới và đổi `slides_old/` thành `slides/`, khôi phục nguyên
> bộ slide cũ 40 trang cùng nguồn LaTeX và tài nguyên. Các mô tả về bản 17/25/32
> trang bên dưới đã hết hiệu lực. Không tiếp tục dùng cấu trúc thiết kế lại.


**Last updated:** 2026-11-05  
**Repository:** Vietnamese legal retrieval on VLSP 2025 DRiLL

Read this page first when resuming work in a new GPT account. It is an
orientation and pointer document; the topic-specific files linked below remain
the source of truth. Read `AGENTS.md` and the relevant source-of-truth document
before changing code, methodology, or results.

## Project in one paragraph

This repository studies reliable retrieval of Vietnamese legal articles. The
current study evaluates representation, hybrid retrieval/reranking, signal
combination, and final-set selection for legal evidence retrieval. Reference
expansion was removed from the current project scope by the owner on 2026-11-05. It is a retrieval research project, not primarily a chatbot. The
normal retrieval and prediction unit is a legal article (`aid`); chunks are
used as an intermediate representation and are aggregated back to article
scores.

## Current practical pipeline

The current Codabench baseline uses:

1. Fixed token windows of 448 tokens with 64-token overlap; references off.
2. VnCoreNLP BM25, VietLegal-Harrier, and multilingual-E5-large, fused with
   WRRF (`k=10`; weights BM25/Harrier/E5 = `0.15/0.65/0.20`).
3. Top-50 candidate chunks, scored by the fixed-window pointwise-MIL
   cross-encoder: epoch 1, checkpoint `1234`, trained for three adaptation
   epochs with a five-epoch scheduler horizon and maximum sequence length 512.
4. LightGBM `enhanced_v2` (61 features).
5. Multiclass–Ridge final-set selection.

Owner-reported Codabench scores for this configuration are public Macro-F2
`0.6978` (submission `1027650`) and private Macro-F2 `0.6825` (submission
`1027647`). It is the current practical baseline because the owner prioritizes
private performance. This choice was made after observing private scores, so it
is an operational choice, not leakage-free scientific model selection. The
older checkpoint-1208 submission remains the public-score record at `0.7055`.
Do not use private results to tune further changes.

## Latest development feature-schema results

These are owner-reported matched fixed-top-2 development results for checkpoint
1234, using the same 438-query dev split, five query-grouped OOF folds, and
seed 42. The `enhanced_v2` values below were corrected by the owner on
2026-11-04; use these numbers in place of the earlier `0.4189/0.6907/0.6114`.

| Stage / schema | Features | Precision@2 | Recall@2 | Macro-F2@2 |
|---|---:|---:|---:|---:|
| Reranker only, epoch 1 | 0 | 0.4064 | 0.6759 | 0.5967 |
| LightGBM `base_v1` | 15 | 0.4201 | 0.7006 | 0.6180 |
| LightGBM `enhanced_v1` | 41 | 0.4189 | 0.6957 | 0.6145 |
| LightGBM `enhanced_v2` | 61 | 0.4223 | 0.7027 | 0.6203 |
| LightGBM `enhanced_v3` | 69 | 0.4224 | 0.6916 | 0.6134 |

The corrected schema table is in `tables_experiment.md`, `docs/experiments.md`,
and the slides. `enhanced_v2` leads Recall@2 and Macro-F2@2; `enhanced_v3`
leads Precision@2 by 0.0001.

The four-policy dev table currently reports fixed top-2 `0.4223/0.7027/0.6203`,
score gap `0.5689/0.7076/0.6747`, fold-safe stopping
`0.5243/0.7303/0.6771`, and Multiclass–Ridge
`0.5757/0.7274/0.6910` (Precision/Recall/Macro-F2). These use a shared saved
`enhanced_v2` ranking and query-grouped evaluation; do not substitute a
`base_v1` row into this table without recomputing all policies on that schema.
The source comparison is documented in `docs/experiments.md` and
`tables_experiment.md`.

## Current notebook workflows

- **Full practical submission:**
  `notebooks/submission/kaggle_pool50_fixed448_refoff_epoch1_ckpt1234_multiclass_ridge_submission.ipynb`.
  Rebuilds retrieval/reranking and refits LightGBM/policies from the attached
  epoch-1 training output and datasets; saves a full reusable results archive.
- **New `base_v1` follow-up:**
  `notebooks/submission/kaggle_pool50_fixed448_refoff_epoch1_ckpt1234_base_v1_submission.ipynb`.
  Uses the full-submission archive above as input, reuses saved retrieval and
  reranker artifacts, and refits `base_v1` LightGBM and the same policy cases.
  It does not reload dense models/checkpoints or rerun retrieval/reranking. This
  notebook has been created and syntax-checked locally; its Kaggle execution
  and Codabench scores are pending.
- **Schema ablation:**
  `notebooks/experiments/reranking/kaggle_lgbm_feature_schema_top2_ablation.ipynb`.
- **Four final policies:**
  `notebooks/experiments/decision_policy/kaggle_compare_four_final_policies_dev.ipynb`.

For the new `base_v1` notebook, attach the full archive named
`pool50_fixed448_refoff_epoch1_ckpt1234_r3_retuned_multiclass_ridge_submission_complete_results.tar.gz`,
the `WeDev-VLSP` train dataset, the `WeDev-VLSP-test` dataset, and Kaggle Secret
`GITHUB_TOKEN`. It needs no Harrier/E5 model attachment, checkpoint attachment,
or separate cache attachment because it consumes the saved pipeline output.

## Dataset and evaluation invariants

- Raw corpus rows are documents; flatten to article-level rows including
  `aid`, `law_id`, and `content_Article` before article retrieval.
- Training uses `qid`; the gold `relevant_laws` values are article IDs (`aid`).
- Submission rows use exactly `qid` and `relevant_laws`, covering each query
  once.
- Corpus and queries are separate inputs. Never change raw data in place.
- Keep comparisons on the same split/candidate set; tune on train/dev only.
- Public/private scores are benchmark reports, never development tuning data.
- Label owner-reported, artifact-verified, published, and illustrative values
  accurately. Do not invent or silently swap measurements.

## Repository map and sources of truth

- `AGENTS.md`: required workflow, safety, and scientific constraints.
- `docs/research_plan.md`: research questions and baseline progression.
- `docs/dataset.md`: schemas, IDs, and preprocessing.
- `docs/evaluation.md`: splits, metrics, and leakage rules.
- `docs/experiments.md`: experiment status, results, configs, and artifacts.
- `docs/execution.md`: local/Kaggle execution behavior.
- `docs/competition_review.md`: official task and other teams' methods.
- `tables_experiment.md`: nine report tables and slide mapping.
- `tables_experiment_catalog.md`: full 21-table catalog.
- `notebooks/README.md`: supported notebook inventory and inputs.

Reusable implementation belongs in `src/drill_ir/`; scripts are thin wrappers;
notebooks orchestrate experiments but should not become the only home for
reusable pipeline logic. Kaggle notebooks that import `drill_ir` must put
`REPO / 'src'` on `sys.path` before the import; this fixes a previously
encountered `ModuleNotFoundError`.

## Slides and editing notes

On 2026-11-05 the owner requested a systematic slide redesign: modern, readable,
no source code, content faithful to the implementation, and reference expansion
excluded from the current project presentation. The compiled deck is
`slides/drill_retrieval.pdf`; its local source is `slides/drill_retrieval.tex`.
The owner rejected appendix separation and shortened tables. The current revision
returns to the pre-edit source: 32 pages, no appendix, full original tables
interleaved with the relevant explanations and diagrams. Only eight standalone
section/stage dividers were removed. `slides_old/` preserves the complete local
pre-edit folder including the 40-page PDF and original LaTeX. Do not overwrite it.
See `docs/presentation_brief.md` for the current editorial contract.

Reference-expansion research entries elsewhere record historical experiments;
they are not instructions to include the component in the current slides or
report. Retain measured provenance when editing historical records.

Slide source files remain ignored by Git except PDFs; preserve that setting.
Compile from `slides/` with `latexmk -xelatex -interaction=nonstopmode
-halt-on-error -file-line-error -outdir=/tmp/drill-slide-check drill_retrieval.tex`.

The owner confirmed 10–15 minutes including a live QA demo. The audience
prioritizes core pipeline behavior, datasets, stage inputs/outputs, and practical
use. Target 8–10 minutes speaking plus 3–4 minutes demo; keep full tables in place and use diagrams to explain processing. Timing remains a rehearsal estimate.

## Recommended next steps

1. Run the `base_v1` follow-up notebook using the attached full checkpoint-1234
   submission archive.
2. Compare its public/private submissions with the current
   `enhanced_v2`/Multiclass–Ridge baseline; report the result without tuning on
   private feedback.
3. If a new schema/policy becomes the operational choice, update the baseline
   record, notebook inventory, result provenance, and slides together. Keep the
   existing measured results and selection caveats intact.
