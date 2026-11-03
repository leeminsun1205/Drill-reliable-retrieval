# Brief nội dung slide trình bày đồ án

Tài liệu này là nguồn thông tin nền để nhóm thiết kế slide thống nhất storyline,
số liệu và cách diễn giải. Đây không phải bản slide hoàn chỉnh. Khi rút gọn nội
dung, cần giữ các cảnh báo về split, nguồn kết quả và mức độ kiểm chứng.

## 1. Thông điệp trung tâm

Đồ án nghiên cứu bài toán truy hồi điều luật tiếng Việt, không phải chatbot hay
hệ thống sinh câu trả lời. Mục tiêu là xác định liệu cấu trúc pháp luật
Điều–Khoản–Điểm và các tham chiếu giữa điều khoản có giúp tìm đúng, đủ các điều
luật liên quan hơn một baseline hybrid retrieval kết hợp cross-encoder reranking
hay không.

Đến hiện tại, đồ án đã xây dựng được pipeline tái lập từ dữ liệu đến submission,
đã có baseline BM25, dense, hybrid, reranker, fine-tuning và LightGBM fusion.
Kết quả có kiểm soát cho thấy fixed token windows 448/64 ổn định hơn biểu diễn
Khoản–Điểm, còn reference expansion tăng candidate recall nhưng không tăng
final F2. Đây là negative result có giá trị: đưa cấu trúc pháp lý vào pipeline
không tự động cải thiện retrieval. Cấu hình cuối vì vậy dùng fixed windows và
references off.

## 2. Storyline đề xuất

Mạch trình bày nên đi theo thứ tự:

1. Văn bản pháp luật dài và có cấu trúc; một câu hỏi có thể cần nhiều điều luật.
2. Truy hồi nguyên article bằng từ khóa hoặc cắt cụt văn bản chưa đủ tốt.
3. Xây dựng baseline từ BM25, dense retrieval, hybrid và reranking.
4. So sánh article, Khoản, Khoản–Điểm và fixed windows có kiểm soát; fixed
   448/64 thắng tổng thể.
5. Fine-tune reranker bằng article-level MIL trên hierarchical chunk bags, rồi
   giữ checkpoint cố định để đo khả năng transfer sang fixed windows.
6. Kết hợp các tín hiệu bằng LightGBM để tạo hệ thống hiện tại.
7. Trình bày ablation cho structure/reference và giới hạn train–inference
   representation mismatch.

Không nên kể đồ án như một chuỗi thử model rời rạc. Trọng tâm khoa học là ảnh
hưởng của biểu diễn cấu trúc và, trong giai đoạn tiếp theo, tham chiếu chéo.

## 3. Dàn ý slide

### Slide 1 — Tiêu đề

- Tên đề tài.
- Sinh viên, giảng viên hướng dẫn, khoa/trường và mốc báo cáo.
- Mô tả một câu: “Truy hồi điều luật tiếng Việt có khai thác cấu trúc
  Khoản–Điểm và tham chiếu pháp luật”.

### Slide 2 — Bối cảnh và động lực

- Văn bản pháp luật dài, có cấu trúc phân cấp và nhiều dẫn chiếu.
- Câu hỏi và điều luật đúng có thể không dùng cùng từ ngữ.
- Một câu hỏi có thể cần nhiều điều luật làm bằng chứng.
- Nếu retrieval bỏ sót điều luật thì các bước hỏi đáp hoặc giải thích phía sau
  không thể phục hồi thông tin đó.

Nên minh họa bằng một câu hỏi và một đoạn Điều–Khoản–Điểm thực tế. Ví dụ chỉ nên
dùng để giải thích cấu trúc, không tự gán nhãn đúng/sai nếu chưa đối chiếu dữ liệu.

### Slide 3 — Phát biểu bài toán

Đầu vào:

- câu hỏi pháp luật tiếng Việt `q`;
- corpus gồm các article, mỗi article có định danh `aid` và thuộc một văn bản
  `law_id`.

Đầu ra là danh sách hoặc tập `aid` liên quan. Một câu hỏi có thể có nhiều `aid`
đúng. Đơn vị retrieval chính là article, dù mô hình có thể dùng chunk làm đơn vị
biểu diễn và sau đó tổng hợp score về article.

### Slide 4 — Dữ liệu DRiLL

- 59.628 article không rỗng được lập chỉ mục.
- 2.190 câu hỏi train có nhãn.
- 312 câu public test và 627 câu private test theo số liệu ban tổ chức.
- Số điều luật đúng trung bình xấp xỉ 1,3 cho mỗi câu hỏi.
- Split nội bộ hiện tại: 1.752 train và 438 dev, seed 42.
- Corpus và query là hai tập riêng; ground truth nằm trong `relevant_laws`.

Vấn đề độ dài:

- với tokenizer `intfloat/multilingual-e5-base`, 17,03% article không vừa giới
  hạn 512 token;
- article dài nhất đo được là 68.385 token;
- không có query train nào vượt 512 token trong audit này.

Thông điệp cần rút ra: cắt cụt article có nguy cơ bỏ mất evidence.

### Slide 5 — Câu hỏi nghiên cứu

- **RQ1:** Khi giữ nguyên retrieval và reranking, mô hình hóa cấu trúc và tham
  chiếu pháp luật có cải thiện hiệu quả retrieval không?
- **RQ2:** Lợi ích có tập trung ở câu hỏi nhiều đáp án hoặc phụ thuộc dẫn chiếu
  không?
- **RQ3:** Reference expansion làm recall tăng và precision giảm như thế nào;
  ảnh hưởng ròng lên macro-F2 là bao nhiêu?

Hiện đồ án mới có kết quả đáng kể cho phần cấu trúc. Chưa được trình bày phần
tham chiếu chéo như một đóng góp đã hoàn thành.

### Slide 6 — Metric và giao thức đánh giá

- Metric chính: macro Precision, macro Recall và F2 tính từ hai giá trị macro.
- Metric chẩn đoán candidate stage: Recall@1/5/10/20/50/100.
- F2 ưu tiên recall hơn precision, phù hợp với yêu cầu hạn chế bỏ sót evidence.
- Mọi phương pháp so sánh cần dùng cùng split và seed.
- Reranker phải được so sánh trên cùng candidate pool.
- Dev dùng để chọn cấu hình; private test không được dùng làm dev set.

Không nhầm macro-F2 chính thức với trung bình F2 theo từng query.

### Slide 7 — Kiến trúc tổng thể

Sơ đồ đề xuất:

```text
Corpus
  ↓
Chuẩn hóa + biểu diễn Article/Khoản/Điểm
  ↓
BM25 ──────────────┐
                   ├─ Weighted RRF → top fused chunks
Dense retrieval ───┘
                              ↓
                    Cross-encoder reranker
                              ↓
                      max score theo aid
                              ↓
                 LightGBM hoặc threshold policy
                              ↓
                    danh sách điều luật cuối
```

Nhấn mạnh các stage chuẩn bị, retrieval, fusion, reranking, evaluation và
analysis được tách riêng.

### Slide 8 — Baseline BM25

| Tokenizer | Precision | Recall | F2 | Recall@100 |
|---|---:|---:|---:|---:|
| Regex/NFKC | 0,1804 | 0,4526 | 0,3477 | 0,8442 |
| PyVi | 0,1963 | 0,4906 | 0,3775 | **0,8786** |
| VnCoreNLP | **0,1963** | **0,4958** | **0,3799** | 0,8775 |

Các run sử dụng cùng dev 438 query, BM25 `k1=1.2`, `b=0.75`, candidate top 100
và final top 3. VnCoreNLP được chọn cho F2 cuối; PyVi vẫn là lựa chọn nhẹ có
candidate recall tốt.

### Slide 9 — Tác động của chunking

| Biểu diễn E5 | F2 | Recall@100 |
|---|---:|---:|
| Article truncate | 0,3716 | 0,8798 |
| Token window 512/64 | **0,4012** | 0,9047 |
| Recursive token 512/64 | 0,3950 | **0,9054** |

Kết luận:

- fixed window cải thiện F2 0,0296 tuyệt đối so với cắt cụt article;
- recursive separator không tự động cải thiện final F2;
- fixed window là non-structural control phù hợp cho thí nghiệm cấu trúc.

### Slide 10 — Biểu diễn cấu trúc pháp luật

Minh họa phân cấp:

```text
Điều 5. Điều kiện ...
  1. Người đăng ký phải:
     a) Có đủ năng lực ...
     b) Có giấy chứng nhận ...
```

Các treatment chính:

- Khoản: giữ mỗi Khoản làm một representation;
- Điểm: ghép phần dẫn của Khoản với từng Điểm để không mất ngữ cảnh;
- Khoản–Điểm: giữ Khoản ngắn nguyên vẹn, chỉ tách Điểm khi Khoản vượt budget;
- trường hợp không nhận diện chắc chắn được cấu trúc sẽ fallback về token window,
  không loại bỏ văn bản.

### Slide 11 — Kết quả structure-aware retrieval

| Biểu diễn E5 | Precision | Recall | F2 | Recall@100 |
|---|---:|---:|---:|---:|
| Token window | 0,2070 | 0,5242 | 0,4012 | 0,9047 |
| Khoản | 0,2215 | 0,5572 | 0,4275 | 0,9013 |
| Điểm + phần dẫn Khoản | **0,2306** | **0,5828** | **0,4464** | **0,9116** |

Diễn giải thận trọng:

- Khoản cải thiện early ranking và final F2 nhưng không cải thiện deep recall;
- Điểm cải thiện F2 và candidate recall ở mọi cutoff đã ghi nhận;
- đây là bằng chứng ban đầu rằng cấu trúc pháp luật có ích trong setup này;
- vẫn cần paired per-query analysis và kiểm tra lại artifact Kaggle.

### Slide 12 — Hybrid retrieval

- BM25 bổ sung tín hiệu trùng khớp từ vựng.
- Dense retrieval bổ sung tương đồng ngữ nghĩa.
- Weighted Reciprocal Rank Fusion kết hợp thứ hạng, tránh cộng trực tiếp hai loại
  raw score khác thang đo.

Các mốc tiêu biểu:

| Cấu hình | F2 dev |
|---|---:|
| BGE-M3 Khoản 512/64 | 0,4595 |
| Hybrid, sparse/dense 0,4/0,6 | 0,4622 |
| Hybrid, sparse/dense 0,2/0,8 | 0,4781 |
| Hybrid 0,2/0,8, RRF k=20 | **0,4820** |

Kết luận: BM25 có ích như tín hiệu bổ sung trọng số thấp; dense vẫn là nhánh
chính. Một số attribution BGE-M3 cũ chưa được xác nhận bằng artifact, vì vậy
không nên dùng chúng để kết luận treatment structure tốt hơn control dưới cùng
encoder.

### Slide 13 — Cross-encoder reranking

Hai candidate flow đã được nghiên cứu:

- `article_rechunk`: lấy article từ first stage rồi chia lại để rerank;
- `retrieved_chunks`: rerank trực tiếp những chunk đã tạo nên candidate pool.

Direct-chunk reranking nhất quán hơn giữa retrieval và reranking, đồng thời giới
hạn ở 50 query–chunk pairs cho mỗi query.

Các mốc zero-shot của pipeline Khoản–Điểm 448/64:

- fixed top-2 F2: 0,5050;
- threshold + fallback 2 F2: 0,5557.

Threshold được chọn và đo trên cùng dev nên đây là kết quả model-selection,
không phải ước lượng test không thiên lệch.

### Slide 14 — Fine-tuning reranker

Quy trình chính:

- mining và training chỉ dùng 1.752 query phía train;
- không đưa gold article vào negative pool;
- positive chunk được chọn bên trong gold article nên vẫn là weak supervision;
- negatives là các fused retrieved chunks khó hoặc bán khó;
- huấn luyện pointwise BCE;
- checkpoint chỉ được chọn theo 438 query dev.

| Model | Precision | Recall | F2 dev |
|---|---:|---:|---:|
| Zero-shot | 0,3586 | 0,6442 | 0,5557 |
| Epoch 1 | 0,4678 | 0,6327 | 0,5910 |
| Epoch 2 | 0,4493 | 0,6504 | 0,5970 |
| Epoch 3 | **0,4871** | 0,6483 | **0,6081** |
| Epoch 4 | 0,4752 | **0,6510** | 0,6061 |

Epoch 3 được chọn theo F2 dev. Epoch 4 có fixed-top-2 tốt hơn nên chỉ nên kết
luận threshold-policy đạt plateau, không kết luận mô hình overfit toàn diện.

### Slide 15 — LightGBM feature fusion

LightGBM kết hợp các tín hiệu đã tính sẵn:

- reranker score và score margin;
- BM25 score/rank;
- dense score/rank;
- chunk fusion score;
- đặc trưng độ dài query/chunk.

Đánh giá dùng five-fold query-grouped out-of-fold prediction:

| Hệ thống | Policy | Precision | Recall | F2 dev |
|---|---|---:|---:|---:|
| Epoch 3 | threshold + fallback 2 | 0,4871 | 0,6483 | 0,6081 |
| LightGBM OOF | threshold + fallback 2 | 0,4841 | **0,6833** | **0,6313** |
| Epoch 3 | fixed top 2 | 0,3653 | 0,6150 | 0,5410 |
| LightGBM OOF | fixed top 2 | **0,3756** | **0,6298** | **0,5547** |

OOF làm giảm nguy cơ fit và đánh giá cùng candidate, nhưng threshold gộp vẫn
được chọn trên 438 dev query nên kết quả vẫn là model-selection diagnostic.

### Slide 16 — Kết quả Codabench đã ghi nhận

| Hệ thống | Split | F2 |
|---|---|---:|
| Epoch 3, threshold chọn trên dev | Private | 0,5589 |
| LightGBM, fixed top 2 | Private | 0,5397 |
| LightGBM, frozen OOF threshold | Private | **0,5882** |
| LightGBM, frozen OOF threshold | Public | 0,6160 |

Có thể đưa organizer two-stage baseline private F2 0,5515 làm mốc tham khảo,
nhưng phải ghi rõ đó là số công bố trên private test, không phải một run cùng
split hay cùng giao thức với kết quả dev của dự án.

Các giá trị Codabench trên là kết quả do chủ dự án báo cáo. Checkout hiện tại
chưa chứa đủ provenance, prediction, ZIP và submission record để kiểm chứng độc
lập. Không được trình bày chúng như số đã audit artifact hoàn chỉnh.

### Slide 17 — Đóng góp hiện tại

**Về kỹ thuật**

- pipeline tái lập từ dữ liệu, retrieval, reranking, evaluation đến submission;
- hỗ trợ BM25, dense, hybrid, multi-dense, cross-encoder và learned fusion;
- cache embedding có fingerprint và hỗ trợ CPU, single-GPU, multi-GPU.

**Về phương pháp**

- parser Khoản–Điểm bảo thủ với fallback không làm mất dữ liệu;
- giữ phần dẫn Khoản trong representation của Điểm;
- direct retrieved-chunk reranking;
- train-only hard-negative mining và pointwise BCE.

**Về thực nghiệm**

- định lượng tác động của truncation và chunking;
- có bằng chứng ban đầu cho lợi ích của cấu trúc Khoản–Điểm;
- tạo baseline mạnh để đánh giá reference-aware retrieval tiếp theo.

### Slide 18 — Hạn chế

- Chưa triển khai cross-reference-aware retrieval.
- Chưa có ma trận đầy đủ: baseline, +structure, +references,
  +structure+references.
- Nhiều run Kaggle mới có số liệu được ghi nhận, chưa có artifact cục bộ để audit.
- Threshold selection trên dev có thể tạo optimism.
- Chưa có paired per-query analysis hoặc kiểm định thống kê.
- Ground truth ở article level; positive chunk chỉ là weak supervision.
- Parser đã audit coverage nhưng chưa có precision/recall thủ công hoàn chỉnh.
- Chưa phân tích theo domain vì dữ liệu không có mapping domain chính thức.
- RM3 không cải thiện profile đã hoàn thành; RM3 lớn hơn và Rocchio chưa có kết
  quả dùng được trong giới hạn runtime Kaggle.

### Slide 19 — Kế hoạch tiếp theo

1. Thu hồi và kiểm tra đầy đủ artifact của các run đã báo cáo.
2. Chạy paired analysis cho token window, Khoản, Điểm và Khoản–Điểm.
3. Trích xuất deterministic các tham chiếu như Điều X, Khoản Y Điều Z.
4. Lưu edge cùng nguồn văn bản, trạng thái parse và trường hợp mơ hồ.
5. Chạy bốn ablation có kiểm soát:
   - baseline;
   - baseline + structure;
   - baseline + references;
   - baseline + structure + references.
6. Phân tích riêng query có nhiều gold article và query phụ thuộc dẫn chiếu.
7. Chốt config, seed, split, checkpoint và tái chạy pipeline cuối.

### Slide 20 — Kết luận

Ba câu có thể dùng để kết bài:

1. Đồ án đã xây dựng được pipeline truy hồi pháp luật tái lập và một baseline
   hybrid–reranker tương đối mạnh.
2. Fixed windows 448/64 đạt kết quả tổng thể tốt hơn Khoản–Điểm; mô hình hóa cấu
   trúc pháp lý không tự động tạo gain và là một negative result quan trọng.
3. Reference expansion tăng candidate recall nhưng giảm nhẹ final F2; cấu hình
   cuối tắt references và dùng learned final-set selection.

## 4. Pipeline mạnh nhất hiện tại

Có thể dùng sơ đồ này trong slide tổng kết kỹ thuật:

```text
VnCoreNLP BM25 + VietLegal-Harrier + multilingual-E5-large
       ↓
Fixed token windows 448/64 + weighted RRF 0.15/0.65/0.20, k=10
       ↓
Top 50 fused retrieved chunks
       ↓
Vietnamese reranker, fixed-window pointwise MIL epoch 1 / checkpoint 1234
       ↓
Max aggregation theo aid
       ↓
LightGBM enhanced_v2 (61 features)
       ↓
Balanced multiclass / multiclass–Ridge final-set selection
```

Baseline thực nghiệm hiện tại dùng fixed-window 448/64, references off, WRRF
`0.15/0.65/0.20`, checkpoint epoch 1 mới `1234`, LightGBM enhanced_v2 và
multiclass–Ridge. Điểm submission là public/private `0.6978/0.6825`; balanced
multiclass cùng checkpoint đạt `0.6967/0.6809`. Cặp checkpoint-1208 cũ
`0.7055/0.6735` vẫn là kết quả public cao nhất. Vì baseline thực tế được ưu
tiên theo private, cần ghi rõ đây là lựa chọn sau khi xem hidden score, không
gọi đó là đánh giá chọn mô hình độc lập.

## 5. Thông tin audit cấu trúc có thể dùng

Audit trên toàn bộ 59.628 article không rỗng ghi nhận:

- 47.959 article (80,43%) được parser Khoản nhận diện;
- 10.066 article (16,88%) không có chuỗi Khoản phù hợp;
- 1.603 article (2,69%) có chuỗi đánh số mơ hồ và được fallback;
- 183.633 Khoản được nhận diện;
- 181.424 Khoản (98,80%) vừa giới hạn 512 token;
- `KhoanDiemChunker` sinh 203.435 representation;
- mọi `aid` đầu vào đều giữ được ít nhất một representation;
- độ dài encoded lớn nhất quan sát được là đúng 512 token.

Các số liệu này phản ánh coverage và hành vi preprocessing, không phải precision
hoặc recall của parser so với nhãn người.

## 6. Quy tắc dùng số liệu

Phải phân biệt ba nhóm:

1. **Đã có artifact cục bộ:** BM25 regex và các audit parser/chunking trong
   `results/`.
2. **Đã chạy nhưng do chủ dự án báo cáo:** phần lớn dense, hybrid, reranker,
   fine-tuning, LightGBM và Codabench; cần lấy artifact về trước khi công bố như
   kết quả đã kiểm chứng độc lập.
3. **Số công bố bên ngoài:** organizer baseline và leaderboard của đội khác;
   chỉ dùng làm mốc tham khảo, không gọi là kết quả của đồ án.

Không được:

- so trực tiếp F2 dev của dự án với F2 private của ban tổ chức như cùng một test;
- gọi dev-tuned threshold là kết quả test không thiên lệch;
- gắn nhãn phương pháp cho các run BGE cũ khi config artifact chưa xác nhận;
- tuyên bố reference-aware retrieval đã hoàn thành;
- suy diễn parser accuracy từ coverage;
- dùng private leaderboard để tiếp tục chọn threshold trong báo cáo nghiên cứu.

## 7. Gợi ý rút gọn theo thời lượng

Nếu có 10–12 phút, dùng khoảng 14 slide:

1. Tiêu đề.
2. Bối cảnh và bài toán.
3. Dữ liệu và metric.
4. Câu hỏi nghiên cứu.
5. Kiến trúc tổng thể.
6. BM25 baseline.
7. Chunking và vấn đề truncation.
8. Biểu diễn Khoản–Điểm.
9. Kết quả structure-aware retrieval.
10. Hybrid + reranking.
11. Fine-tuning + LightGBM.
12. Kết quả tổng hợp/Codabench.
13. Hạn chế và công việc tiếp theo.
14. Kết luận.

Đưa chi tiết parser, training protocol, threshold sweep, audit và bảng kết quả
mở rộng xuống appendix.

## 8. Nguồn đối chiếu trong repository

- Câu hỏi nghiên cứu và lộ trình: `docs/research_plan.md`.
- Schema và thống kê dữ liệu: `docs/dataset.md`.
- Metric và chống leakage: `docs/evaluation.md`.
- Cấu hình, pipeline và kết quả: `docs/experiments.md`.
- Cách chạy local/Kaggle: `docs/execution.md`.
- Organizer baseline và các đội thi: `docs/competition_review.md`.

Nếu số liệu trong brief này khác source-of-truth tương ứng, ưu tiên tài liệu
source-of-truth mới nhất và cập nhật lại brief trước khi làm slide.
