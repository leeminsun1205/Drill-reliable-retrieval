# Cấu hình chính: đầu vào, đầu ra và fine-tuning

Đối chiếu ngày 2026-11-11 với bộ nạp artifact demo, tài liệu thí nghiệm và
notebook `notebooks/training/kaggle_finetune_reranker_fixed_refoff_3epochs_sched5.ipynb`.
Đây là ghi chú giải thích, không phải lần huấn luyện/đánh giá mới.

## Phạm vi và nguồn cấu hình

Cấu hình chính là Article window 448/64, reference-off, BM25/Harrier/E5,
WRRF10 trọng số 0.15/0.65/0.20, pool-50 chunks, checkpoint-1234, LightGBM
enhanced_v2 và Multiclass–Ridge top-1..5. Generic YAML
`configs/hybrid_reranker_multi_dense.yaml` là template khác (k=20, trọng số và
chunking khác), không dùng nó để khẳng định tham số của run đã chọn. Demo đọc
và kiểm tra cấu hình đóng băng trong complete-results archive.

## Đơn vị và số lượng

- Corpus: 2.157 văn bản, 59.628 điều luật; đơn vị đầu ra là aid.
- Train/dev: 1.752/438 câu hỏi từ 2.190 câu có nhãn, seed 42 và giữ split IDs.
- BM25: top-100 điều; Harrier/E5: top-100 theo nhánh dense ở mức chunk.
- Dense: mainguyen9/vietlegal-harrier-0.6b và intfloat/multilingual-e5-large.
- Chunk: tối đa 448 token theo tokenizer chia đoạn BAAI/bge-m3, overlap 64.
  Overlap không phải 64 token cộng vào chiều dài mỗi chunk.
- Fusion: weighted RRF k=10; k ở đây là hằng số làm mượt hạng.
- Cross-encoder: 50 chunk ứng viên, mỗi cặp câu hỏi–chunk tối đa 512 token.
  Gộp max theo aid nên có thể có ít hơn 50 điều sau reranking.
- LightGBM: mỗi hàng là cặp (q, aid), enhanced_v2 hiện có 57 đặc trưng;
  artifact cũ 61 cột có thêm bốn cột provenance reference, được demo đọc đúng
  tên/thứ tự cột. Không tự cắt cột của model đã lưu.
- Multiclass–Ridge: chọn một tiền tố top-k, k từ 1 đến 5, không trả cố định 5.
- Batch rerank demo: 8; khác batch fine-tuning. Runtime CPU/GPU không phải
  tham số thay thế mô hình hoặc chính sách khoa học.

## Đầu vào/đầu ra từng tầng

Câu hỏi và corpus article-level -> danh sách ứng viên từ ba nhánh -> 50 chunk
sau WRRF -> cross-encoder score từng cặp -> max score mỗi aid -> vector đặc
trưng từng (q, aid) -> LightGBM score và ranking aid -> chính sách chọn độ dài
-> danh sách aid đã chọn. Labels/gold chỉ dùng khi học và đánh giá.

Submission ánh xạ qid sang relevant_laws chứa aid; demo bổ sung text, law_id,
score cho nguồn rồi gọi LLM để sinh câu trả lời. aid là ID, không phải số điều
hiển thị trong văn bản. LLM sinh trả lời không được fine-tune trong pipeline này.

## Tạo mẫu fine-tune

Warm start là Vietnamese reranker gắn trong notebook, không mặc định thay bằng
BAAI/bge-reranker-v2-m3 của template YAML. Dùng phía train để tạo handoff ứng
viên, tuyệt đối loại 438 câu dev khỏi mining.

Một mẫu là (query, bag các passages thuộc cùng aid, label). Mỗi gold aid tạo
một positive bag; dùng chunk có trong ứng viên hoặc fallback bằng cùng chunker
nếu thiếu. Teacher chỉ xếp hạng âm, không chọn gold. Positive preprocessing
đặt cap 32 representations/aid; mỗi epoch trainer dùng tối đa bốn passages/bag
với seeded-cycle, không nhất thiết bốn chunk đầu và không xóa các passages còn lại.

Âm được lấy từ các aid không thuộc gold. Teacher chấm chunk, gộp max theo aid,
lấy tối đa 30 aid âm đứng đầu và bỏ năm aid âm đầu. Từ pool còn lại, lấy ngẫu
nhiên không lặp 10 nếu pool >15, ngược lại lấy năm; thiếu số lượng cần lấy thì
bỏ query theo code mining. Negative bag giữ các candidate passages của aid
được chọn, không phải chỉ một chunk đạt điểm cao nhất.

Không suy ra tổng số mẫu huấn luyện từ 1.752 câu hỏi hoặc từ giới hạn 5/10 âm.
Số bag dương/âm thực tế và query bị bỏ phải đọc metadata/audit của run.

## Objective và optimizer

Với bag của aid có chunk logits z_i, lấy z_aid=max_i z_i. BCEWithLogitsLoss
so sánh z_aid với y=1 (gold) hoặc y=0 (âm). Đây là pointwise MIL BCE, không phải
pairwise ranking loss và không gán tất cả chunk của gold thành positive độc lập.

Notebook chính dùng:

| Tham số | Giá trị |
|---|---|
| GPU train | 2, DDP |
| Per-device batch | 1 bag |
| Gradient accumulation | 8 |
| Effective batch | 16 bags/optimizer update |
| Bag passages mỗi epoch | tối đa 4, seeded-cycle |
| Pair max length | 512 token |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 (default script, notebook không override) |
| Weight decay | 0.01 (default script, notebook không override) |
| Epoch thực chạy | 3 |
| LR scheduler horizon | 5 epoch, linear warmup/decay |
| Seed | 42 |

Checkpoint theo epoch là 1234/2471/3709. Cấu hình chính dùng epoch 1/1234.
Không diễn giải điều đó thành epoch 1 tốt nhất ở reranker-only: epoch 3 có F2@2
reranker-only cao hơn, còn kết quả sau LightGBM trên slides hỗ trợ epoch 1.
Các giới hạn về đối chiếu metric và lựa chọn deployment hậu nghiệm được giữ
trong docs/experiments.md và docs/evaluation.md.

LightGBM và các head Multiclass–Ridge được học riêng từ đặc trưng/ranking có
nhãn và kiểm tra theo query-fold; chúng không nhận gradient từ BCE của reranker.
Hai dense encoder và LLM trả lời không được fine-tune trong treatment chính này.

## Multiclass–Ridge học cụ thể thế nào?

Đối chiếu `src/drill_ir/analysis/cardinality_confirmatory.py` và
`prefix_utility.py`: đầu vào là ranking aid/score đã có, không phải raw text.
Với mỗi câu, tính F2 từng tiền tố k=1..min(5, số ứng viên), đối chiếu gold.
Multiclass label là argmax F2, hòa chọn k nhỏ hơn; đây không phải số gold aid.
Mỗi query tạo một hàng multiclass. Ridge tạo một hàng cho mỗi lựa chọn k,
target là F2 của tiền tố tương ứng, tối đa năm hàng/query.

Hai head đều dùng `_prefix_features` (21 chiều khi maximum_k=5): năm điểm,
bốn score gap, mean/std/range/entropy toàn danh sách, k/5, điểm tại k,
boundary gap, mean/std của prefix và tail, và khối lượng softmax của prefix.
Multiclass dùng vector lấy với cutoff=1 cố định, nên một số cột là hằng;
Ridge dùng vector với cutoff=k của từng phương án. Mỗi nhánh có StandardScaler
riêng được fit trên phía train của fold.

Multiclass là LogisticRegression(class_weight=balanced), học lớp k tối ưu.
Ridge là hồi quy tuyến tính có L2, học dự đoán F2 của từng lựa chọn top-k;
đầu ra không được khẳng định là xác suất hay chắc chắn nằm trong [0,1].
Ensemble hiện tại dùng lambda*Z(log(p_k+1e-12))+(1-lambda)*Z(ridge_k),
với Z chuẩn hóa trên các lựa chọn k của riêng query. Chọn argmax, rồi lấy
đúng tiền tố của ranking có sẵn; không sắp xếp lại aid và không dùng utility
matrix của nhánh cost_sensitive_multiclass trong quyết định ensemble này.
Lambda được chọn bằng cross-fitting bên trong training folds, theo F2 tổng hợp
của evaluator, khác với target per-query F2 dùng để dạy Ridge. Đánh giá theo
năm outer query folds; triển khai refit trên dữ liệu phát triển có nhãn và lưu
cả head/scaler/lambda. Những giới hạn post-selection đã ghi trong experiments
và evaluation vẫn áp dụng, không suy ra đây là đánh giá độc lập của toàn bộ
quá trình chọn cấu hình.

Ví dụ minh họa, không phải kết quả thực nghiệm: ranking A,B,C,D,E và gold A,C.
F2 top-1..5 lần lượt 0.5556/0.5000/0.9091/0.8333/0.7692. Nhãn multiclass là
k=3 dù chỉ có hai gold aid; Ridge học cả năm điểm utility này từ năm hàng.
