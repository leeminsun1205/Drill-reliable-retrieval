# Rà soát cấu trúc và mạch trình bày — 06/11/2026

Đối chiếu hai PDF hướng dẫn của thầy, yêu cầu chi tiết trong `doc/`, bộ slides
đã chốt và nguồn phương pháp/số liệu trong `../docs/`, `../tables_experiment.md`.
Không đổi phương pháp, dự đoán hoặc giá trị kết quả trong lần biên tập này.

## Rà từ đầu xuống cuối

- **Tóm tắt:** giữ một đoạn trong khoảng 200–300 từ, có bài toán, dữ liệu,
  chuỗi phương pháp và kết quả; URL mã nguồn đặt sau tóm tắt.
- **Giới thiệu:** nối vai trò bằng chứng với bài toán xếp hạng tập điều luật;
  định nghĩa câu hỏi, mã aid, danh sách đầu ra và gold; bốn câu hỏi nghiên
  cứu được gắn với bốn nhóm thực nghiệm.
- **Nghiên cứu liên quan:** giải thích vai trò của truy xuất từ vựng/ngữ
  nghĩa, reranker và tổng hợp tín hiệu; nhận xét bảng so sánh rồi nối tới
  lựa chọn độ dài đầu ra. Khác biệt với EDM là cơ chế quyết định điểm cắt,
  không khẳng định mọi đội khác đều dùng top-k cố định hay đồ án vượt họ.
- **Pipeline:** mô tả cả chuẩn bị chỉ mục và xử lý truy vấn. Chỉ rõ BM25
  dùng điều luật, hai nhánh ngữ nghĩa dùng đoạn; 50 đoạn được chấm lại,
  gộp về điều luật, tạo 57 đặc trưng, xếp hạng và chọn tiền tố. Tách phạm
  vi đánh giá truy xuất khỏi phần LLM phục vụ demo trong ảnh của chủ đồ án.
- **Phương pháp từng bước:** thêm lý do chuyển từ vector độc lập sang
  chấm cặp; định nghĩa sigmoid và số mẫu trong mất mát; giải thích OOF
  bằng năm nhóm câu hỏi và nối điểm OOF với bộ chọn độ dài. Giữ sự phân
  biệt 50 đoạn ứng viên với Recall@50 trên điều luật.
- **Dữ liệu/độ đo:** nhận xét biến thiên số gold, liên hệ với top-2 cố định;
  nối ký hiệu tập dự đoán với danh sách đầu ra; giải thích P@2, R@2,
  Macro-F2 và Recall@K. Nêu phần dữ liệu học cho reranker và đánh giá chéo
  của các bước sau; không cộng mức tăng từng tầng thành hiệu ứng độc lập.
- **Kết quả:** mỗi bảng/hình có đoạn dẫn, điều kiện so sánh và nhận xét
  kết quả. Các bảng từ biểu diễn đến chính sách được nối theo thứ tự
  quyết định của pipeline. Giữ nguyên toàn bộ 10 hàng benchmark đã duyệt.
- **Phân tích lỗi:** định nghĩa đơn vị là cặp câu hỏi–điều luật; ba nhóm
  không chồng lấn; nhận xét nhóm lớn nhất và liên hệ với hướng cải thiện.
  Mã aid trong ca minh họa được gọi là mã ứng viên, tránh nhầm với số Điều
  trong văn bản. Phân biệt số đếm theo cặp với Recall trung bình theo câu hỏi.
- **Kết luận:** khép lại chuỗi quyết định, kết quả và ứng dụng; giữ giới hạn
  lựa chọn cấu hình sau phản hồi private, giới hạn độ dài năm kết quả và
  ưu tiên cải thiện thứ hạng/độ bao phủ/điểm cắt.

## Bản đồ dẫn và nhận xét bảng/hình

| Đối tượng | Đoạn dẫn / mục đích | Nhận xét / liên kết sau đó |
|---|---|---|
| Bảng 1 — nghiên cứu liên quan | Cùng bài toán DRiLL, mốc private công bố | Điểm chung truy xuất–xếp hạng, khác tín hiệu/chọn đầu ra, dẫn tới selector |
| Hình 1 — pipeline | Chuẩn bị ba chỉ mục và ánh xạ về điều luật | Đi xuyên toàn bộ truy vấn → 50 đoạn → reranker → 57 đặc trưng → LightGBM → chọn kết quả |
| Bảng 2 — đặc trưng | Vì sao điểm reranker cần được bổ sung bằng tín hiệu khác | Cách dùng 57 đặc trưng, đầu ra điểm điều luật và OOF cho bước chọn độ dài |
| Bảng 3 — dữ liệu | Quy mô và phân hoạch cố định | Số gold thay đổi dẫn tới nhu cầu đầu ra thích ứng |
| Bảng 4 — biểu diễn | Cố định encoder/phân hoạch, thay cách biểu diễn | Article window thắng F2 và Recall@100; chọn làm biểu diễn tiếp theo |
| Bảng 5 — thành phần truy xuất | Giữ Article window, chọn trọng số ngoài fold đang chấm | Tính bổ sung của BM25/dense và nhu cầu đưa gold lên đầu |
| Bảng 6 — reranker | Cùng tập đoạn ứng viên, đầu ra top-2 | Fine-tune hơn mô hình ban đầu 0.0114 F2, dẫn sang kiểm tra các epoch |
| Hình 2 — epoch | Cùng top-2 trước/sau LightGBM | Thứ tự epoch thay đổi; epoch 1 dẫn sau enhanced v2 |
| Bảng 7 — schema | Gắn lại với Bảng 2, giữ epoch 1/top-2/folds | Bộ 57 dẫn ba độ đo; thêm tám đặc trưng tĩnh không cải thiện |
| Bảng 8 — chọn số kết quả | Sau biểu diễn và xếp hạng, khảo sát độ dài | Cân bằng Precision/Recall của Multiclass–Ridge và learned stopping |
| Bảng 9 — benchmark | Đối chiếu toàn hệ thống trên public/private | Khoảng cách với baseline/đội cao hơn, không suy ra hiệu ứng từng thành phần |
| Hình 3 — lỗi | Đếm cặp gold bị thiếu, giải thích ba nhóm | Nhóm ngoài top-5 lớn nhất, cần cải thiện thứ hạng lẫn điểm cắt |
| Bảng 10 — ca cụ thể | Một ca cắt thiếu, một ca giữ thừa | Nhận xét từng dòng dựa vào nhãn, thứ hạng và nội dung điều luật |

## Quy ước thống nhất

- `điều luật`: đơn vị truy xuất/đánh giá; `cấp Điều` dùng khi nói cấp cấu trúc.
- `đoạn văn bản (chunk)`: định nghĩa lần đầu, sau đó dùng `đoạn`.
- `tập ứng viên`: dùng thay pool; số 50 ở đầu vào reranker là số đoạn.
- `aid` là mã điều luật trong dữ liệu; không coi mã này là số Điều pháp lý.
- `Article window`, `WRRF`, `Vietnamese Reranker`, `LightGBM`,
  `Multiclass–Ridge`: giữ tên mô hình/phương pháp, không thêm API/thư viện.
- Decimal point `.`; số lượng hàng nghìn chia bằng khoảng trắng mảnh;
  mã định danh giữ nguyên. Main font Times New Roman 12 pt, giãn dòng 1.15.
- Số liệu đã chốt giữ nguyên: enhanced_v2 top-2 0.4269/0.7017/0.6217;
  ensemble 0.5757/0.7274/0.6910; Codabench 0.6978/0.6825.

Giới hạn nguồn số liệu của bảng checkpoint và gói dự đoán được lưu riêng ở
`../docs/evaluation.md` và `../docs/experiments.md`; không đưa lịch sử chỉnh
số hoặc ghi chú biên tập vào nội dung học thuật.

## Kết quả kiểm tra bản biên dịch cuối

- 15 trang A4; Times New Roman; 10 bảng, 3 hình và 12 nguồn được dẫn trong bài.
- Tóm tắt 265 từ theo phép đếm khoảng trắng, trong khoảng 200–300.
- Không có cảnh báo LaTeX, tràn dòng, thiếu ký tự hoặc tham chiếu không xác định.
- Tất cả bảng/hình được dẫn; nhãn không trùng; mọi tài liệu tham khảo đều được trích.
- Các ô dữ liệu số trong bảng trùng bản trước khi rà soát, gồm cả hai ca minh họa.
- Kiểm tra 22 hàng P/R/F2 theo công thức chính thức trong sai số làm tròn bốn chữ số.
- Đã xem toàn bộ 15 trang qua bản kết xuất và xem lớn trang pipeline, dữ liệu,
  phân tích lỗi. Không thay phông/cỡ chữ hoặc thu nhỏ bảng để ép số trang.
- Không huấn luyện lại mô hình, đổi dự đoán hoặc chạy lại thí nghiệm trong lần này.
