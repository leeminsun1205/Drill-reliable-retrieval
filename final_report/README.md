> **Cập nhật mới nhất:** Tóm tắt dùng “Kết quả đánh giá cuối cùng”, nêu các thành phần hỏi đáp sau truy xuất chỉ phục vụ demo. Mục 4.4 và hướng phát triển đã bổ sung giả thuyết/hướng thử làm giàu tiêu đề–tóm tắt, đặc trưng đánh giá và lọc bằng chứng bằng LLM như FPT IS/EDM, kèm kiểm chứng và chi phí. Slides có thêm biểu đồ epoch cùng nhận xét ở trang 25, tổng 32 trang.

# Báo cáo cuối kỳ

- **Bản đọc:** `main.pdf`, 15 trang A4 gồm bìa, tóm tắt/mục lục và tài liệu tham khảo.
- **Nguồn:** `main.tex`; hai phần hình/nhận xét riêng trong `sections/`.
- **Bản nhóm gửi ban đầu:** `original/main.tex` và `original/main.pdf`.
- **Hướng dẫn của thầy:** `huongdanvietbai-1.pdf` và `doc/huongdanvietbai.pdf`;
  yêu cầu chi tiết ở `doc/yeu_cau_bao_cao.md`.
- **Rà soát từng phần:** `review_notes.md` ghi cách kiểm tra cấu trúc, mạch
  phương pháp, dẫn/nhận xét cho toàn bộ 10 bảng và 3 hình.

## Nội dung hiện tại

Mạch bài: tóm tắt → bài toán và bốn câu hỏi nghiên cứu → nền tảng/nghiên cứu
liên quan → phương pháp → dữ liệu, độ đo và kết quả theo từng bước → phân
tích lỗi → kết luận, ứng dụng và giới hạn. Có 12 tài liệu tham khảo.

Hình pipeline dùng nguyên ảnh của chủ đồ án tại
`../slides/figures/Drill_pipeline_final.png`. Văn bản giải thích chuẩn bị chỉ
mục, ba nhánh truy xuất, 50 đoạn ứng viên, reranker, gộp về điều luật, 57 đặc
trưng, LightGBM và chọn số điều luật. Phần LLM trong ảnh phục vụ demo.

Bảng 9 giữ đủ 10 hệ thống và hai cột điểm của slides đã được chủ đồ án xác
minh. Số đặc trưng trình bày là 15/37/57/65. Enhanced v2 epoch 1 dùng
P/R/F2 = 0.426941/0.701696/0.621680, hiển thị 0.4269/0.7017/0.6217.
Không thay đổi giá trị kết quả trong lần rà soát văn phong và liên kết này.

## Thuật ngữ và định dạng

- `điều luật`: đơn vị truy xuất và đánh giá; mã `aid` là định danh trong dữ liệu.
- `đoạn văn bản (chunk)`: định nghĩa lần đầu, sau đó dùng `đoạn`.
- `tập ứng viên`: thay cho pool; đầu vào reranker là 50 **đoạn**, khác 50 điều luật.
- Giữ tên các mô hình/phương pháp; không đưa API, Lucene, mã checkpoint hoặc
  lịch scheduler vào bài. Chi tiết tái lập nằm trong repo và cấu hình.
- Số thập phân dùng dấu chấm; số lượng hàng nghìn dùng khoảng trắng mảnh.
  Mã định danh giữ nguyên. URL mã nguồn đặt ngay sau tóm tắt.
- Times New Roman 12 pt, giãn dòng 1.15; lề trái/phải/trên/dưới 3/2/2.5/2.5 cm.
  Bốn kiểu phông được nạp trực tiếp từ `fonts/`, được Git bỏ qua;
  nguồn tải/cài lại ở `fonts/README.md`.

## Hình thực nghiệm và nguồn

- `figures/reranker_epochs.csv/.tex/.pdf`: hai đường đều dùng top-2.
  Epoch 3 dẫn khi chỉ dùng reranker; epoch 1 dẫn sau LightGBM enhanced v2.
  Các điểm là kết quả checkpoint đã được chủ đồ án cung cấp, ghi trong
  `../docs/experiments.md`; không chạy lại huấn luyện để vẽ hình.
- `figures/error_counts.csv/.tex/.pdf`: 206 cặp gold bị bỏ sót, phân thành
  60 ngoài tập ứng viên, 97 ngoài top-5 và 49 bị cắt tiền tố.
- `sections/error_analysis.tex`: hai ca qid 14663/2406 được đối chiếu với
  nhãn, thứ hạng và nội dung điều luật. Mã aid không phải số Điều pháp lý.

Nguồn phân tích lỗi là gói
`../results/pool50_fixed448_refoff_epoch1_ckpt1234_r3_retuned_multiclass_ridge_submission_complete_results.tar.gz`.
Lệnh tái lập: `scripts/audit_report_errors.py` từ thư mục gốc repo; đầu ra
ở `../results/report_error_audit_1234/`. Dự đoán ensemble tái dựng khớp cả
năm fold và tổng P/R/F2 0.5757229833/0.7274353120/0.6910165043.

Số checkpoint-screen chủ đồ án cung cấp và gói OOF dùng cho phân tích lỗi
là hai nguồn riêng. Khác biệt top-2 0.621680 so với 0.6113609814 được giữ
trong `../docs/evaluation.md` và `../docs/experiments.md`; không gọi chúng
là cùng một tệp xếp hạng. Các ghi chú sửa số/phiên bản không đưa vào bài.

## Biên dịch

Trong `final_report/figures/`, chỉ biên dịch lại khi sửa nguồn hình:

```bash
latexmk -xelatex error_counts.tex
latexmk -xelatex reranker_epochs.tex
```

Trong `final_report/`:

```bash
latexmk -xelatex -synctex=1 -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf
pdffonts main.pdf
```

Kiểm tra cuối gồm số trang, cảnh báo LaTeX, liên kết bảng/hình/tài liệu,
font thực tế và bảo toàn các ô số liệu so với bản trước rà soát.

Kiểm tra hoàn tất: 15 trang, 10 bảng, 3 hình, 12 nguồn đều được dẫn; tóm tắt
265 từ; không cảnh báo LaTeX; 22 hàng P/R/F2 đúng trong sai số làm tròn;
các ô số liệu giữ nguyên. Bản kết xuất đã được xem để rà bố cục.
