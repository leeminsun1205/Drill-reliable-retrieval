> **Demo đất đai 3–5 phút (2026-11-10):** [demo_script.md](demo_script.md) thay thế kịch bản demo 2 phút, mở rộng đủ tính năng hiện có. Mục tiêu khoảng 4 phút 30 giây khi kết hợp 1–2 lượt trực tiếp với hội thoại đã chạy trước được công bố rõ; thời gian chưa đo qua diễn tập và phụ thuộc API.

> **Kịch bản demo (2026-11-10):** [demo_script.md](demo_script.md) có lời dẫn, thao tác, bản rút gọn và câu chuyển khi chờ/lỗi. Bản mở rộng khoảng 2 phút lời nói, chưa tính chờ; thay thế chứ không cộng dồn vào đoạn demo 20–30 giây của kịch bản 10 phút.

> **Bản PDF kịch bản:** [slides/presentation_script.pdf](../slides/presentation_script.pdf), 5 trang A4; xuất từ kịch bản 3 người bằng XeLaTeX, phông Noto Serif/Inter. Nguồn LaTeX nằm cùng thư mục slides (được bỏ qua bởi quy tắc Git hiện tại).

> **Kịch bản 3 người (2026-11-08):** xem [presentation_script.md](presentation_script.md), bám PDF 32 trang hiện tại. Người 1: phần 1–2; người 2: phần 3; người 3: phần 4 và demo ngắn. Mục tiêu khoảng 10 phút, thời gian chưa đo qua diễn tập; dùng LightGBM top-2 F2 0.6217 theo PDF mới nhất.

# Brief nội dung slide trình bày đồ án

> **Chốt sửa số và phạm vi demo:** chủ dự án xác nhận nhập nhầm F2;
> enhanced_v2 và fixed-top-2 dùng 0.6203 (từ P=0.4223, R=0.7027). Thêm một
> slide Demo ngay sau kết luận: truy xuất cung cấp bằng chứng cho RAG; LLM
> sẵn có minh họa hỏi đáp, chưa đánh giá sâu tầng sinh câu trả lời. Không
> khẳng định LLM mạnh nên không còn cần đánh giá, hay truy xuất bảo đảm toàn
> bộ chất lượng RAG. PDF hiện có 31 trang.


> **Đã duyệt tên gọi và cập nhật code:** giữ nguyên ký hiệu độ đo; “Article
> window” là tên trình bày chính thức (code vẫn token_window). Đổi “First-stage
> components” thành “Thành phần truy xuất”; rút tiêu đề cắt cụt, đổi màu ô minh
> họa con để khác nền, tách trọng số khỏi caption dài. Code LightGBM mặc định
> bỏ bốn cột ref, đúng 57 đầu vào với hai dense; có chế độ legacy rõ ràng để
> đọc artifact 61 cột. Không có kết quả thực nghiệm mới từ lần sửa code này.


> **Nhận xét bảng:** thống nhất năm bảng đầu thành chữ xanh lá ngoài khung,
> cùng cỡ chữ và khoảng cách dưới bảng. Bảng 6 không thêm nhận xét để giữ
> kích thước. Giữ nguyên metric và kích thước bảng.


> **Rút phần reranker theo yêu cầu chủ dự án:** một slide suy luận (50 chunk
> → cross-encoder → max theo aid → ranking), một slide fine-tuning (mẫu
> query–article bag, gold/âm khó, max logit và pointwise MIL BCE). Bảng reranker
> giữ nguyên hàng/cột/metric, bỏ khung cấu hình và thêm nhận xét ngắn.
> Không thêm thí nghiệm mới; PDF giảm một trang.


> **Kiểu khung hiện hành:** Chủ dự án duyệt khung phẳng như sơ đồ LightGBM:
> nền nhạt, viền mảnh, bo góc, không bóng đổ. Áp dụng template chung cho
> block/exampleblock/alertblock và các hộp sơ đồ; chỉ chỉnh style/đệm khung,
> giữ nguyên nội dung và số liệu. Không thay palette toàn bộ slide.


> **Phần 1 đã duyệt và chỉnh:** mục tiêu “đúng và đủ các điều liên quan”;
> hình pháp luật ghi minh họa cấu trúc. Phát biểu bài toán tách đầu vào,
> xử lý ba bước và đầu ra mong muốn; gold chỉ ở phía đối chiếu, không là
> inference input. Ví dụ qid 730 và hai aid đã được xác minh từ dữ liệu local
> (xem docs/dataset.md). Bốn câu hỏi nghiên cứu gắn với biểu diễn, hybrid/
> reranking, LightGBM và số đầu ra; giữ demo hỏi đáp là ứng dụng minh họa.


> **Rút trang chuyển giai đoạn:** bỏ bốn trang 3.1–3.4 theo yêu cầu chủ dự
> án. Giữ nguyên nội dung và bảng; PDF hiện còn 32 trang.


> **Số đếm đặc trưng mới nhất:** Theo yêu cầu chủ dự án, slide không tính
> bốn cột provenance ref hằng 0 trong pipeline reference-off. Số đếm trình
> bày base/v1/v2/v3 = 15/37/57/65; LightGBM chính và kết luận dùng 57.
> Bảng có chú thích cách đếm; metric và schema artifact gốc không đổi.
> Chưa chạy lại mô hình hay đổi chiều đầu vào code. Các ghi chú “61” cũ
> bên dưới mô tả schema artifact, không phải số đếm trình bày hiện tại.


> **Tinh chỉnh slide LightGBM:** đặt câu hỏi định hướng rõ, tăng khoảng cách
> giữa dòng mở đầu và sơ đồ bằng ngắt đoạn tường minh. Ba khung rút còn cặp
> (q, aid)/61 đặc trưng → ước lượng liên quan → điểm và thứ hạng. Bỏ khung
> huấn luyện; thay bằng chú thích nhóm tín hiệu, ký hiệu, nhãn và OOF ngắn.
> Bảng thực nghiệm không thay đổi.


> **Giai đoạn 3.4 đến cuối — đã được chủ dự án duyệt:** thêm đầu vào ranking
> LightGBM và đầu ra top-k; mô tả learned stopping theo ranh giới và ensemble
> theo các lựa chọn top-1..5, giải thích fold-safe ngắn. Bảng 5 giữ nguyên số
> liệu, thêm nhận xét F2/Recall. Benchmark là đối chiếu toàn hệ thống, kết luận
> phân biệt dev/OOF với Codabench. Khác biệt với EDM/FPT IS chưa là nguyên nhân
> được xác nhận của chênh lệch điểm. Trang cuối chuyển sang demo; không thêm trang.


> **LightGBM:** Gộp hai slide nội dung thành một sơ đồ đầu vào (nhóm tín hiệu,
> 61 đặc trưng) → phân loại nhị phân → điểm và thứ hạng aid; giữ giải thích
> OOF theo qid. Bảng schema ablation ngay sau đó giữ nguyên. PDF còn 36 trang.


> **Quyết định màu sắc mới nhất:** Khôi phục đúng nguồn trước mọi lần đổi
> palette (màu và khung gốc). Giữ kết luận gộp, bỏ tham khảo, sửa lỗi nhỏ,
> Bảng 5 phóng to. Những mô tả đổi palette phía dưới không còn áp dụng.


> **Cân bằng màu theo phản hồi mới nhất:** nền nội dung trắng, thanh tiêu đề
> xanh trung bình (#345F96) và chữ trắng; tiêu đề khung nền xanh nhạt rõ hơn
> (#D9E6F4), chữ navy đậm; thân khung #EDF3FA. Giữ footer sáng, không bóng
> đổ, không thêm nhiều gam màu. Thay thế palette quá tối/quá nhạt trước đó.


> **Palette hiện hành:** Theo phản hồi bản trước quá tối, dùng nền trắng,
> tiêu đề chữ navy đậm trên trắng, xanh royal sáng làm điểm nhấn, nền khung
> xanh rất nhạt. Bỏ các thanh nền tối ở đầu/chân trang; trang chuyển phần
> cũng có nền trắng và dải xanh nhỏ. Giữ nội dung, sơ đồ và số liệu.


> **Hiệu chỉnh tương phản:** Sau phản hồi chữ bị chìm, giữ một gam xanh nhưng
> dùng chữ nội dung gần đen/đen trên nền gần trắng; tiêu đề khung xanh đậm,
> chữ trắng. Ưu tiên đọc rõ khi trình chiếu hơn các tone gần nhau.


> **Màu sắc mới nhất:** Chủ dự án yêu cầu một gam màu với nhiều tone.
> Palette xanh slate: chữ/tiêu đề đậm, viền vừa, nền khung rất nhạt;
> bỏ đỏ/vàng/teal khác gam và bóng đổ. Nội dung, thứ tự, kích thước bảng
> và bố cục được giữ nguyên trong lần đổi màu này.


> **Quyết định mới nhất:** Chủ dự án yêu cầu bỏ hẳn trang tài liệu tham khảo
> khỏi slide. PDF còn 37 trang; danh mục nguồn trong report/tài liệu nghiên
> cứu vẫn được giữ. Ghi chú về năm nguồn trên slide bên dưới đã hết hiệu lực.


> **Tài liệu tham khảo:** Theo yêu cầu chủ dự án, slide tham khảo dùng một cột,
> năm mục chữ lớn hơn: DRiLL overview, BM25, RRF, E5-large và LightGBM.
> Mỗi mục hiển thị tên nguồn ngắn gắn với vai trò, tác giả/năm và liên kết gốc; danh mục đầy đủ trong
> tài liệu nghiên cứu không bị rút gọn.


> **Sửa lỗi nhỏ sau khi gộp kết luận:** sửa Article truncation, 448/64, hai
> câu nhận xét chunking; in đậm Recall 0.7303 thay vì 0.7274; bỏ cột thứ tự
> và ghi rõ benchmark là đối chiếu. Giữ đủ hệ thống, metric và số liệu; chưa
> chỉnh các vấn đề bố cục khác. Ghi chú lựa chọn baseline hậu nghiệm đã có
> trên slide kết luận mới.


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


Cập nhật 2026-11-05. Đây là hướng dẫn cho bộ slide đã biên tập tại
`slides/drill_retrieval.tex` và PDF tương ứng. Bảng số liệu report trong
`tables_experiment.md` vẫn là nguồn chuẩn; không suy ra kết quả mới khi rút gọn.

## Phạm vi và thông điệp

Demo là hỏi đáp pháp luật có dẫn chứng. Nghiên cứu đánh giá truy xuất bằng chứng
ở cấp điều luật (`aid`); chưa đánh giá định lượng tầng sinh câu trả lời.
Theo quyết định của chủ dự án, reference expansion được loại khỏi phạm vi
trình bày và report hiện tại. Không đề xuất chạy thêm treatment này.

Bốn câu hỏi dẫn dắt: biểu diễn điều luật; truy xuất và reranking; kết hợp tín
hiệu; chọn số điều trả về. Không thu hẹp toàn bộ đồ án vào riêng chunking.

## Cấu trúc và yêu cầu biên tập hiện tại

Chủ dự án xác nhận thời lượng 10–15 phút có demo, sau đó bác bỏ cách chuyển
bảng sang phụ lục và giảm cột số liệu. Quyết định mới nhất: giữ cấu trúc giải
thích thành phần rồi đến bảng thực nghiệm ngay tại chỗ; giữ đủ hàng/cột của
các bảng gốc. Không chia phụ lục, không thay sơ đồ bằng các đoạn chữ.

Bản hiện tại được chỉnh từ nguồn trước lần thiết kế lại, có 32 trang thay vì
40: chỉ bỏ tám trang chuyển phần/giai đoạn riêng. Giữ thứ tự các nội dung,
sáu bảng thực nghiệm đầy đủ, các hình chia đoạn, ví dụ reranking, sơ đồ học
MIL, sơ đồ LightGBM và ví dụ tính WRRF. JSON gốc được đổi thành sơ đồ quan hệ
văn bản–điều–chunk. Sửa chính tả, 488/64 thành 448/64 và metric in đậm của R7.
Bảng benchmark giữ các hệ thống và cột điểm, bỏ cột thứ tự để tránh hiểu là
thứ hạng chính thức. Chưa xác nhận lại toàn bộ nguồn public của đội khác.

Thời lượng chưa được đo qua diễn tập. Cần lướt các định nghĩa/cấu hình, dành
thời gian cho luồng xử lý và demo; không tự cắt bảng để ép thời lượng.
Các ví dụ trước/sau reranking có nhãn minh họa, không phải dự đoán đã đo.
Bản đầy đủ trước chỉnh nằm ở `slides_old/`; không ghi đè bản này.

Demo được đối chiếu với `demo/streamlit/`: trả các điều đã chọn gồm aid,
nội dung article, law_id và score. Tầng trả lời nhận câu hỏi và bằng chứng;
giao diện hiển thị riêng căn cứ. Văn bản mỗi điều gửi API giới hạn 5.000 ký tự.
Không khẳng định đã chạy thử demo chỉ từ việc đọc mã.

## Chính xác về phương pháp và số liệu

Ngân sách reranker là 50 chunk, không phải 50 điều. Điểm chunk được max về aid.
Recall@50 trong bảng retrieval tính theo điều luật. BM25 đóng góp hạng điều cha;
hai dense đóng góp hạng chunk trong direct-chunk WRRF. Chunk trùng chính xác
được gộp đóng góp. LightGBM là bộ phân loại nhị phân dùng điểm để xếp hạng.
Multiclass–Ridge kết hợp log-xác suất độ dài và lợi ích tiền tố sau chuẩn hóa;
chọn tiền tố tốt nhất trong tối đa năm vị trí, không dự đoán trực tiếp số gold.

Bảng chunking trên slide giữ ba hàng gốc; R2 report còn có hàng Khoản–Điểm. Không diễn giải bảng slide thành toàn bộ R2. R3 báo cáo cross-fitted
hybrid, không thay bằng điểm fit trên toàn dev. R6 giữ số liệu enhanced_v2 đã
được sửa: 0.4223 / 0.7027 / 0.6203. R7 dùng cùng ranking enhanced_v2:
Multiclass–Ridge dẫn Precision/F2; learned stopping dẫn Recall (0.7303).
Kết quả do chủ dự án cung cấp không được nâng thành artifact-verified chỉ nhờ
kiểm tra slide. Chưa có paired uncertainty, không kết luận ý nghĩa thống kê.

Baseline thực dụng checkpoint 1234 đạt public/private 0.6978/0.6825; việc chọn
baseline ưu tiên private là hậu nghiệm và phải được công khai. OOF downstream
không loại bỏ thiên lệch do sử dụng dev nhiều lần để chọn backbone.

## Thiết kế

Nền trắng, chữ navy, xanh teal nhấn; bỏ bóng đổ và footer nhiều dải màu.
Mỗi trang có một tiêu đề thể hiện thông điệp, sơ đồ hoặc bảng chính, một kết
luận ngắn khi cần. Không đưa code hoặc JSON lên slide. Giảm caption dài, giữ cấu hình cần thiết ngay cạnh sơ đồ hoặc bảng tương ứng. In đậm metric tốt nhất theo cột;
không dùng in đậm cả hàng thay cho quy tắc này.

Hội đồng ưu tiên giải thích cốt lõi và luồng xử lý; dùng thuật ngữ khi cần và
không đưa code lên slide. Thời gian dự kiến cần xác nhận lại qua diễn tập.
