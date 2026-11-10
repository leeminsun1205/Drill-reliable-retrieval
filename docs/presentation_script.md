# Kịch bản thuyết trình 3 người — khoảng 10 phút

Soạn theo `slides/drill_retrieval.pdf` hiện tại, 32 trang. Thời gian là phân bổ
ước tính, chưa đo qua diễn tập. Ưu tiên lời nói liền mạch; không đọc từng hàng
trong bảng. Các dòng trong ngoặc vuông là chỉ dẫn thao tác, không đọc thành lời.
Không phân vai theo tên thành viên khi chưa có chỉ định.

| Người | Phần phụ trách | Trang PDF | Thời lượng dự kiến |
|---|---|---|---|
| 1 | Phần 1 và 2 | 1–11 | 2 phút 30 giây |
| 2 | Phần 3 | 12–28 | 5 phút 30 giây |
| 3 | Phần 4, demo ngắn, kết thúc | 29–32 | 2 phút |

Demo dự kiến 20–30 giây, dùng một hội thoại đã chạy sẵn và mở nguồn; không chờ
model khởi động trong thời gian thuyết trình. Nếu hội đồng yêu cầu nhập câu mới,
thời gian chờ suy luận được tính thêm. Các mốc F2 là số trên slides, không phải
lần đánh giá mới. Không diễn giải chuỗi mốc thành ablation một biến; không tự
biến bảng đối chiếu đội khác thành bảng thứ hạng chính thức.

## Người 1 — Phần 1 và 2

[Trang 1–3. Lướt trang mục lục và trang chuyển phần.]

Kính chào thầy và các bạn. Chúng em là nhóm 18. Hôm nay, nhóm xin trình bày đề tài
“Đánh giá và cải tiến truy xuất bằng chứng pháp lý cho hỏi đáp pháp luật Việt Nam”,
nghiên cứu trên bộ dữ liệu VLSP 2025 DRiLL.

Bài trình bày đi từ bối cảnh và dữ liệu, đến phương pháp, kết quả thực nghiệm và
ứng dụng minh họa. Trọng tâm của nhóm là tìm đúng và đủ điều luật làm bằng chứng
trước khi sinh câu trả lời.

[Trang 4.]

Truy xuất pháp luật có một số khó khăn đặc thù. Văn bản thường dài, có cấu trúc
Điều, Khoản, Điểm và chứa nhiều thuật ngữ chuyên ngành. Trong khi đó, người dùng
thường hỏi bằng ngôn ngữ đời thường, khác với cách diễn đạt trong luật.

Ngoài ra, một câu hỏi có thể cần nhiều điều luật để trả lời đầy đủ. Vì vậy, tìm
được một đoạn văn có từ khóa giống câu hỏi chưa chắc đã đủ căn cứ để kết luận.

[Trang 5–6.]

Từ đó, nhóm xác định đầu vào là câu hỏi và kho điều luật đã được lập chỉ mục.
Hệ thống thực hiện ba việc: tìm các ứng viên, xếp hạng mức liên quan, rồi chọn
số điều luật cần trả về. Mỗi điều được nhận diện bằng mã aid.

Ví dụ trên slide hỏi viên chức có được tạm hoãn nghĩa vụ quân sự hay không. Đầu
ra mong muốn là những điều luật liên quan, thay vì chỉ một câu trả lời nghe có
vẻ hợp lý.

Nhóm đặt ra bốn câu hỏi nghiên cứu: biểu diễn điều luật thế nào; kết hợp truy
xuất từ vựng với ngữ nghĩa có hiệu quả không; LightGBM có giúp xếp hạng tốt hơn
không; và nên trả về số điều cố định hay thay đổi theo từng câu hỏi.

[Trang 7–9. Lướt trang chuyển phần.]

Để kiểm chứng, nhóm sử dụng dữ liệu DRiLL. Kho văn bản có 2.157 văn bản pháp luật,
tương ứng 59.628 điều. Dữ liệu gốc tổ chức theo văn bản, nhưng đơn vị truy xuất
và đánh giá của nhóm là từng điều luật.

Trong 2.190 câu hỏi có nhãn, nhóm chia thành 1.752 câu huấn luyện và 438 câu phát
triển. Hai tập kiểm tra public và private lần lượt có 312 và 627 câu hỏi. Tập
phát triển được dùng để so sánh các phương án; số liệu test được báo cáo riêng.

[Trang 10–11.]

Về đánh giá, Precision cho biết trong các điều trả về, có bao nhiêu điều đúng.
Recall cho biết trong các điều cần tìm, hệ thống đã tìm được bao nhiêu.

DRiLL tính Precision và Recall trung bình trên các câu hỏi, sau đó tổng hợp thành
Macro-F2. Chỉ số này ưu tiên Recall, nhưng vẫn phạt việc trả về quá nhiều điều
không liên quan.

Do đó, hệ thống phải giải quyết đồng thời hai yêu cầu: không bỏ sót căn cứ quan
trọng và không đưa quá nhiều căn cứ thừa. Sau đây, em xin chuyển phần trình bày
cho bạn thứ hai để giải thích cách nhóm xử lý hai yêu cầu này và các kết quả đạt được.

## Người 2 — Phần 3

[Trang 12–13. Lướt trang chuyển phần, chỉ vào pipeline.]

Cảm ơn bạn. Từ bài toán vừa nêu, nhóm xây dựng pipeline gồm bốn tầng chính:
truy xuất lai để tìm ứng viên, cross-encoder để chấm lại mức liên quan, LightGBM
để kết hợp tín hiệu và cuối cùng là chính sách chọn số điều trả về.

Trên sơ đồ, phần lập chỉ mục được chuẩn bị từ kho luật. Khi có câu hỏi mới, hệ
thống sử dụng các chỉ mục này để tìm và xếp hạng bằng chứng. Phần sinh câu trả
lời bên phải là ứng dụng minh họa, được giới thiệu ở cuối bài.

[Trang 14–16.]

Đầu tiên là biểu diễn văn bản. Nhánh từ vựng tìm các thuật ngữ trùng khớp, còn
nhánh dense mã hóa văn bản thành vector để tìm theo ngữ nghĩa.

Với dense, nhóm so sánh ba cách chia văn bản: cắt cụt điều luật, chia theo Khoản
và chia cửa sổ theo Điều. Cách cắt cụt có thể bỏ mất thông tin ở cuối điều dài.
Hai cách còn lại sử dụng cửa sổ 448 token, chồng lấp 64 token để giữ ngữ cảnh
quanh ranh giới đoạn.

Trong bảng một, Article window đạt Recall@50 là 0,9256 và F2@2 là 0,5098, cao
nhất trong ba phương án. Vì vậy, nhóm sử dụng cách biểu diễn này cho các bước
sau. Các chunk được dùng để tìm kiếm, nhưng kết quả cuối vẫn quy về điều luật.

[Trang 17–19.]

Tiếp theo, nhóm kết hợp ba nhánh truy xuất: BM25, VietLegal-Harrier và
multilingual-E5-large. BM25 bổ sung tín hiệu từ khóa, trong khi hai mô hình dense
cung cấp tín hiệu ngữ nghĩa.

Để hợp nhất, nhóm sử dụng Weighted RRF. Ý tưởng là tổng hợp vị trí xếp hạng từ
các nhánh theo trọng số, thay vì cộng trực tiếp những điểm số có thang đo khác
nhau. Ứng viên đứng cao ở nhiều nhánh sẽ được ưu tiên.

Mỗi nhánh lấy top-100, sau đó chọn 50 chunk cho reranker. Trong bảng hai, kết hợp
cả ba nhánh đạt Recall@50 là 0,9387 và F2@2 là 0,5252, cao nhất trong các cấu
hình đang so sánh.

Điều này cho thấy các nhánh có tính bổ sung. Tuy nhiên, tìm được ứng viên trong
top-50 chưa có nghĩa là những điều đúng đã đứng ở đầu danh sách. Đó là lý do cần
tầng reranking.

[Trang 20–22.]

Cross-encoder đọc đồng thời câu hỏi và từng chunk để chấm mức liên quan chi tiết
hơn. Khi nhiều chunk thuộc cùng một điều luật, nhóm lấy điểm cao nhất làm điểm
của điều đó.

Về fine-tuning, dữ liệu chỉ có nhãn ở cấp Điều, chưa chỉ rõ chunk nào chứa bằng
chứng. Vì vậy, nhóm xem các chunk của một điều như một nhóm mẫu, lấy logit cao
nhất để học nhãn liên quan của điều bằng pointwise MIL BCE.

Mẫu dương đến từ điều luật gold; mẫu âm khó là những điều không thuộc gold nhưng
dễ bị mô hình nhầm là liên quan. Việc bổ sung gold được thực hiện khi tạo mẫu
huấn luyện, không sử dụng nhãn để hỗ trợ suy luận trên test.

Bảng ba cho thấy F2@2 tăng từ 0,5252 của truy xuất lai lên 0,5853 khi thêm
cross-encoder chưa fine-tune, và đạt 0,5967 sau fine-tune epoch một. Như vậy,
reranking và thích nghi mô hình với dữ liệu đều có ích trong so sánh này.

[Trang 23–25.]

Sau reranker, nhóm tiếp tục đặt câu hỏi: có thể kết hợp thêm thông tin từ các
nhánh truy xuất để xếp hạng tốt hơn không?

LightGBM nhận các đặc trưng của cặp câu hỏi và điều luật, gồm điểm, thứ hạng,
sự đồng thuận giữa các nhánh, thống kê chunk và mức khớp từ. Mô hình ước lượng
mức liên quan rồi sắp xếp lại các điều ứng viên.

Trong bảng bốn, enhanced v2 với 57 đặc trưng đạt F2@2 là 0,6217, tốt nhất trong
các schema được so sánh. Cấu hình 65 đặc trưng không tốt hơn, nên thêm đặc trưng
không đồng nghĩa với cải thiện kết quả.

Biểu đồ tiếp theo cũng cho thấy cần chọn checkpoint theo toàn pipeline. Epoch
ba tốt nhất khi chỉ xét reranker, nhưng epoch một tốt nhất sau LightGBM ở cùng
đầu ra top-2. Đây là cơ sở hỗ trợ lựa chọn epoch một cho pipeline đang trình bày.

[Trang 26–27.]

Đến đây, danh sách đã được xếp hạng, nhưng vẫn còn quyết định: nên lấy bao nhiêu
điều cho mỗi câu hỏi?

Nhóm so sánh bốn chính sách: lấy top-k cố định, dừng theo khoảng cách điểm,
learned stopping và tổ hợp Multiclass–Ridge. Learned stopping học quyết định
lấy thêm hay dừng; Multiclass–Ridge đánh giá các lựa chọn từ top-một đến top-năm.
Các chính sách học được sử dụng đánh giá theo fold để hạn chế việc học và đánh
giá trên cùng câu hỏi.

Bảng năm ghi nhận Multiclass–Ridge đạt F2 cao nhất là 0,6910, còn learned stopping
đạt Recall cao nhất là 0,7303. Kết quả cho thấy số điều đầu ra là một thành phần
quan trọng, bên cạnh việc tìm và xếp hạng ứng viên.

[Trang 28.]

Cuối cùng, trên Codabench, hệ thống đạt Macro-F2 là 0,6978 ở public và 0,6825 ở
private. Bảng này dùng để đối chiếu kết quả toàn hệ thống với các phương pháp
khác, không phải một ablation kiểm soát từng thành phần hay bảng thứ hạng chính
thức do nhóm tự xác định.

Như vậy, nhóm đã xem xét cả cách biểu diễn, truy xuất, xếp hạng và số điều đầu
ra. Em xin chuyển phần cuối cho bạn thứ ba để tổng kết kết quả và minh họa ứng dụng.

## Người 3 — Phần 4 và demo

[Trang 29–30.]

Cảm ơn bạn. Từ các thực nghiệm vừa trình bày, nhóm rút ra ba nhận xét chính.

Thứ nhất, giữ thông tin của điều luật bằng Article window và kết hợp các nhánh
truy xuất giúp tìm ứng viên đầy đủ hơn trong các cấu hình đã so sánh.

Thứ hai, cross-encoder và LightGBM hỗ trợ đưa những điều liên quan lên cao hơn.
Thứ ba, điều chỉnh số lượng đầu ra theo câu hỏi giúp cân bằng giữa tìm đủ bằng
chứng và hạn chế điều luật thừa.

Các mốc trên slide mô tả quá trình hoàn thiện hệ thống. Vì có những khác biệt
về cấu hình và chính sách đánh giá, nhóm không xem hiệu số giữa mọi mốc là mức
đóng góp độc lập của từng thành phần.

Hệ thống vẫn còn khoảng cách với một số phương pháp trong bảng đối chiếu. Các
hướng có thể kiểm chứng tiếp gồm làm giàu tiêu đề, đặc trưng bằng LLM và tăng
cường dữ liệu. Tuy nhiên, nhóm chưa xác định các khác biệt này giải thích bao
nhiêu chênh lệch điểm.

[Trang 31. Chuyển sang ứng dụng với hội thoại đã chạy sẵn.]

Để minh họa ứng dụng, nhóm xây dựng giao diện hỏi đáp pháp luật Việt Nam. Ở đây,
pipeline vừa trình bày tìm và chọn điều luật, sau đó LLM sẵn có tổng hợp câu trả
lời từ các bằng chứng đó.

Trong hội thoại minh họa này, câu trả lời có mã nguồn đi kèm. Khi bấm vào mã,
người dùng có thể mở và đối chiếu nguyên văn điều luật ở bảng bên cạnh. Phần
nguồn mặc định đóng để giữ nội dung hội thoại dễ theo dõi.

[Bấm một trích dẫn và mở điều luật tương ứng. Không đọc toàn bộ câu trả lời.]

Điều nhóm muốn nhấn mạnh là người dùng có thể kiểm tra căn cứ phía sau câu trả
lời. Demo cho thấy một cách sử dụng pipeline, còn trọng tâm đánh giá của đồ án
vẫn là chất lượng truy xuất. Nhóm chưa đánh giá định lượng đầy đủ tầng sinh,
và kết quả truy xuất không tự bảo đảm mọi câu trả lời đều chính xác.

[Trang 32.]

Đến đây, nhóm xin kết thúc phần trình bày. Chúng em cảm ơn thầy và các bạn đã
lắng nghe, và mong nhận được câu hỏi cùng góp ý để tiếp tục hoàn thiện đề tài.
