# Kịch bản nói demo hỏi đáp pháp luật Việt Nam

Dùng sau slide 31, do người 3 trình bày. Khoảng 2 phút lời nói, chưa đo qua diễn
tập và chưa tính chờ API/suy luận. Đây là bản demo mở rộng, không cộng dồn vào
đoạn demo 20–30 giây của kịch bản thuyết trình 10 phút trước đó. Nếu tổng thời
gian bị giới hạn ở 10 phút, cần rút phần khác hoặc dùng đoạn ngắn cuối tài liệu.

Chuẩn bị trước: chạy một câu để kiểm tra pipeline/API, chuẩn bị hội thoại đã
có kết quả và giữ phần nguồn đóng. Nếu dùng kết quả chuẩn bị sẵn, nói rõ đó là
hội thoại đã chạy trước. Không hứa mức phạt hay một câu trả lời pháp lý cụ thể;
chỉ giới thiệu những kết quả thực sự hiện trên giao diện. Nguồn trong corpus
không tự xác nhận hiệu lực hiện tại của văn bản.

Các dòng trong ngoặc vuông là thao tác, không đọc thành lời.

## 1. Chuyển từ slide sang ứng dụng

[Mở giao diện ứng dụng.]

“Sau phần đánh giá truy xuất, em xin minh họa cách nhóm đưa pipeline vào một ứng
dụng hỏi đáp pháp luật Việt Nam. Người dùng có thể đặt câu hỏi bằng ngôn ngữ tự
nhiên, xem câu trả lời và mở các điều luật được sử dụng làm bằng chứng.”

## 2. Đặt câu hỏi pháp luật

[Tạo hội thoại mới. Nhập: Xe máy vượt đèn đỏ bị xử phạt như thế nào?]

“Ví dụ, em đặt câu hỏi: Xe máy vượt đèn đỏ bị xử phạt như thế nào?

Sau khi kiểm tra phạm vi và thông tin của tình huống, hệ thống truy xuất các
ứng viên, xếp hạng lại và chọn những điều luật liên quan. LLM sau đó được yêu
cầu tổng hợp câu trả lời từ các bằng chứng này.”

[Trong khi chờ, chỉ trạng thái xử lý.]

“Các trạng thái đang hiển thị giúp người dùng theo dõi hệ thống đang tìm điều
luật hay đang soạn câu trả lời.”

[Nếu hệ thống hỏi làm rõ, trả lời đúng một thông tin được yêu cầu hoặc bấm một
lựa chọn. Không khẳng định chắc chắn câu mẫu sẽ tạo nhánh làm rõ.]

## 3. Kiểm tra bằng chứng

[Khi có câu trả lời, không đọc toàn bộ. Bấm một trích dẫn aid thực sự có trong
câu trả lời; nếu không có trích dẫn bấm được, mở bằng nút Văn bản nguồn.]

“Ở đây, ngoài nội dung trả lời, hệ thống còn hiển thị các điều luật đã truy xuất.
Em mở phần nguồn để đối chiếu nguyên văn.

Mỗi điều có mã aid và điểm xếp hạng ở ngay đầu. Điểm này phục vụ xếp thứ tự liên
quan, không phải xác suất câu trả lời đúng. Phần nguồn mặc định đóng để giữ
khung hội thoại gọn, và chỉ mở khi người dùng muốn kiểm tra.”

## 4. Minh họa bộ nhớ hội thoại

[Đóng bảng nguồn. Nhập: Còn ô tô thì sao?]

“Tiếp theo, em hỏi ngắn gọn: Còn ô tô thì sao?

Ứng dụng gửi tối đa năm lượt hỏi–đáp gần nhất làm ngữ cảnh để hiểu câu hỏi tiếp
nối. Hệ thống vẫn truy xuất lại bằng chứng cho câu hỏi mới, thay vì chỉ dựa vào
câu trả lời trước.”

[Quan sát kết quả thực tế; không nói rằng hệ thống đã hiểu đúng nếu màn hình
đang hỏi lại hoặc trả lỗi.]

## 5. Câu hỏi gợi ý và lưu hội thoại

[Chỉ ba câu hỏi gợi ý khi chúng đã xuất hiện. Bấm một câu nếu thời gian cho phép.]

“Dưới câu trả lời, AI gợi ý ba câu hỏi liên quan. Người dùng có thể bấm trực tiếp
để hỏi tiếp, không cần nhập lại.

Các hội thoại được lưu trên trình duyệt và có thể chuyển qua lại ở thanh bên.
Ngoài ra, ứng dụng hỗ trợ sao chép câu trả lời và xuất hội thoại kèm nguồn thành
file Markdown.”

[Nếu API gợi ý lỗi và giao diện có cảnh báo, nói: “Lượt này dịch vụ chưa tạo được
gợi ý; câu trả lời và phần nguồn vẫn được giữ lại.”]

## 6. Kết thúc demo

[Chuyển về slide 32.]

“Qua demo, nhóm muốn minh họa việc kết nối kết quả truy xuất với một giao diện
hỏi đáp có thể đối chiếu nguồn. Phần được đánh giá định lượng trong đồ án là
truy xuất bằng chứng; chất lượng sinh câu trả lời và độ chính xác pháp lý hiện
tại vẫn cần được đánh giá riêng.

Phần demo của nhóm xin kết thúc tại đây. Chúng em cảm ơn thầy và các bạn.”

## Tùy chọn: giới hạn phạm vi, nếu còn thời gian

[Tạo hội thoại riêng, hỏi Bạn là ai? rồi hỏi Gợi ý món ăn tối nay. Chỉ thực hiện
khi đã kiểm tra endpoint; phân loại ngoài phạm vi vẫn có thể gọi API.]

“Trợ lý cũng được giới hạn phạm vi sử dụng. Khi hỏi về danh tính, hệ thống giới
thiệu chức năng tra cứu pháp luật Việt Nam. Với câu hỏi ngoài phạm vi, hệ thống
được thiết kế để hướng người dùng quay lại câu hỏi pháp luật.”

Không tuyên bố bộ phân loại chặn đúng mọi câu hỏi ngoài phạm vi; chưa có đánh
giá định lượng cho tính năng này.

## Bản rút gọn khi chỉ còn 30 giây

[Mở một hội thoại đã chạy trước, bấm Văn bản nguồn.]

“Đây là một hội thoại minh họa đã được nhóm chạy trước. Pipeline tìm và chọn
những điều luật liên quan, sau đó LLM tổng hợp câu trả lời từ bằng chứng.
Người dùng có thể mở mã nguồn để đối chiếu nguyên văn điều luật, như em đang
thao tác trên màn hình. Trọng tâm đánh giá của đồ án là truy xuất; tầng sinh
câu trả lời vẫn cần được đánh giá riêng. Phần demo của nhóm xin kết thúc tại đây.”

## Câu chuyển khi phải chờ hoặc gặp lỗi

Nếu xử lý lâu: “Hiện hệ thống đang xử lý yêu cầu trên môi trường Kaggle. Để giữ
thời gian trình bày, em chuyển sang hội thoại đã chạy trước để minh họa phần
kết quả và đối chiếu nguồn.”

Nếu lỗi: “Lượt gọi trực tiếp này đang gặp lỗi xử lý. Em xin dùng hội thoại đã
chuẩn bị trước để tiếp tục minh họa giao diện; đây không phải kết quả của lượt
vừa gửi.”
