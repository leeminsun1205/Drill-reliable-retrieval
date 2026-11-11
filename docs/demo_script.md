# Kịch bản demo đất đai — mục tiêu 3–5 phút

Dùng sau slide 31, người 3 trình bày. Phân bổ mục tiêu khoảng 4 phút 30 giây,
chưa đo qua diễn tập. Demo này thay thế bản 2 phút, không cộng thêm vào bản cũ.
Nếu phần báo cáo đã chiếm 10 phút thì tổng buổi trình bày sẽ dài hơn 10 phút.

Đã đối chiếu thao tác với giao diện React hiện tại. Không có lần suy luận thật
để xác nhận câu trả lời pháp lý hay một câu mẫu chắc chắn sẽ kích hoạt hỏi rõ.
Không đọc sẵn kết luận pháp luật; chỉ nhận xét những gì thực sự hiện trên màn hình.

## Chuẩn bị

- Chạy kiểm tra pipeline/API trước. Mở sẵn hội thoại đất đai có câu hỏi chính,
  câu hỏi tiếp nối và ba gợi ý; giữ nguồn đóng. Tạo hội thoại riêng có lời giới
  thiệu, từ chối ngoài phạm vi, và một nhánh hỏi rõ nếu đã quan sát được.
- Câu chính: “Cần chuẩn bị giấy tờ gì khi chuyển nhượng quyền sử dụng đất?”
- Câu tiếp nối: “Nếu đất đó đang có tranh chấp thì sao?”
- Câu sửa: “Nếu đất đó đang được thế chấp tại ngân hàng thì sao?”
- Hỏi chức năng: “Bạn là ai?”; ngoài phạm vi: “Gợi ý món ăn tối nay.”
- Chọn 1–2 lượt gọi trực tiếp để giữ thời gian; các nhánh còn lại có thể thao
  tác trên hội thoại đã chạy trước, phải giới thiệu rõ. Nếu mọi lượt đều chạy
  trực tiếp thì thời gian có thể vượt 5 phút do chờ API. Không đổi hội thoại
  trong cùng tab khi đang xử lý; chờ xong hoặc mở tab dự phòng cùng địa chỉ.
- Không cố tạo lỗi thật. Chỉ giới thiệu nút Gửi lại nếu có lượt lỗi đã lưu.
  Không tuyên bố điểm xếp hạng là xác suất đúng hoặc nguồn chắc chắn còn hiệu lực.

Các đoạn trong ngoặc vuông là thao tác, không đọc. Nếu dùng hội thoại chuẩn bị
sẵn thì thay “em đặt câu hỏi” bằng “đây là câu hỏi nhóm đã chạy trước”.

## 1. Mở ứng dụng — khoảng 20 giây

[Mở giao diện. Tạo cuộc trò chuyện mới. Chỉ các thẻ chủ đề.]

“Sau phần đánh giá thực nghiệm, em xin minh họa ứng dụng hỏi đáp pháp luật Việt
Nam được xây dựng từ pipeline của nhóm.

Giao diện có các câu hỏi gợi ý theo chủ đề để người dùng bắt đầu. Trong phần
này, em chọn tình huống đất đai và lần lượt minh họa việc đặt câu hỏi, kiểm tra
bằng chứng, hỏi tiếp và quản lý hội thoại.”

## 2. Câu hỏi đất đai và quá trình xử lý — khoảng 35 giây

[Bấm thẻ Đất đai, hoặc nhập câu chính.]

“Câu hỏi đầu tiên là: Cần chuẩn bị giấy tờ gì khi chuyển nhượng quyền sử dụng đất?

Sau khi nhận câu hỏi, hệ thống kiểm tra phạm vi và xem tình huống có cần làm rõ
hay không. Nếu thiếu thông tin thiết yếu, hệ thống có thể hỏi lại thay vì tự
đặt giả định. Nếu đủ thông tin, pipeline tìm các ứng viên, xếp hạng và chọn
những điều luật làm bằng chứng.”

[Chỉ trạng thái và thời gian xử lý. Nếu có câu hỏi làm rõ, chọn đúng một câu
đầy đủ trong các lựa chọn thực tế.]

“Người dùng có thể theo dõi từng trạng thái và thời gian xử lý. Khi bắt đầu có
câu trả lời, nội dung được hiển thị dần để người dùng đọc ngay.”

## 3. Đọc kết quả và đối chiếu nguồn — khoảng 35 giây

[Khi có kết quả, chỉ phần giải thích và trích dẫn. Bấm aid có thật hoặc nút
Văn bản nguồn.]

“Đây là câu trả lời hệ thống tạo từ những bằng chứng đã chọn. Em mở trích dẫn
để xem nguyên văn điều luật trong bảng bên phải.

Mỗi nguồn hiển thị mã aid và điểm xếp hạng ngay đầu. Điểm này thể hiện tín hiệu
xếp hạng mức liên quan, không phải xác suất câu trả lời đúng. Người dùng có thể
đọc và đối chiếu trực tiếp, thay vì chỉ tin vào phần diễn giải của AI.

Phần nguồn mặc định đóng; khi mở, nội dung bắt đầu ở đầu văn bản. Em đóng lại
để tiếp tục hội thoại.”

## 4. Bộ nhớ và câu hỏi tiếp nối — khoảng 25 giây

[Nhập câu tiếp nối hoặc mở hội thoại đất đai đã chạy trước có câu này.]

“Tiếp theo, em hỏi: Nếu đất đó đang có tranh chấp thì sao?

Câu hỏi không lặp lại toàn bộ tình huống. Ứng dụng gửi tối đa năm lượt hỏi–đáp
gần nhất làm ngữ cảnh để hiểu đang nói về chuyển nhượng quyền sử dụng đất.
Hệ thống vẫn tìm lại bằng chứng cho câu hỏi mới; câu trả lời cũ không được xem
là căn cứ pháp lý.”

[Chỉ kết quả thực tế. Nếu đang hỏi rõ hoặc lỗi thì không tuyên bố đã hiểu đúng.]

## 5. Gợi ý hỏi tiếp và sửa câu hỏi — khoảng 35 giây

[Chỉ ba gợi ý thực tế. Bấm một câu nếu dùng lượt live; nếu không, chuyển sang
hội thoại chuẩn bị sẵn đã có lượt hỏi từ gợi ý và nói rõ.]

“Dưới câu trả lời, AI gợi ý ba câu hỏi liên quan. Người dùng có thể bấm vào một
câu để hỏi tiếp ngay, không phải nhập lại. Câu được chọn vẫn đi qua quy trình
kiểm tra và truy xuất bằng chứng.”

[Bấm Sửa câu hỏi ở câu tiếp nối. Thay bằng câu sửa đã chuẩn bị. Có thể bấm Hủy
để chỉ minh họa thao tác; không nói đã gửi lại nếu thực tế đã hủy.]

“Nếu muốn thay đổi tình huống, người dùng có thể sửa câu hỏi, chẳng hạn đổi từ
đất đang tranh chấp sang đất đang thế chấp. Khi gửi bản sửa, hệ thống bỏ các
lượt phía sau và xây dựng lại ngữ cảnh từ điểm này để không trộn hai tình huống.”

## 6. Lưu và quản lý hội thoại — khoảng 40 giây

[Bấm Hủy nếu đang sửa. Đổi tên hội thoại thành Tra cứu đất đai. Bấm Sao chép,
rồi Xuất .md. Chuyển sang một hội thoại khác và quay lại.]

“Ở thanh bên, mỗi cuộc trò chuyện được quản lý riêng. Em có thể đổi tên thành
Tra cứu đất đai và chuyển qua lại giữa các cuộc trò chuyện; ngữ cảnh gửi lên
chỉ lấy từ hội thoại đang chọn.

Câu trả lời có thể sao chép, còn toàn bộ hội thoại kèm văn bản nguồn có thể
xuất thành file Markdown. Lịch sử được lưu trên trình duyệt để mở lại, không
phải đăng nhập vào một tài khoản của ứng dụng.”

[Tải lại trang nếu phiên demo đã ổn định và đúng địa chỉ Ngrok; chọn lại hội
thoại nếu cần. Sau đó tạo hội thoại trống, mở xác nhận xóa và xóa hội thoại
trống đó, tránh xóa bản demo đã chuẩn bị.]

“Khi tải lại cùng địa chỉ, các hội thoại vẫn được lưu. Người dùng cũng có thể
tạo cuộc trò chuyện mới để bắt đầu với ngữ cảnh trống hoặc xóa hội thoại không
còn cần; ứng dụng yêu cầu xác nhận trước khi xóa.”

## 7. Giới hạn phạm vi và tiện ích đọc — khoảng 30 giây

[Mở hội thoại kiểm tra phạm vi đã chạy trước và nói rõ. Hiển thị lượt Bạn là ai?
và Gợi ý món ăn tối nay.]

“Trong hội thoại đã chuẩn bị này, khi hỏi Bạn là ai, trợ lý giới thiệu chức năng
hỗ trợ tra cứu pháp luật Việt Nam. Với yêu cầu gợi ý món ăn, hệ thống hướng
người dùng quay lại phạm vi pháp luật. Đây là hành vi được thiết kế cho demo,
không phải khẳng định bộ phân loại luôn đúng.”

[Quay về hội thoại đất đai dài. Cuộn lên, chỉ nút Về tin nhắn mới nhất rồi bấm.
Nếu có lượt lỗi đã lưu thì chỉ nút Gửi lại, không cố tạo sự cố.]

“Khi cuộn lên đọc lại, giao diện ngừng tự kéo xuống; người dùng có thể quay về
tin mới nhất bằng nút này. Nếu một lượt xử lý gặp lỗi, ứng dụng giữ thông báo
và cho phép gửi lại. Bố cục cũng hỗ trợ màn hình nhỏ với bảng nguồn dạng phủ.”

## 8. Kết thúc — khoảng 20 giây

[Chuyển về slide 32.]

“Qua demo, nhóm muốn minh họa cách kết nối pipeline truy xuất với một giao diện
hỏi đáp có thể đối chiếu nguồn và tiếp tục theo ngữ cảnh.

Trọng tâm đánh giá định lượng của đồ án vẫn là chất lượng truy xuất. Chất lượng
sinh câu trả lời và độ chính xác pháp lý hiện tại cần được đánh giá riêng.
Phần demo của nhóm xin kết thúc tại đây. Chúng em cảm ơn thầy và các bạn.”

## Câu chuyển khi chậm hoặc lỗi

“Lượt xử lý trực tiếp đang mất thêm thời gian. Để giữ thời lượng trình bày, em
chuyển sang hội thoại đã chạy trước để minh họa kết quả và đối chiếu nguồn.”

Nếu lỗi, nói rõ: “Lượt vừa gửi đang gặp lỗi xử lý. Hội thoại em mở sau đây là
kết quả đã chuẩn bị trước, không phải kết quả của lượt vừa gửi.”
