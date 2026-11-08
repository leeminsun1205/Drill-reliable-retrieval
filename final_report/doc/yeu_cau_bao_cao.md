# HƯỚNG DẪN VÀ YÊU CẦU CHI TIẾT VIẾT BÁO CÁO BÀI TẬP LỚN / ĐỒ ÁN MÔN HỌC
## MÔN HỌC: HỌC MÁY TIÊN TIẾN VÀ ỨNG DỤNG (ADVANCED MACHINE LEARNING & APPLICATIONS)
### Tổng quy mô báo cáo: ~15 Trang (Chuẩn định dạng Báo cáo Khoa học Academic / LNCS / IEEE)

---

## I. MỤC TIÊU VÀ YÊU CẦU CHUNG

1. **Mục tiêu:**
   - Vận dụng các kiến thức lý thuyết đã học trong môn Học máy tiên tiến (từ Tổng quan ML/DL, Tiền xử lý dữ liệu, Các mô hình phân loại/hồi quy tuyến tính, Cây quyết định & Ensemble Learning, Phân cụm Unsupervised, đến Mạng nơ-ron MLP & Deep Learning) vào việc giải quyết một bài toán thực tế cụ thể.
   - Rèn luyện kỹ năng nghiên cứu khoa học, xây dựng mô hình, thực nghiệm đánh giá, so sánh và viết báo cáo thuật toán chuẩn academic (dựa trên cấu trúc bài báo khoa học chuẩn như định dạng trong `huongdanvietbai.pdf`).

2. **Quy định chung về định dạng:**
   - **Độ dài:** Khuyến nghị khoảng **15 trang A4** (không tính phụ lục).
   - **Font chữ & Định dạng:** Times New Roman (12pt) hoặc Calibri/Arial, giãn dòng 1.15 - 1.5, lề Trái 3cm, Phải 2cm, Trên 2.5cm, Dưới 2.5cm.
   - **Cấu trúc chuẩn:** Áp dụng cấu trúc bài báo nghiên cứu ứng dụng: Abstract $\rightarrow$ Introduction $\rightarrow$ Related Work $\rightarrow$ Methodology $\rightarrow$ Experiments & Results $\rightarrow$ Conclusion $\rightarrow$ References.
   - **Hình vẽ & Bảng biểu:** Tất cả hình vẽ (Figures) và Bảng (Tables) phải có đánh số, chú thích rõ ràng và được trích dẫn trực tiếp trong nội dung văn bản.
   - **Công thức toán học:** Viết dưới dạng Math Standard (LaTeX / MS Equation editor), đánh số thứ tự công thức nằm bên phải.

---

## II. BỐ CỤC PHÂN PHỔ NỘI DUNG THEO TỪNG TRANG (~15 TRANG)

Below là bảng phân bổ dung lượng chi tiết cho từng phần để đảm bảo tính cân đối và chuyên sâu:

| Số trang | Tên phần / Chương | Nội dung cốt lõi cần trình bày |
| :---: | :--- | :--- |
| **Trang 1** | **Trang Bìa & Thông tin chung** | Tên trường/khoa, Tên môn học, Tên đề tài, Họ tên sinh viên, Mã số sinh viên, Giảng viên hướng dẫn, Ngày tháng năm. |
| **Trang 2** | **Tóm tắt (Abstract), Từ khóa & Mục lục** | **Abstract** (200 - 300 từ): Tóm tắt tóm lược bối cảnh, bài toán, phương pháp đề xuất, kết quả thực nghiệm đạt được. **Keywords**: 4-6 từ khóa cốt lõi. **Mục lục** tự động. |
| **Trang 3 - 4** | **Chương 1: Giới thiệu & Đặt vấn đề (Introduction)** | **1.1 Bối cảnh & Lý do chọn đề tài:** Ý nghĩa thực tiễn và tính cấp thiết của bài toán. <br>**1.2 Phát biểu bài toán:** Đầu vào (Input), Đầu ra (Output) và thách thức kỹ thuật. <br>**1.3 Mục tiêu & Đóng góp chính:** Liệt kê các đóng góp chính của báo cáo (ví dụ: đề xuất cải tiến, so sánh mô hình, đạt hiệu năng cao). <br>**1.4 Cấu trúc báo cáo:** Tóm tắt ngắn gọn nội dung các phần tiếp theo. |
| **Trang 5 - 6** | **Chương 2: Cơ sở lý thuyết & Nghiên cứu liên quan (Related Work)** | **2.1 Tổng quan kiến thức nền tảng:** Hệ thống các khái niệm ML/DL liên quan (Regression, Decision Trees, Random Forest, Boosting, MLP, v.v.). <br>**2.2 Tổng quan nghiên cứu liên quan:** Phân tích 3 - 5 bài báo/mô hình đã có trên thế giới về cùng bài toán, chỉ ra điểm mạnh và hạn chế của họ. |
| **Trang 7 - 9** | **Chương 3: Phương pháp đề xuất (Proposed Methodology)** | **3.1 Quy trình tổng thể (Pipeline):** Sơ đồ khối kiến trúc hệ thống từ dữ liệu thô đến kết quả dự đoán. <br>**3.2 Tiền xử lý & Trích xuất đặc trưng:** Chuẩn hóa dữ liệu (Min-Max/Standardization), mã hóa, xử lý Outliers/Missing values, Feature Selection hoặc giảm chiều PCA. <br>**3.3 Chi tiết Mô hình & Kiến trúc:** Trình bày chi tiết công thức toán học, hàm kích hoạt (Activation functions), sơ đồ tầng mạng (Layer architecture). <br>**3.4 Hàm mất mát (Loss Function) & Tối ưu hóa:** Mô tả chi tiết hàm Mất mát (MSE, Cross-Entropy, Custom Loss) và thuật toán tối ưu (Adam, SGD). |
| **Trang 10 - 13** | **Chương 4: Thực nghiệm & Đánh giá kết quả (Experiments & Results)** | **4.1 Tập dữ liệu (Dataset):** Mô tả nguồn dữ liệu, kích thước, phân bố tập Train/Val/Test. <br>**4.2 Thiết lập thực nghiệm:** Môi trường phần cứng/phần mềm, danh sách Siêu tham số (Hyperparameters). <br>**4.3 Độ đo đánh giá (Evaluation Metrics):** Công thức toán và ý nghĩa của các độ đo (Accuracy, Precision, Recall, F1-score, ROC-AUC, MSE, SDR, SIR, SAR,...). <br>**4.4 Kết quả so sánh & Biểu đồ:** Bảng so sánh kết quả giữa phương pháp đề xuất với các mô hình Baseline. Biểu đồ Loss/Accuracy qua các Epochs, Confusion Matrix. <br>**4.5 Phân tích & Thảo luận (Discussion):** Giải thích tại sao mô hình thành công/thất bại, phân tích Ablation Study (ảnh hưởng của từng thành phần). |
| **Trang 14** | **Chương 5: Kết luận & Hướng phát triển (Conclusion & Future Work)** | **5.1 Tóm tắt kết quả:** Tổng kết các mục tiêu đã hoàn thành và hiệu năng đạt được. <br>**5.2 Hạn chế của đề tài:** Các điểm chưa tối ưu về dữ liệu hoặc thời gian tính toán. <br>**5.3 Hướng phát triển tương lai:** Đề xuất các cải tiến về kiến trúc hoặc mở rộng ứng dụng. |
| **Trang 15** | **Tài liệu tham khảo (References) & Phụ lục** | Trích dẫn tài liệu chuẩn IEEE/LNCS (tối thiểu 8 - 15 bài báo/sách uy tín). Phụ lục link Github Code/Demo (nếu có). |

---

## III. HƯỚNG DẪN CHI TIẾT TỪNG CHƯƠNG VÀ CÁC TIÊU CHÍ ĐÁNH GIÁ

### 1. Trang Bìa và Trang Tóm Tắt (Trang 1 - 2)
- **Tóm tắt (Abstract):** Cần viết ngắn gọn trong 1 đoạn văn (khoảng 200 - 300 từ) theo công thức:
  1. *Bối cảnh:* 1-2 câu về lý do bài toán quan trọng.
  2. *Vấn đề:* 1 câu về khó khăn/hạn chế của các phương pháp hiện tại.
  3. *Giải pháp:* Mô tả mô hình/phương pháp bạn áp dụng trong đồ án.
  4. *Kết quả:* Nêu con số cụ thể thể hiện độ chính xác hoặc hiệu năng vượt trội đạt được.

### 2. Chương 1: Giới thiệu & Đặt vấn đề (Trang 3 - 4)
- Phải làm rõ được **Bài toán là gì?** (Classification, Regression, Clustering, hay Multi-task Learning).
- Nêu rõ **Đầu vào (Input)** là gì (Ảnh, Văn bản, Âm thanh, Dữ liệu bảng - Tabular Data) và **Đầu ra (Output)** là gì.
- Liệt kê 2-3 **Đóng góp chính (Key Contributions)** dưới dạng bullet points để người chấm thi thấy rõ công sức thực hiện.

### 3. Chương 2: Cơ sở lý thuyết & Nghiên cứu liên quan (Trang 5 - 6)
- Trích dẫn các mô hình gốc (Background Models): Ví dụ nếu dùng Ensemble Learning thì trích dẫn tư tưởng Random Forest/XGBoost; nếu dùng Mạng nơ-ron thì mô tả kiến trúc MLP / Backpropagation.
- Xây dựng **Bảng so sánh các nghiên cứu trước đây** (Literature Review Table) gồm các cột: *Tác giả/Năm - Phương pháp - Tập dữ liệu - Độ chính xác - Hạn chế*.

### 4. Chương 3: Phương pháp đề xuất (Trang 7 - 9) *(Trọng tâm lý thuyết & thuật toán)*
- **Vẽ Sơ đồ khối (Architecture Diagram):** Thể hiện rõ dòng chảy dữ liệu (Dataflow) từ Input $\rightarrow$ Preprocessing $\rightarrow$ Model Layers $\rightarrow$ Loss Computation $\rightarrow$ Output.
- **Biểu diễn Toán học:** Viết rõ công thức biến đổi của từng bước. 
  - Ví dụ: Công thức chuẩn hóa $x' = \frac{x - \mu}{\sigma}$, công thức hàm kích hoạt $\text{ReLU}(z) = \max(0, z)$, hoặc công thức hàm Mất mát $\mathcal{L}$.

### 5. Chương 4: Thực nghiệm & Đánh giá kết quả (Trang 10 - 13) *(Trọng tâm thực hành)*
- **Trực quan hóa Dữ liệu (Data Visualization):** Đưa vào biểu đồ phân bố nhãn (Class distribution), biểu đồ ma trận tương quan (Correlation heatmap).
- **Bảng so sánh Kết quả (Performance Comparison Table):** So sánh ít nhất 2 - 3 thuật toán baseline khác nhau (ví dụ: Logistic Regression vs Random Forest vs MLP).
- **Biểu đồ thực nghiệm:** Biểu đồ đường biểu diễn Loss/Accuracy qua các vòng lặp huấn luyện (Epochs) để phân tích hiện tượng Overfitting/Underfitting.
- **Phân tích lỗi (Error Analysis / Confusion Matrix):** Giải thích lý do tại sao mô hình đoán sai ở một số mẫu cụ thể.

### 6. Chương 5 & Tài liệu tham khảo (Trang 14 - 15)
- **Tài liệu tham khảo:** Viết đúng định dạng chuẩn: `[Số thứ tự] Tên tác giả, "Tên bài báo", Tên tạp chí/Hội nghị, Năm xuất bản.`
  *Ví dụ:* `[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in CVPR, pp. 770–778, 2016.`

---

## IV. ĐIỂM CỘNG ĐẶC BIỆT KHI CHẤM ĐỒ ÁN (BONUS MARKS)

1. **Có phần so sánh Ablation Study:** Đánh giá vai trò của từng thành phần (ví dụ: So sánh mô hình khi CÓ và KHÔNG CÓ bước chuẩn hóa PCA/Standardization, hoặc khi thay đổi hàm kích hoạt).
2. **Mã nguồn sạch sẽ:** Có liên kết Github đính kèm mã nguồn chạy hoàn chỉnh (Jupyter Notebook / Python script) có comment rõ ràng.
3. **Trình bày đẹp mắt:** Hình vẽ sắc nét (độ phân giải cao), bảng biểu căn chỉnh ngay ngắn, không lỗi chính tả.

---
*Chúc bạn hoàn thành bài báo cáo môn Học máy tiên tiến và ứng dụng đạt kết quả cao nhất!*
