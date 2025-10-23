Transformer for English–Vietnamese Translation

PyTorch implementation of a Transformer model for machine translation from English to Vietnamese.

🔍 Giới thiệu

Dự án này hiện thực hóa mô hình Transformer — một kiến trúc dựa hoàn toàn trên cơ chế Attention — để dịch câu từ tiếng Anh sang tiếng Việt.
Mô hình được xây dựng từ đầu bằng PyTorch, bao gồm đầy đủ các thành phần: Encoder, Decoder, Positional Encoding, Multi-Head Attention và cơ chế Masking trong quá trình huấn luyện và dịch tự động.

🧠 Kiến trúc Transformer
1. Kiến trúc tổng thể

Transformer hoạt động theo mô hình encoder–decoder, trong đó:

Encoder mã hóa câu nguồn (tiếng Anh) thành các vector ngữ nghĩa.

Decoder giải mã các vector này để sinh ra câu đích (tiếng Việt).

Quy trình tổng quát:
Input → Token Embedding → Positional Encoding → Encoder Stack → Decoder Stack → Output

2. Multi-Head Attention

Cơ chế Attention cho phép mô hình tập trung vào các phần quan trọng của câu ở mọi vị trí.
Với Multi-Head Attention, mỗi "head" học một khía cạnh khác nhau của ngữ cảnh:

Head 1: Quan hệ ngữ pháp

Head 2: Quan hệ ngữ nghĩa

Head 3: Thông tin vị trí

Head 4: Tổng hợp ngữ cảnh

Điều này giúp mô hình hiểu sâu hơn về cấu trúc ngôn ngữ và ý nghĩa toàn cục.

3. Positional Encoding

Vì Transformer không dùng RNN nên cần bổ sung thông tin thứ tự vị trí cho từng token.
Positional Encoding được tính bằng các hàm sin và cos ở các tần số khác nhau, đảm bảo mô hình phân biệt được thứ tự các từ trong câu.

4. Encoder Stack

Mỗi lớp trong encoder gồm hai khối chính:

Multi-Head Self-Attention

Feed Forward Network

Mỗi khối đều có Residual Connection và Layer Normalization, giúp ổn định gradient và cải thiện khả năng hội tụ của mô hình.

5. Decoder Stack

Decoder bao gồm ba thành phần:

Masked Self-Attention: chỉ cho phép mô hình nhìn thấy các từ trước đó khi sinh từ tiếp theo.

Encoder–Decoder Attention: kết hợp thông tin từ encoder để hiểu ngữ cảnh nguồn.

Feed Forward Network: biến đổi đặc trưng phi tuyến tính.

Cơ chế masking được áp dụng để tránh mô hình “nhìn trước” các từ trong tương lai khi dịch.

6. Feed Forward Network

Là mạng hai lớp tuyến tính hoạt động độc lập tại từng vị trí token.
Giúp mô hình tăng khả năng biểu diễn phi tuyến và học được quan hệ phức tạp giữa các từ.

7. Quá trình Dịch (Inference)

Transformer dịch câu theo phương pháp autoregressive — sinh từng từ một, dựa vào các từ đã sinh trước đó.
Ví dụ:

Bước 1: “xin” → “chào”

Bước 2: “xin chào” → “bạn”

Bước 3: “xin chào bạn” → “có khỏe không”

📊 Kết quả Huấn luyện

Final Training Loss: 0.8670

Final Validation Loss: 0.8911

Training Perplexity: 2.38

Validation Perplexity: 2.44

Best Model: Epoch 19 (val_loss: 0.8843)

Tổng số tham số: 958,299

📚 Dữ liệu Huấn luyện

Số lượng câu song ngữ: 20 cặp câu Anh–Việt

Từ vựng nguồn (English): 72 từ

Từ vựng đích (Vietnamese): 91 từ

⚙️ Tính năng & Kỹ thuật

Label Smoothing: 0.1

Cosine Annealing Learning Rate

AdamW Optimizer with Weight Decay

Gradient Clipping

CrossEntropyLoss (bỏ qua padding)

🚀 Ưu điểm của Transformer

Song song hóa: Xử lý toàn bộ chuỗi cùng lúc, tăng tốc huấn luyện.

Hiểu phụ thuộc dài hạn: Cơ chế Self-Attention giúp mô hình nắm bắt các quan hệ xa trong câu.

Hiệu quả tính toán: Giảm độ phức tạp so với RNN truyền thống.
