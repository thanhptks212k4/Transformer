Transformer for English-Vietnamese Translation
PyTorch implementation of Transformer model for machine translation from English to Vietnamese.

Chi tiết Cơ chế Transformer
1. Kiến trúc Tổng thể
Transformer sử dụng kiến trúc encoder-decoder hoàn toàn dựa trên cơ chế attention:

text
Input → Token Embedding → Positional Encoding → Encoder Stack → Decoder Stack → Output
2. Multi-Head Attention
Công thức cơ bản:

text
Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V
Multi-Head:

python
# Trong code:
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,        # embedding dimension
    nhead=4,            # 4 attention heads
    dim_feedforward=512,
    dropout=0.1,
    batch_first=True
)
Mỗi head học các representation khác nhau:

Head 1: Quan hệ ngữ pháp

Head 2: Quan hệ ngữ nghĩa

Head 3: Quan hệ vị trí

Head 4: Kết hợp tổng hợp

3. Positional Encoding
Vì Transformer không có RNN nên cần cung cấp thông tin vị trí:

python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
Ví dụ với d_model=4:

text
Vị trí 0: [sin(0), cos(0), sin(0/100), cos(0/100)]
Vị trí 1: [sin(1), cos(1), sin(1/100), cos(1/100)]
4. Encoder Stack
Mỗi encoder layer gồm:

text
Input → Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm → Output
Residual Connection & LayerNorm:

python
# Sub-layer 1: Self-Attention
x = x + LayerNorm(MultiHeadAttention(x))

# Sub-layer 2: Feed Forward  
x = x + LayerNorm(FFN(x))
5. Decoder Stack
Decoder có 3 sub-layers:

Masked Self-Attention: Chỉ nhìn thấy các token trước đó

Encoder-Decoder Attention: Kết hợp thông tin từ encoder

Feed Forward Network

Masking trong decoder:

python
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

# Ví dụ với seq_len=3:
# [[0, -inf, -inf],
#  [0,    0, -inf], 
#  [0,    0,    0]]
6. Feed Forward Network
python
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
Trong code:

python
dim_feedforward=512  # Mở rộng từ 128 → 512
activation='relu'    # Non-linearity
7. Quá trình Dịch (Inference)
Autoregressive decoding:

text
Bước 1: "xin" → Decoder → "chào"
Bước 2: "xin chào" → Decoder → "bạn"  
Bước 3: "xin chào bạn" → Decoder → "có"
...
Training Results
Final Training Loss: 0.8670

Final Validation Loss: 0.8911

Training Perplexity: 2.38

Validation Perplexity: 2.44

Best Model: Epoch 19 (val_loss: 0.8843)

Model Parameters: 958,299

Dataset
20 English-Vietnamese sentence pairs

Source vocabulary: 72 words

Target vocabulary: 91 words

Training Features
Label smoothing (0.1)

Cosine annealing learning rate

AdamW optimizer with weight decay

Gradient clipping

CrossEntropyLoss with padding ignore

Usage
python
# Training
model = ImprovedTransformer(src_vocab_size=72, tgt_vocab_size=91)
trainer = OptimizedTrainer(model, train_loader, val_loader, device, pad_idx)
trainer.train(num_epochs=25)

# Translation
translator = ImprovedTranslator(model, src_vocab, tgt_vocab, device)
translation = translator.translate("hello how are you")
print(translation)  # "xin chào bạn có khỏe không"
Ưu điểm của Transformer
Parallelization: Xử lý toàn bộ sequence cùng lúc

Long-range dependencies: Self-attention nắm bắt phụ thuộc xa

Computational efficiency: Giảm số phép tính so với RNN

Scalability: Dễ dàng mở rộng model size

Project Structure
model.py: Transformer implementation

trainer.py: Training logic

translator.py: Inference

data_utils.py: Dataset handling

main.py: Main script

Requirements
PyTorch

Matplotlib

NumPy
