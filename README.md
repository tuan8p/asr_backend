# asr_backend
Mở terminal anaconda hoặc bất kỳ và chứa các thư viện yêu cầu sau
Run code: python main.py
## Cách xử lý
Nhận voice từ thiết bị micro, sau đó đưa qua model ASR để chuyển thành text.
Tiếp theo đưa qua model NLP để hiểu ngữ nghĩa và trả về kiểu dict với các thông tin text, text tương quan, similarity, lệnh thực thi, điều kiện thực thi nếu có (operator, loại điều kiện, value, đơn vị).
## Thông tin
Model sử dụng cho ASR là model whisper small đã pretrain và finetune lại với dataset custom từ giọng nói dùng trong smart home.
Model sử dụng cho Hiểu ngữ nghĩa là model MiniLM-L12 đã được pretrain và finetune từ Sentence Transformers team.
Dataset được làm từ Zalo AI TTS và FPT AI TTS.
Cấu trúc dataset gồm 1 folder clip chứa các audio của tập train, dev, test.
1 file validated.tsv chứa toàn bộ thông tin các audio của 3 tập; 3 file train.tsv và dev.tsv và test.tsv chứa thông tin từng tập; file clip_durations.tsv là thời gian các audio trên 3 tập và file commands.tsv chứa 50 câu lệnh dùng tỏng smart home.
Dataset có tổng 50 câu đại diện cho các lệnh dùng trong smart home.
Mỗi giọng góp mặt trong dataset đều nói đủ 50 câu.
### Train
350 audio
Giới tính nam: Bắc 1 người, Nam 1 người, Trung 1 người.
Giới tính nữ: Bắc 1 người, Nam 2 người, Trung 1 người.
### Dev
150 audio
Giới tính nam: Bắc 1 người.
Giới tính nữ: Bắc 1 người, Nam 1 người.
### Test
100 audio
Giới tính nam: Nam 1 người.
Giới tính nữ: Trung 1 người.
## Giọng nói
Khi bấm GET api http://localhost:8000/asr thì trên terminal hiện thời gian đếm ngược để ghi âm.
Thời gian ghi âm 3 giây kể từ lúc GET api.
Khi nói vào micro thì không để quá gần miệng và âm lượng khi nói không quá to vừa đủ nghe, nói tròn vành rõ từng chữ, tốc độ nói không quá nhanh, không dùng từ đặc trưng vùng miền.
## Câu lệnh không điều kiện
ví dụ:
    mở cửa
    tắt đèn
## Câu lệnh có 1 điều kiện
Bắt đầu điều kiện từ chữ "khi", "nếu", "lúc", "sau",
Trườn hợp quạt là đặc biệt không có chữ để phân biệt vế điều kiện.
### Nhiệt độ
Thang đo độ C
Nếu độ K thì trừ 273
Nóng khi > 30 độ C
Lạnh khi < 20 độ C
Cấu trúc:
<nhiệt độ|nóng|lạnh> <khoảng|trên|dưới|...>? <value> <độ c|độ k>
<nóng|lạnh>
### Độ ẩm
Thang đo %
Độ ẩm cao > 70%
Nồm (độ ẩm cực cao) > 90%
Độ ẩm thấp < 30%
<độ ẩm|nồm|khô> <khoảng|trên|dưới|...>? <value> "%"
<ẩm|nồm|khô>
### Ánh sáng
Chỉ có 2 trạng thái là
tối khi độ sáng < 20 lux
sáng khi độ sáng > 30 lux
Cấu trúc:
<sáng|tối>
### Quạt
Thang đo % và mức 1,2,3
Mức 1 tương đương 30% (yếu nhất)
Mức 2 tương đương 70%
Mức 3 tương đương 100%
Cấu trúc:
<mức | tốc độ | quay> + <khoảng|trên|dưới|...>? <value> "%"
<nhanh|chậm|...>
### Thời gian đếm ngược
Thang đo giây
Nếu nhập vào có giờ hoặc phút đều quy ra giây
Cấu trúc: <sau> <hour> giờ <min> phút <second> giây