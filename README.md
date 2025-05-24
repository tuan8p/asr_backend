# asr_backend
Run code: python main.py
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