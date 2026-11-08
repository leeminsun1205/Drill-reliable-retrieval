# Times New Roman cho báo cáo

Bộ phông Microsoft Times New Roman tải từ gói Core Fonts `times32.exe`:
https://downloads.sourceforge.net/corefonts/times32.exe

Bốn tệp `Times.TTF`, `Timesbd.TTF`, `Timesi.TTF`, `Timesbi.TTF` được đặt tại
thư mục này để XeLaTeX nạp trực tiếp. Các tệp phông được Git bỏ qua, không
phân phối trong mã nguồn. Giữ nguyên các tệp nếu biên dịch lại trên máy này.

Cài lại khi cần, từ thư mục `final_report/` (cần curl và cabextract):

```bash
curl --fail --location --output /tmp/drill-times32.exe https://downloads.sourceforge.net/corefonts/times32.exe
cabextract -F '*.TTF' -d fonts /tmp/drill-times32.exe
```
