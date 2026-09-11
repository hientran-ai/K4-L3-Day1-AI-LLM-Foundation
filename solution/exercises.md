# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 1.5, các phản hồi có xu hướng chuyển từ ổn định, trực tiếp và dễ lặp lại sang đa dạng, giàu cách diễn đạt hơn. Ở 1.5, model có thể nêu sự thật bất ngờ hơn nhưng cũng dễ lan man hoặc kém nhất quán; temperature chỉ điều khiển cách lấy mẫu nên không bảo đảm mỗi lần gọi sẽ khác nhau.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi chọn temperature khoảng 0.2 cho chatbot hỗ trợ khách hàng. Mức thấp giúp câu trả lời nhất quán, bám sát chính sách và hạn chế việc tự sáng tạo thông tin, trong khi vẫn cho phép diễn đạt tự nhiên hơn mức 0.0.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload tạo 10.000 × 3 × 350 = 10,5 triệu output token/ngày. Theo bảng giá của bài, chi phí output là khoảng 105 USD cho GPT-4o và 6,30 USD cho mini, nên GPT-4o đắt hơn khoảng 16,67 lần (chưa tính input, chênh 98,70 USD/ngày). GPT-4o xứng đáng cho ca phân tích khiếu nại phức tạp, giá trị cao; mini phù hợp cho FAQ, phân loại yêu cầu hoặc tóm tắt ngắn với lưu lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Persona giáo viên tiểu học thường cho câu trả lời ngắn, dùng từ quen thuộc và ví blockchain như một cuốn sổ mà nhiều bạn cùng giữ bản sao. Persona chuyên gia tài chính có xu hướng dài và chính xác kỹ thuật hơn, nhắc đến sổ cái phân tán, cơ chế đồng thuận, hàm băm và tính bất biến. Như vậy, system prompt không đổi câu hỏi cốt lõi nhưng định hướng độ sâu, từ vựng, ví dụ và đối tượng mà model nhắm tới.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn thử nghiệm tiếng Việt 111 từ, công thức thô cho 111 / 0,75 = 148 token, còn `count_tokens` của GPT-4o đếm được 132 token; tiktoken thấp hơn ước lượng 16 token, tương đương khoảng 10,8%. Con số chính xác phụ thuộc nội dung và tokenizer; tiếng Việt có thể tốn nhiều token hơn tiếng Anh cùng độ dài vì dấu thanh, ký tự Unicode và các âm tiết cách nhau bằng khoảng trắng có thể bị tokenizer tách thành nhiều mảnh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất với câu trả lời dài hoặc model có độ trễ cao, chẳng hạn trợ lý hội thoại hay sinh nội dung, vì người dùng nhìn thấy phần đầu ngay và cảm nhận hệ thống phản hồi nhanh. Non-streaming phù hợp hơn khi kết quả ngắn, phải nhận đủ trước khi validate/parse JSON, kiểm duyệt nội dung, hoặc khi client muốn logic tích hợp đơn giản.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giãn nhanh khoảng cách giữa các lần thử, nhờ đó giảm tải cho API và cho server thêm thời gian hồi phục, trong khi lỗi thoáng qua vẫn được thử lại sớm. Nếu hàng nghìn client cùng dùng delay cố định, chúng có thể thử lại đồng thời theo từng “đợt sóng” (thundering herd), tiếp tục làm server quá tải. Trong sản phẩm thật nên thêm jitter ngẫu nhiên vào backoff để các client không còn đồng bộ.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt: “Bạn là trợ giảng thân thiện của khóa AI, giải thích chính xác, ngắn gọn bằng tiếng Việt; dùng một ví dụ đơn giản khi khái niệm khó và nói rõ khi bạn không chắc chắn.” Cụm “ngắn gọn bằng tiếng Việt” giữ câu trả lời dễ đọc và nhất quán với người học; yêu cầu nói rõ sự không chắc chắn giúp hạn chế cách trình bày thông tin suy đoán như một sự thật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ giữ ba lượt gần nhất, nên ngữ cảnh quan trọng ở đầu cuộc trò chuyện bị mất và chưa có bộ nhớ dài hạn. Tôi sẽ cải thiện bằng cách tóm tắt những message sắp bị loại thành một memory ngắn, lưu riêng các sự kiện/nguyện vọng đã xác nhận, rồi gửi phần tóm tắt cùng ba lượt gần nhất; đồng thời đặt ngưỡng token để chi phí không tăng vô hạn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
