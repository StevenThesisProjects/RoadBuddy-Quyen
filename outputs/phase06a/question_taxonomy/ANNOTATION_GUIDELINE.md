# Hướng dẫn gán nhãn taxonomy câu hỏi RoadBuddy

## 1. Mục tiêu

Tài liệu này hướng dẫn hai người gán nhãn độc lập phân loại yêu cầu bằng chứng của 298 câu hỏi trong frozen validation. Kết quả được dùng để phân tích lỗi theo nhóm câu hỏi và lựa chọn hướng novelty cho Phase 06B.

Việc gán nhãn phải hoàn toàn độc lập với kết quả mô hình. Annotator không được xem prediction, raw response, correctness, leaderboard hoặc cấu hình thắng trong lúc annotation.

## 2. File sử dụng

File đầu vào:

- `outputs/phase06a/question_taxonomy/taxonomy_annotation_blind.csv`

Mỗi annotator tạo một bản sao và lưu lần lượt thành:

- `outputs/phase06a/question_taxonomy/annotator_1.csv`
- `outputs/phase06a/question_taxonomy/annotator_2.csv`

Không thay đổi, xóa hoặc sắp xếp lại `sample_id`; không thêm hoặc bỏ dòng. Mỗi file kết quả phải có đúng 298 `sample_id` duy nhất.

## 3. Các cột cần gán nhãn

### `visual_required`

Ghi `1` khi cần quan sát ít nhất một khung hình tĩnh để trả lời; ghi `0` khi không cần. Ví dụ: nhận diện biển báo, màu đèn, loại phương tiện, vị trí làn đường hoặc vật thể trong cảnh.

### `temporal_required`

Ghi `1` khi cần quan sát chuyển động, thứ tự sự kiện, hướng di chuyển hoặc thay đổi theo thời gian; ghi `0` khi một khung hình phù hợp đã đủ. Ví dụ: xe đang rẽ hướng nào, phương tiện nào đi trước hoặc xe có đổi làn hay không.

### `traffic_knowledge_required`

Ghi `1` khi cần kiến thức hoặc quy tắc giao thông bên ngoài hình ảnh; ghi `0` khi có thể trả lời bằng quan sát trực tiếp. Ví dụ: quyền ưu tiên, ý nghĩa pháp lý của biển báo, quy định làn đường hoặc hành vi được phép.

### `mixed_or_ambiguous`

Ghi `1` khi cần nhiều loại bằng chứng không thể tách rõ, hoặc nội dung/video không đủ rõ để gán chắc chắn; ngược lại ghi `0`.

### `primary_label`

Chọn đúng một nhãn:

- `visual_static`: bằng chứng chính là quan sát tĩnh.
- `temporal`: bằng chứng chính là thông tin theo thời gian/chuyển động.
- `traffic_knowledge`: bằng chứng chính là kiến thức hoặc luật giao thông.
- `mixed`: cần kết hợp nhiều loại bằng chứng và không có loại nào chi phối rõ.
- `ambiguous`: câu hỏi, video hoặc yêu cầu bằng chứng không đủ rõ.

### `annotator_note`

Có thể để trống nếu quyết định rõ. Ghi chú ngắn khi chọn `mixed`, `ambiguous`, video khó quan sát hoặc cần giải thích cho adjudication.

## 4. Quy trình gán nhãn

1. Đọc câu hỏi và các lựa chọn trả lời.
2. Xem video khi câu hỏi chưa đủ để xác định loại bằng chứng.
3. Gán bốn trục nhị phân bằng `0` hoặc `1`.
4. Chọn đúng một `primary_label`.
5. Ghi `annotator_note` nếu quyết định chưa rõ.
6. Hoàn thành đủ 298 dòng mà không trao đổi nhãn với annotator còn lại.
7. Kiểm tra không còn ô trống trong năm cột nhãn bắt buộc.

Các trục nhị phân có thể đồng thời bằng `1`. Ví dụ, câu hỏi có thể vừa cần quan sát biển báo (`visual_required=1`) vừa cần biết quy định của biển (`traffic_knowledge_required=1`). `primary_label` thể hiện loại bằng chứng chính.

## 5. Những điều không được làm

- Không xem prediction, correctness hoặc artifact đánh giá mô hình.
- Không dùng đáp án đúng/sai của mô hình để suy ngược taxonomy.
- Không tự động sinh nhãn bằng mô hình ngôn ngữ hoặc VQA.
- Không trao đổi hoặc sao chép nhãn giữa hai annotator trước khi hoàn tất hai file độc lập.
- Không dùng `support_frames` hoặc kết quả public test.
- Không thay đổi câu hỏi, lựa chọn, video path, `sample_id` hoặc `group_id`.

## 6. Agreement và adjudication

Sau khi nhận đủ hai file độc lập, pipeline sẽ:

1. Kiểm tra đúng 298 IDs và schema hợp lệ.
2. Tính Cohen's kappa cho từng trục và `primary_label`.
3. Xuất danh sách các dòng bất đồng.
4. Người adjudicate xem lại bất đồng và tạo `taxonomy_adjudicated.csv`.
5. Freeze thành `taxonomy_frozen.csv` cùng `taxonomy_manifest.json` và SHA-256 provenance.

Người adjudicate không thay đổi dòng đã đồng thuận nếu không có lý do được ghi lại. Quyết định adjudication cần có tên người thực hiện, ngày thực hiện và ghi chú khi cần.

## 7. Checklist bàn giao

- [ ] Có đúng 298 dòng dữ liệu và 298 `sample_id` duy nhất.
- [ ] Không thay đổi membership hoặc nội dung gốc.
- [ ] Bốn trục nhị phân chỉ chứa `0` hoặc `1`.
- [ ] Mỗi dòng có đúng một `primary_label` hợp lệ.
- [ ] Không còn giá trị trống ở các cột bắt buộc.
- [ ] Annotation độc lập và không xem kết quả mô hình.
- [ ] Lưu đúng tên `annotator_1.csv` hoặc `annotator_2.csv`.
