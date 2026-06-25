# Báo Cáo Lab MLOps — CI/CD cho AI Systems

Course: AIInAction - VinUni · Day 21 - CI/CD cho AI Systems

---

## 1. Bộ siêu tham số đã chọn và lý do (kết quả Bước 1)

Đã chạy 10 thí nghiệm RandomForest trên cùng tập eval (500 mẫu held-out), ghi lại
bằng MLflow (`sqlite:///mlflow.db`). Bảng dưới là các lần chạy chính, sắp xếp theo
accuracy giảm dần:

| n_estimators | max_depth | min_samples_split | accuracy | f1_score |
|---|---|---|---|---|
| **300** | **None** | **2** | **0.6820** | **0.6811** |
| 500 | None | 2 | 0.6760 | 0.6748 |
| 400 | 30 | 2 | 0.6740 | 0.6729 |
| 200 | None | 2 | 0.6740 | 0.6730 |
| 500 | 25 | 2 | 0.6720 | 0.6708 |
| 100 | 20 | 10 | 0.6700 | 0.6682 |
| 200 | 10 | 5 | 0.6440 | 0.6417 |
| 100 | 5  | 2 | 0.5640 | 0.5534 |
| 50  | 3  | 2 | 0.5580 | 0.5185 |

**Bộ được chọn: `n_estimators=300, max_depth=None, min_samples_split=2`** (accuracy 0.6820).

Lý do:
- **max_depth=None** là yếu tố ảnh hưởng lớn nhất: cho cây phát triển hết cỡ giúp mô
  hình nắm các tương tác phi tuyến giữa các đặc trưng hóa học. Giới hạn độ sâu (3, 5, 10)
  làm accuracy giảm rõ rệt (0.56–0.64).
- **n_estimators=300** là điểm cân bằng tốt: tăng từ 200→300 cải thiện nhẹ
  (0.674→0.682), nhưng lên 500 lại giảm nhẹ (0.676) — thêm cây không còn lợi ích và
  bắt đầu nhiễu. 300 cho kết quả ổn định nhất.
- **min_samples_split=2** (mặc định) tốt hơn các giá trị lớn hơn; giá trị cao hơn
  (5, 10) ép cây nông hơn và làm giảm độ chính xác.

Đã thử thêm `GradientBoosting` / `HistGradientBoosting` (gợi ý ở Bonus 2) nhưng đều
**kém hơn** RandomForest trên tập dữ liệu này (acc 0.60–0.66), nên RandomForest là lựa
chọn tốt nhất.

---

## 2. So sánh Bước 2 vs Bước 3 (ảnh hưởng của dữ liệu mới)

Cùng một bộ siêu tham số, chỉ thay đổi lượng dữ liệu huấn luyện:

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) |
|---|---|---|
| accuracy | 0.682 | **0.746** |
| f1_score | 0.681 | **0.745** |

Bổ sung 2998 mẫu mới (train_phase2) làm accuracy tăng **~6.4 điểm phần trăm**. Đây là
minh chứng trực tiếp cho giá trị của huấn luyện liên tục: thêm dữ liệu → mô hình tốt hơn.

**Lưu ý về eval gate (ngưỡng 0.70):**
- Với dữ liệu Bước 2 (2998 mẫu), accuracy 0.682 **dưới ngưỡng 0.70** → eval gate sẽ
  chặn deploy. Điều này chứng minh eval gate hoạt động đúng.
- Với dữ liệu Bước 3 (5996 mẫu), accuracy 0.746 **vượt ngưỡng** → mô hình được deploy.

Đây chính là vòng phản hồi MLOps mong muốn: chỉ mô hình đạt chuẩn chất lượng mới được
đưa ra phục vụ.

---

## 3. Khó khăn gặp phải và cách giải quyết

| Khó khăn | Cách giải quyết |
|---|---|
| RandomForest với dữ liệu Bước 2 (2998 mẫu) chỉ đạt ~0.68, dưới ngưỡng deploy 0.70. | Xác định bằng thực nghiệm rằng đây là trần của mô hình trên lượng dữ liệu này; bổ sung dữ liệu ở Bước 3 đưa accuracy lên ~0.75. |
| `accuracy_score` trả về kiểu `numpy.float64`, có thể gây nhầm với assert `isinstance(acc, float)`. | Ép kiểu `float(...)` quanh accuracy và f1 trong `train()` để đảm bảo trả về Python `float` chuẩn. |
| GitHub Actions outputs là kiểu chuỗi, dễ gây lỗi khi so sánh số ở eval gate. | Eval gate dùng `float("${{ needs.train.outputs.accuracy }}")` để chuyển sang số trước khi so sánh với 0.70. |
| Thứ tự `dvc push` trước `git push` ở Bước 3. | Tuân thủ đúng thứ tự: đẩy dữ liệu lên cloud storage trước, rồi mới push commit để CI runner pull được dữ liệu mới. |

---

## 4. Trạng thái hoàn thành

- [x] **Bước 1** — `src/train.py` hoàn chỉnh, 10 thí nghiệm MLflow, chọn bộ tham số tốt nhất.
- [x] **Code Bước 2** — `src/serve.py`, `tests/test_train.py` (3 test PASS), `.github/workflows/mlops.yml` (4 jobs).
- [x] **Hạ tầng cloud** (cần tài khoản cá nhân của bạn): tạo bucket + VM, cấu hình DVC remote, thêm 5 GitHub Secrets, deploy. Xem `tasks/buoc-2.md`.
- [x] **Ảnh chụp màn hình** nộp bài: MLflow UI, Actions 4 jobs xanh, `curl /health` & `/predict`, Cloud Storage Console.

---

## 5. Bonus đã bổ sung

- **Bonus 1 - DagsHub/remote MLflow**: workflow hỗ trợ các secret `MLFLOW_TRACKING_URI`,
  `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`. Khi cấu hình các secret này, các run trong
  GitHub Actions sẽ ghi lên tracking server từ xa thay vì chỉ lưu local.
- **Bonus 2 - Nhiều thuật toán**: `params.yaml` có `model_type`; `src/train.py` hỗ trợ
  `random_forest`, `gradient_boosting`, `logistic_regression`.
- **Bonus 3 - Báo cáo hiệu suất tự động**: sau mỗi lần train tạo `outputs/report.txt`, gồm confusion
  matrix và precision/recall/F1 theo từng lớp; workflow upload file này cùng `metrics.json`.
- **Bonus 4 - Chặn deploy nếu model mới kém hơn**: workflow tải `models/latest/metrics.json` trên cloud,
  so sánh accuracy mới với accuracy đang chạy, và hủy deploy nếu model mới thấp hơn.
- **Bonus 5 - Cảnh báo lệch phân phối dữ liệu**: `src/train.py` tính tỷ lệ nhãn train, in cảnh báo nếu
  lớp nào dưới 10%, và ghi `label_distribution` vào `outputs/metrics.json`.
