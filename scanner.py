from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from PIL import Image
import os

# === 1. Load model YOLOv8 segmentasi ===
model = YOLO("yolov8m-seg.pt")  # pastikan model segmentasi digunakan
reader = easyocr.Reader(["en", "id"])  # OCR bahasa Inggris & Indonesia

# === 2. Baca gambar ===
image_path = "C:/Users/User/Pictures/Project PCD/Dokumen.jpg"  # ubah sesuai path kamu
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("❌ Gambar tidak ditemukan!")

orig = image.copy()
h, w = image.shape[:2]

# === 3. Deteksi area dokumen ===
results = model(image, verbose=False)
masks = getattr(results[0], "masks", None)
boxes = getattr(results[0], "boxes", None)

# === 4. Ambil mask atau fallback ke bounding box ===
if masks is not None and masks.data is not None and len(masks.data) > 0:
    print("✅ Mask dokumen ditemukan dari YOLOv8-seg")
    mask = masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (w, h))
    mask = (mask > 0.5).astype(np.uint8) * 255
else:
    print("⚠️ Mask tidak tersedia, fallback ke bounding box YOLOv8")
    if boxes is None or len(boxes) == 0:
        raise ValueError("❌ Tidak ada dokumen terdeteksi!")
    x1, y1, x2, y2 = map(int, boxes.xyxy[0])
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

# === 5. Refinement kontur ===
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.GaussianBlur(mask, (5, 5), 0)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

# === 6. Temukan tepi dokumen ===
peri = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.015 * peri, True)

if len(approx) < 4:
    print("⚠️ Titik sudut tidak sempurna, gunakan convex hull")
    approx = cv2.convexHull(contour)

# Urutkan titik
pts = approx.reshape(-1, 2)
rect = np.zeros((4, 2), dtype="float32")
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]      # top-left
rect[2] = pts[np.argmax(s)]      # bottom-right
diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]   # top-right
rect[3] = pts[np.argmax(diff)]   # bottom-left

# === 7. Transformasi perspektif sementara (tanpa margin) ===
(tl, tr, br, bl) = rect
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxWidth = int(max(widthA, widthB))
maxHeight = int(max(heightA, heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)
warped_temp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

# === 8. Deteksi jenis & rasio dokumen (tanpa margin) ===
h_temp, w_temp = warped_temp.shape[:2]
current_ratio = h_temp / w_temp

ratios = {
    "A4": 1.414,
    "A5": 1.414,
    "KTP": 0.63,
    "Kwitansi": 2.0
}
closest_name, closest_ratio = min(ratios.items(), key=lambda x: abs(x[1] - current_ratio))
print(f"📏 Rasio: {current_ratio:.3f}, tipe mirip: {closest_name}")

# === 9. Tambahkan margin hanya untuk KTP ===
if closest_name == "KTP":
    print("➕ Menambahkan margin untuk KTP agar tidak terpotong.")
    margin = 0.05  # 5% margin untuk KTP
    dx = int(w * margin)
    dy = int(h * margin)

    rect[0][0] = max(rect[0][0] - dx, 0)
    rect[0][1] = max(rect[0][1] - dy, 0)
    rect[1][0] = min(rect[1][0] + dx, w)
    rect[1][1] = max(rect[1][1] - dy, 0)
    rect[2][0] = min(rect[2][0] + dx, w)
    rect[2][1] = min(rect[2][1] + dy, h)
    rect[3][0] = max(rect[3][0] - dx, 0)
    rect[3][1] = min(rect[3][1] + dy, h)

# === 10. Transformasi perspektif final ===
(tl, tr, br, bl) = rect
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxWidth = int(max(widthA, widthB))
maxHeight = int(max(heightA, heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

# === 11. Peningkatan warna ===
b, g, r = cv2.split(warped)
b = cv2.equalizeHist(b)
g = cv2.equalizeHist(g)
r = cv2.equalizeHist(r)
merged = cv2.merge((b, g, r))
enhanced = cv2.convertScaleAbs(merged, alpha=1.3, beta=25)
blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
final = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)

# === 12. Penyesuaian rasio dokumen ===
h, w = final.shape[:2]
current_ratio = h / w
target_ratio = closest_ratio
if abs(current_ratio - target_ratio) > 0.05:
    if current_ratio < target_ratio:
        new_h = int(w * target_ratio)
        pad_total = max(0, new_h - h)
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        final = cv2.copyMakeBorder(final, pad_top, pad_bottom, 0, 0,
                                   cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        new_w = int(h / target_ratio)
        pad_total = max(0, new_w - w)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        final = cv2.copyMakeBorder(final, 0, 0, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=[255, 255, 255])

# === 13. Koreksi orientasi ===
if closest_name in ["A4", "A5"] and w > h:
    final = cv2.rotate(final, cv2.ROTATE_90_CLOCKWISE)
elif closest_name == "KTP" and h > w:
    final = cv2.rotate(final, cv2.ROTATE_90_CLOCKWISE)

# === 14. OCR (EasyOCR) ===
print("🔍 Membaca teks dengan EasyOCR...")
gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
ocr_results = reader.readtext(gray)
extracted_text = "\n".join([res[1] for res in ocr_results])

print("\n===== Teks Terbaca =====")
print(extracted_text)
print("=========================\n")

# === 15. Simpan sebagai gambar dan konversi ke PDF ===
output_image = f"hasil_scan_{closest_name.lower()}.jpg"
cv2.imwrite(output_image, final)

# Konversi ke PDF
output_pdf = f"hasil_scan_{closest_name.lower()}.pdf"
image_pil = Image.open(output_image)
rgb_image = image_pil.convert("RGB")
rgb_image.save(output_pdf)

# === 16. Tampilkan hasil ===
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.title(f"Hasil Scan ({closest_name})")
plt.axis("off")
plt.show()

print(f"✅ Hasil akhir disimpan sebagai '{output_pdf}'")

# === 17. Buka hasil PDF ===
os.startfile(output_pdf)
