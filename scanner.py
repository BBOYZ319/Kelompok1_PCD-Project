from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage import measure, morphology, filters  # 🆕 Tambahan library pendukung kontur

# === 1. Load model YOLOv8 segmentasi ===
model = YOLO("yolov8m-seg.pt")
reader = easyocr.Reader(["en", "id"])

# === 2. Baca gambar ===
image_path = "C:/Users/User/Pictures/Project PCD/L.jpg"
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
    mask = (mask > 0.5).astype(np.uint8)
else:
    print("⚠️ Mask tidak tersedia, fallback ke bounding box YOLOv8")
    if boxes is None or len(boxes) == 0:
        raise ValueError("❌ Tidak ada dokumen terdeteksi!")
    x1, y1, x2, y2 = map(int, boxes.xyxy[0])
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

# === 5. Refinement kontur dengan scikit-image ===
mask = morphology.remove_small_objects(mask.astype(bool), min_size=500)
mask = morphology.binary_closing(mask, morphology.disk(7))
mask = morphology.binary_opening(mask, morphology.disk(5))
mask = filters.gaussian(mask, sigma=2) > 0.3
mask = (mask * 255).astype(np.uint8)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) == 0:
    raise ValueError("❌ Kontur tidak ditemukan setelah peningkatan deteksi.")
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

# === 7. Transformasi perspektif sementara ===
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

# === 8. Deteksi jenis dokumen ===
h_temp, w_temp = warped_temp.shape[:2]
current_ratio = h_temp / w_temp
ratios = {"A4": 1.414, "A5": 1.414, "KTP": 0.63, "Kwitansi": 2.0}
closest_name, closest_ratio = min(ratios.items(), key=lambda x: abs(x[1] - current_ratio))
print(f"📏 Rasio: {current_ratio:.3f}, tipe mirip: {closest_name}")

# === 9. Transformasi perspektif final ===
warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

# === 10. Peningkatan warna ===
b, g, r = cv2.split(warped)
b, g, r = [cv2.equalizeHist(x) for x in [b, g, r]]
merged = cv2.merge((b, g, r))
enhanced = cv2.convertScaleAbs(merged, alpha=1.3, beta=25)
blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
final = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)

# === 11. Penyesuaian rasio dokumen ===
h, w = final.shape[:2]
current_ratio = h / w
target_ratio = closest_ratio
if abs(current_ratio - target_ratio) > 0.05:
    if current_ratio < target_ratio:
        new_h = int(w * target_ratio)
        pad_total = new_h - h
        pad_top, pad_bottom = pad_total // 2, pad_total - pad_total // 2
        final = cv2.copyMakeBorder(final, pad_top, pad_bottom, 0, 0,
                                   cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        new_w = int(h / target_ratio)
        pad_total = new_w - w
        pad_left, pad_right = pad_total // 2, pad_total - pad_total // 2
        final = cv2.copyMakeBorder(final, 0, 0, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=[255, 255, 255])

# === 12. Fungsi rotasi manual ===
def rotate_image_manual(image):
    """Putar gambar manual dengan tombol: 'r'=kanan, 'l'=kiri, ESC=selesai."""
    temp = image.copy()
    while True:
        cv2.imshow("🌀 Rotasi Manual (r=kanan, l=kiri, ESC=selesai)", temp)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            temp = cv2.rotate(temp, cv2.ROTATE_90_CLOCKWISE)
        elif key == ord('l'):
            temp = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif key == 27 or key == 13:  # ESC atau Enter
            break
    cv2.destroyAllWindows()
    return temp

# Jalankan rotasi manual
print("🌀 Tekan 'R' atau 'L' untuk rotasi, ESC/Enter untuk lanjut...")
final = rotate_image_manual(final)

# === 13. OCR ===
print("🔍 Membaca teks dengan EasyOCR...")
gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
ocr_results = reader.readtext(gray)
extracted_text = "\n".join([res[1] for res in ocr_results])

print("\n===== Teks Terbaca =====")
print(extracted_text)
print("=========================\n")

# === 14. Simpan hasil ===
output_image = f"hasil_scan_{closest_name.lower()}.jpg"
cv2.imwrite(output_image, final)

# Konversi ke PDF
output_pdf = f"hasil_scan_{closest_name.lower()}.pdf"
image_pil = Image.open(output_image)
rgb_image = image_pil.convert("RGB")
rgb_image.save(output_pdf)

# === 15. Tampilkan hasil ===
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.title(f"Hasil Scan ({closest_name})")
plt.axis("off")
plt.show()

print(f"✅ Hasil akhir disimpan sebagai '{output_pdf}'")
os.startfile(output_pdf)
