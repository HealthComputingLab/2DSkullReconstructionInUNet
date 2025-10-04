import cv2

# ==== Cấu hình ====
image_path = r'D:\ToanLD_20231033M\1_MainProject\Unet_Architecture\Image_Painting\dataset\dataset\train\0.jpg'   # ← đường dẫn ảnh cần test
cut_size = (50, 50)                # ← kích thước vùng cắt (h, w)

# ==== Biến lưu ====
coords = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cut_h, cut_w = cut_size

        # Đảm bảo không vượt khỏi ảnh
        if x + cut_w > img.shape[1] or y + cut_h > img.shape[0]:
            print(f"❌ Vị trí ({x}, {y}) vượt giới hạn ảnh.")
            return

        # Vẽ hình chữ nhật tại vị trí cắt
        img_copy = img.copy()
        cv2.rectangle(img_copy, (x, y), (x + cut_w, y + cut_h), (0, 0, 255), 2)
        cv2.imshow("Test Mask Crop", img_copy)

        # Lưu và in tọa độ
        coords.clear()
        coords.append((x, y))
        print(f"✅ Selected crop coordinate: x = {x}, y = {y}")

# ==== Load ảnh ====
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

cv2.imshow("Test Mask Crop", img)
cv2.setMouseCallback("Test Mask Crop", click_event)

print("🖱️ Click vào ảnh để chọn vùng cắt.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==== Sau khi click xong, lấy kết quả ====
if coords:
    x, y = coords[0]
    print(f"\n👉 Final selected coordinate: (x={x}, y={y})")
else:
    print("\n❗ Không có vùng nào được chọn.")
