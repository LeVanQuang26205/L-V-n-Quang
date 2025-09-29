import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
from glob import glob

# --- CẤU HÌNH DỰ ÁN KHẨU TRANG ĐEN (2 LỚP) ---
MODEL_LOAD_PATH = 'mask_angle_2class_vgg16.pth'
CLASS_NAMES = [
    'frontal_black_mask',  # Lớp 0 (Chính diện)
    'side_black_mask'  # Lớp 1 (Nghiêng/Cạnh)
]
IMAGE_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- HÀM TẢI MÔ HÌNH ---
def load_model(num_classes):
    """Xây dựng và tải trọng số mô hình VGG16 đã huấn luyện."""
    model = models.vgg16(weights='DEFAULT')
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    if not os.path.exists(MODEL_LOAD_PATH):
        print(f"❌ LỖI: Không tìm thấy file mô hình tại {MODEL_LOAD_PATH}.")
        print("Vui lòng đảm bảo file mô hình đã được tạo bởi train.py và nằm cùng thư mục.")
        exit()

    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


# --- HÀM TIỀN XỬ LÝ ẢNH ---
def preprocess_image(image_path, img_size):
    """Tải và tiền xử lý ảnh từ đường dẫn."""
    try:
        # Tải ảnh, chuyển về RGB và áp dụng các biến đổi
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        # Bỏ qua các file không phải ảnh hoặc file bị lỗi
        return None

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0)  # Thêm dimension batch
    return input_tensor.to(DEVICE)


# --- HÀM DỰ ĐOÁN CHÍNH ---
def predict_image(model, image_path):
    """Thực hiện dự đoán góc chụp cho một ảnh."""
    input_tensor = preprocess_image(image_path, IMAGE_SIZE)
    if input_tensor is None:
        return None, None, None

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)

    predicted_class_name = CLASS_NAMES[predicted_idx.item()]
    confidence = probabilities[0][predicted_idx.item()].item() * 100

    # Dịch kết quả sang Tiếng Việt
    angle_display = 'CHINH DIEN' if 'frontal' in predicted_class_name else 'NGHIENG/CANH'

    return angle_display, confidence, probabilities


# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == '__main__':

    # Kiểm tra tham số dòng lệnh
    if len(sys.argv) < 2:
        print("\nSử dụng: python predit_folder.py <>")
        print("-" * 30)
        # Sử dụng đường dẫn của bạn làm ví dụ mặc định
        print(r"VÍ DỤ: python predit_folder.py C:\Users\Admin\PycharmProjects\PythonProject\Nhandien_khuonmat\TestTTNT")
        print("-" * 30)
        exit()

    # Lấy đường dẫn folder từ tham số dòng lệnh thứ nhất
    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print(f"❌ LỖI: Đường dẫn không phải là một folder hợp lệ: {folder_path}")
        exit()

    print("\n--- BẮT ĐẦU DỰ ĐOÁN FOLDER ẢNH ---")
    model = load_model(len(CLASS_NAMES))

    # Lấy danh sách tất cả file ảnh trong folder
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(folder_path, ext)))

    if not image_files:
        print(f"❌ LỖI: Không tìm thấy file ảnh nào trong folder: {folder_path}")
        exit()

    print(f"✅ Tìm thấy {len(image_files)} ảnh để dự đoán.")
    print("=" * 50)

    for i, image_path in enumerate(image_files):

        angle_display, confidence, probabilities = predict_image(model, image_path)

        # Định dạng output chi tiết
        if angle_display:
            frontal_conf = probabilities[0][0].item() * 100
            side_conf = probabilities[0][1].item() * 100

            print(f"[{i + 1}/{len(image_files)}] Ảnh: {os.path.basename(image_path)}")
            print(f"  > DỰ ĐOÁN: {angle_display} ({confidence:.2f}%)")
            print(f"  > CHI TIẾT: [Chính diện: {frontal_conf:.2f}%] | [Nghiêng: {side_conf:.2f}%]")
            print("-" * 50)
        else:
            print(
                f"[{i + 1}/{len(image_files)}] Ảnh: {os.path.basename(image_path)} - BỎ QUA (File không phải ảnh hợp lệ)")

    print("\n--- QUÁ TRÌNH DỰ ĐOÁN HOÀN TẤT! ---")