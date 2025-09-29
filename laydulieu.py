import os
import shutil
import random
from glob import glob

# --- CẤU HÌNH DỰ ÁN KHẨU TRANG ĐEN ---
# Thư mục chứa 2 folder lớp gốc (frontal_black_mask và side_black_mask)
DATA_DIR = 'DataMask/all_data'
# Thư mục sẽ chứa các folder train/ và val/
OUTPUT_DIR = 'DataMask'
TRAIN_RATIO = 0.8  # Tỷ lệ 80% cho huấn luyện


def setup_data_mask():
    """Xóa cấu trúc cũ và chia dữ liệu 2 lớp khẩu trang mới."""
    print("--- BẮT ĐẦU CHIA TẬP DỮ LIỆU ---")

    # 1. Xóa các thư mục train/ và val/ cũ
    for subset in ['train', 'val']:
        subset_path = os.path.join(OUTPUT_DIR, subset)
        if os.path.exists(subset_path):
            shutil.rmtree(subset_path)

    # 2. Lặp qua các folder lớp gốc (frontal_black_mask, side_black_mask)
    for class_name in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_name)
        if os.path.isdir(class_path):
            # Lấy tất cả file ảnh (JPG, PNG)
            files = glob(os.path.join(class_path, '*'))
            files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not files:
                print(f"❌ CẢNH BÁO: Không tìm thấy ảnh trong thư mục: {class_name}")
                continue

            random.shuffle(files)

            # 3. Tính toán và chia tập train/val
            train_count = int(len(files) * TRAIN_RATIO)
            train_files = files[:train_count]
            val_files = files[train_count:]

            # 4. Tạo folder đích và copy file
            for subset, file_list in [('train', train_files), ('val', val_files)]:
                dest_dir = os.path.join(OUTPUT_DIR, subset, class_name)
                os.makedirs(dest_dir, exist_ok=True)

                for src_file in file_list:
                    file_name = os.path.basename(src_file)
                    dest_file = os.path.join(dest_dir, file_name)
                    shutil.copyfile(src_file, dest_file)

            print(f"✅ Hoàn thành chia lớp '{class_name}': {len(train_files)} (Train) | {len(val_files)} (Val)")


if __name__ == '__main__':
    # Đảm bảo DataMask/all_data tồn tại trước khi chạy
    if not os.path.exists(DATA_DIR):
        print(
            f"❌ LỖI: Không tìm thấy thư mục dữ liệu gốc tại {DATA_DIR}. Vui lòng tạo cấu trúc DataMask/all_data trước.")
    else:
        setup_data_mask()
        print("--- CHIA TẬP DỮ LIỆU HOÀN TẤT! ---")