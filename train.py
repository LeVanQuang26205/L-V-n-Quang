import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import os
import time
import numpy as np

# --- CẤU HÌNH DỰ ÁN KHẨU TRANG ĐEN ---
DATA_DIR = 'DataMask'
MODEL_SAVE_PATH = 'mask_angle_2class_vgg16.pth'
NUM_EPOCHS = 30  # Tăng số Epoch để có thời gian Dừng sớm (tối ưu nhất)
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
IMAGE_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- CẤU HÌNH CHO ỔN ĐỊNH VÀ HỌC LIÊN TỤC ---
PATIENCE = 7  # Số lượng epoch chờ Val Loss không cải thiện trước khi GIẢM LR (LR Scheduler)
ES_PATIENCE = 15  # Số lượng epoch chờ Val Acc không cải thiện trước khi DỪNG SỚM (Early Stopping)


# --- HÀM TẢI DỮ LIỆU VÀ TĂNG CƯỜNG ---
def get_data_loaders(data_dir, batch_size, img_size):
    """Tải và áp dụng biến đổi cho tập Train và Val, bao gồm Data Augmentation mạnh."""

    # Data Augmentation mạnh mẽ cho tập TRAIN
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Biến đổi đơn giản cho tập VAL
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.classes


# --- HÀM XÂY DỰNG MÔ HÌNH VÀ TẢI TRỌNG SỐ ---
def build_model(num_classes, model_save_path, device):
    """Xây dựng VGG16 và tải trọng số đã lưu nếu tồn tại để học liên tục."""
    model = models.vgg16(weights='DEFAULT')

    # Đóng băng các lớp Convolution (CONV)
    for param in model.parameters():
        param.requires_grad = False

    # Thay thế lớp phân loại (FC layer) cuối cùng
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    # TẢI TRỌNG SỐ ĐÃ HUẤN LUYỆN (HỌC LIÊN TỤC)
    if os.path.exists(model_save_path):
        print(f"✅ Phát hiện mô hình đã lưu, tải trọng số từ {model_save_path} để tiếp tục huấn luyện...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("❌ Không tìm thấy mô hình đã lưu. Bắt đầu huấn luyện từ đầu.")

    model = model.to(device)
    return model


# --- HÀM HUẤN LUYỆN CHÍNH ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_acc = 0.0

    # Logic Dừng Sớm
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # LOGIC DỪNG SỚM & ĐIỀU CHỈNH LR
            if phase == 'val':
                # Bộ điều chỉnh LR (theo dõi Val Loss)
                scheduler.step(epoch_loss)

                # Lưu mô hình tốt nhất (theo dõi Val Acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"*** Lưu mô hình mới (Acc: {best_acc:.4f}) tại {MODEL_SAVE_PATH} ***")
                    early_stop_counter = 0  # Đặt lại bộ đếm khi mô hình cải thiện
                else:
                    early_stop_counter += 1

                # Logic Dừng Sớm (Early Stopping)
                if early_stop_counter >= ES_PATIENCE:
                    print(f"\n🛑 DỪNG SỚM: Val Acc không cải thiện sau {ES_PATIENCE} Epoch. Kết thúc huấn luyện.")
                    time_elapsed = time.time() - since
                    print(f"Huấn luyện hoàn tất trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
                    print(f"Độ chính xác tốt nhất trên tập Val: {best_acc:.4f}")
                    return

    time_elapsed = time.time() - since
    print(f"Huấn luyện hoàn tất trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Độ chính xác tốt nhất trên tập Val: {best_acc:.4f}")


if __name__ == '__main__':
    train_loader, val_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    num_classes = len(class_names)

    print(f"Mô hình sẽ phân loại {num_classes} lớp: {class_names}")

    if num_classes != 2:
        print(
            f"❌ LỖI: Dự án này yêu cầu 2 lớp, nhưng tìm thấy {num_classes}. Vui lòng kiểm tra lại folder DataMask/train và DataMask/val.")
        exit()

    # Xây dựng mô hình và TẢI TRỌNG SỐ
    model = build_model(num_classes, MODEL_SAVE_PATH, DEVICE)

    # Tối ưu hóa chỉ các tham số của lớp phân loại cuối cùng
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # BỘ ĐIỀU CHỈNH TỐC ĐỘ HỌC (LR Scheduler) - ĐÃ SỬA LỖI VERBOSE
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=PATIENCE
        # Đã xóa verbose=True để khắc phục lỗi
    )

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS)