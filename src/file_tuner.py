"""Fine-tune model HSEmotion tren dataset nho"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np

# Fix Unicode cho terminal Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# dinh dang anh ho tro
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def _is_image_file(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_EXTENSIONS)

from src.constants import EMOTION_LABELS_6 as MODEL_CLASSES


class EmotionFineTuner:
    """Fine-tune EfficientNet tu HSEmotion tren dataset tu thu thap

    Chien luoc:
    - Dong bang backbone -> giu dac trung hoc tu 450k anh
    - Mo dong bang 2 blocks cuoi + classifier -> hoc them data nho
    - Split tu dong 70/15/15 (train/val/test)
    """

    def __init__(self, model_name: str = 'enet_b0_8_best_afew'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Thiet bi: {self.device.upper()}")

        # load model truc tiep (ko qua HSEmotionRecognizer vi no xoa classifier)
        print(f"  Dang load model goc: {model_name} ...")
        model = self._load_model(model_name)

        self.model = model.to(self.device)

        # transform cho train (co RandomHorizontalFlip)
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # transform cho validation / test (khong co RandomHorizontalFlip)
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Buoc 1: Dong bang TOAN BO tham so
        for param in self.model.parameters():
            param.requires_grad = False

        # Buoc 2: Chi mo dong bang duy nhat lop phan loai classifier (Cach 1)
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Tham so co the train: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = []
        self.label_map = {}

    @staticmethod
    def _load_model(model_name: str) -> nn.Module:
        """Load EfficientNet tu file .pt cua HSEmotion (khac phuc incompatibility)"""
        import timm
        from hsemotion.facial_emotions import get_model_path

        path = get_model_path(model_name)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt

        # rename key classifier.0.weight -> classifier.weight (timm moi)
        if 'classifier.0.weight' in sd:
            sd['classifier.weight'] = sd.pop('classifier.0.weight')
            sd['classifier.bias'] = sd.pop('classifier.0.bias')

        # chon architecture theo model_name
        if 'b2' in model_name:
            arch_name = 'tf_efficientnet_b2.ns_jft_in1k'
        else:
            arch_name = 'tf_efficientnet_b0.ns_jft_in1k'

        # Load model 8 classes de lay weights goc
        model_8 = timm.create_model(arch_name, pretrained=False, num_classes=8)
        missing = model_8.load_state_dict(sd, strict=False)
        if missing.unexpected_keys:
            print(f"  Unexpected keys: {missing.unexpected_keys}")
        if missing.missing_keys:
            real_missing = [k for k in missing.missing_keys if 'classifier' not in k]
            if real_missing:
                print(f"  Keys bi thieu: {real_missing}")

        # Tao model 6 classes de train phu hop voi dataset 6 labels
        model_6 = timm.create_model(arch_name, pretrained=False, num_classes=6)

        # Copy backbone weights tu model_8 sang model_6
        sd_6 = model_6.state_dict()
        for k, v in model_8.state_dict().items():
            if 'classifier' not in k:
                sd_6[k].copy_(v)
        model_6.load_state_dict(sd_6)

        # Khoi tao weights va bias cho classifier cua model_6 tu model_8 cho dung 6 class
        # 8 classes goc: Anger(0), Contempt(1), Disgust(2), Fear(3), Happiness(4), Neutral(5), Sadness(6), Surprise(7)
        # 6 classes dich: Anger(0), Disgust(2), Happiness(4), Neutral(5), Sadness(6), Surprise(7)
        indices_6 = [0, 2, 4, 5, 6, 7]
        with torch.no_grad():
            model_6.classifier.weight.copy_(model_8.classifier.weight[indices_6])
            model_6.classifier.bias.copy_(model_8.classifier.bias[indices_6])

        return model_6

    def load_dataset(self, dataset_dir: str, val_split: float = 0.15,
                     test_split: float = 0.15, batch_size: int = 16,
                     is_test_only: bool = False):
        """Doc dataset tu cau truc: dataset_dir/ClassName/image.jpg

        Ten folder phai khop voi MODEL_CLASSES
        """
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Khong tim thay dataset: {dataset_dir}")

        raw = datasets.ImageFolder(dataset_dir, is_valid_file=_is_image_file)
        folder_classes = raw.classes

        # xay dung mapping index ImageFolder -> index model
        self.label_map = {}
        self.class_names = []
        skipped = []
        for i, cls in enumerate(folder_classes):
            if cls in MODEL_CLASSES:
                self.label_map[i] = MODEL_CLASSES.index(cls)
                self.class_names.append(cls)
            else:
                skipped.append(cls)

        if skipped:
            print(f"  Bo qua class ko hop le: {skipped}")
            print(f"  Class hop le: {MODEL_CLASSES}")

        if not self.class_names:
            raise ValueError("Khong co class hop le nao trong dataset!")

        def _safe_remap(y: int) -> int:
            remapped = self.label_map.get(y)
            if remapped is None:
                raise ValueError(
                    f"Dataset folder index {y} ko co trong label_map. "
                    f"Co the folder ten ko hop le. "
                    f"Class hop le: {MODEL_CLASSES}"
                )
            return remapped

        current_transform = self.val_transform if is_test_only else self.train_transform
        full_dataset = datasets.ImageFolder(
            dataset_dir,
            transform=current_transform,
            target_transform=_safe_remap,
            is_valid_file=_is_image_file
        )

        n = len(full_dataset)
        if is_test_only:
            self.test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            self.train_loader = None
            self.val_loader = None
            print(f"\n  Test Only Dataset: {n} anh | {len(self.class_names)} classes: {self.class_names}")
            return None, None, self.test_loader

        n_test = int(n * test_split)
        n_val = int(n * val_split)
        if val_split > 0 and n_val == 0:
            n_val = 1
        if test_split > 0 and n_test == 0:
            n_test = 1
        n_train = n - n_val - n_test

        if n_train <= 0:
            raise ValueError(f"Dataset qua nho ({n} anh). Can it nhat 10 anh/class.")

        lengths = [n_train, n_val, n_test] if n_test > 0 else [n_train, n_val]
        splits = random_split(
            full_dataset, lengths,
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(splits[0], batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(splits[1], batch_size=batch_size, shuffle=False, num_workers=0)
        if n_test > 0:
            self.test_loader = DataLoader(splits[2], batch_size=batch_size, shuffle=False, num_workers=0)
        else:
            self.test_loader = None

        print(f"\n  Dataset: {n} anh | {len(self.class_names)} classes: {self.class_names}")
        print(f"  Train: {n_train} anh | Val: {n_val} anh | Test: {n_test} anh")

        return self.train_loader, self.val_loader, self.test_loader

    def train(self, epochs: int = 15, lr: float = 1e-4,
              save_path: str = 'models/finetuned.pt') -> dict:
        """Fine-tune model va luu weights tot nhat

        Returns:
            history: dict chua loss/accuracy tung epoch
        """
        if self.train_loader is None:
            raise RuntimeError("Chua load dataset! Goi load_dataset() truoc.")

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        best_val_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        print(f"\n  BAT DAU FINE-TUNE -- {epochs} EPOCH")
        print(f"  LR={lr} | Device={self.device.upper()} | Batch={self.train_loader.batch_size}\n")

        for epoch in range(epochs):
            # Train
            self.model.train()
            t_loss = t_correct = t_total = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                t_loss += loss.item()
                _, predicted = outputs.max(1)
                t_correct += predicted.eq(labels).sum().item()
                t_total += labels.size(0)

            # Validation
            self.model.eval()
            v_loss = v_correct = v_total = 0

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    v_loss += loss.item()
                    _, predicted = outputs.max(1)
                    v_correct += predicted.eq(labels).sum().item()
                    v_total += labels.size(0)

            train_acc = 100.0 * t_correct / t_total
            val_acc = 100.0 * v_correct / v_total
            avg_t_loss = t_loss / len(self.train_loader)
            avg_v_loss = v_loss / len(self.val_loader)

            history['train_loss'].append(avg_t_loss)
            history['val_loss'].append(avg_v_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            saved = ''
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'class_names': self.class_names,
                    'model_name': self.model_name,
                    'val_acc': val_acc,
                    'epoch': epoch + 1,
                }, save_path)
                saved = '  * Saved!'

            print(f"Epoch [{epoch+1:2d}/{epochs}] "
                  f"Train Loss: {avg_t_loss:.4f} Acc: {train_acc:5.1f}% | "
                  f"Val Loss: {avg_v_loss:.4f} Acc: {val_acc:5.1f}%{saved}")

            scheduler.step()

        print(f"\n{'='*55}")
        print(f"  Fine-tune hoan tat!")
        print(f"  Best val accuracy: {best_val_acc:.1f}%")
        print(f"  Model da luu tai: {save_path}")
        print(f"{'='*55}\n")

        return history

    @property
    def test_data(self):
        return self.test_loader, self.class_names
