import os, shutil, random

src_real = "data/processed/real"
src_fake = "data/processed/fake"

train_real = "data/train/real"
train_fake = "data/train/fake"
val_real = "data/val/real"
val_fake = "data/val/fake"

os.makedirs(train_real, exist_ok=True)
os.makedirs(train_fake, exist_ok=True)
os.makedirs(val_real, exist_ok=True)
os.makedirs(val_fake, exist_ok=True)

def split_data(src, train_dst, val_dst, split=0.8):
    files = os.listdir(src)
    random.shuffle(files)
    split_idx = int(len(files) * split)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    for f in train_files:
        shutil.move(os.path.join(src, f), os.path.join(train_dst, f))
    for f in val_files:
        shutil.move(os.path.join(src, f), os.path.join(val_dst, f))

split_data(src_real, train_real, val_real)
split_data(src_fake, train_fake, val_fake)
