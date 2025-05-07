import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import json
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Настройка путей и конфигурации
class Config:
    def __init__(self):
        print("Initializing configuration...")
        self.data_dir = "data/cyrillic_handwriting"
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.train_labels = os.path.join(self.data_dir, "train.tsv")
        self.test_labels = os.path.join(self.data_dir, "test.tsv")
        self.alphabet = " абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        self.save_dir = "data/experiments/test"
        self.num_epochs = 10
        self.image_width = 256
        self.image_height = 32
        self.train_batch_size = 64
        self.val_batch_size = 128
        self.early_stopping_patience = 10
        print(f"Configuration set: data_dir={self.data_dir}, num_epochs={self.num_epochs}")

config = Config()

# Создаем необходимые директории
print("Creating directories if they don't exist...")
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs("data/predictions", exist_ok=True)
print(f"Directories created: {config.save_dir}, data/predictions")

# 2. Загрузка и подготовка данных
def load_tsv(file_path):
    """Загрузка TSV файла с метками"""
    print(f"Loading TSV file: {file_path}")
    start_time = time.time()
    df = pd.read_csv(file_path, sep='\t', header=None, names=['image_path', 'text'])
    invalid_rows = df[df['text'].isna() | df['text'].apply(lambda x: not isinstance(x, str))]
    if not invalid_rows.empty:
        print(f"Found {len(invalid_rows)} invalid rows in {file_path}:")
        print(invalid_rows)
    df = df[df['text'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    print(f"Loaded {len(df)} valid rows in {time.time() - start_time:.2f} seconds")
    return df

# Загружаем тренировочные данные
print("Loading and splitting training data...")
start_time = time.time()
train_df = load_tsv(config.train_labels)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
print(f"Train split: {len(train_df)} samples, Validation split: {len(val_df)} samples")
print(f"Saving train and validation splits to TSV files...")
train_df.to_csv(os.path.join(config.data_dir, 'train_split.tsv'), sep='\t', index=False, header=False)
val_df.to_csv(os.path.join(config.data_dir, 'val_split.tsv'), sep='\t', index=False, header=False)
print(f"Data loading and splitting completed in {time.time() - start_time:.2f} seconds")

# 3. Вспомогательные классы
class AverageMeter:
    """Вычисление и хранение средних значений"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'

class Tokenizer:
    """Кодирование/декодирование текста"""
    def __init__(self, alphabet):
        print("Initializing tokenizer...")
        self.char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
        self.char_map[CTC_BLANK] = 0
        self.char_map[OOV_TOKEN] = 1
        self.rev_char_map = {val: key for key, val in self.char_map.items()}
        print(f"Tokenizer initialized with {len(self.char_map)} characters")

    def encode(self, word_list):
        enc_words = []
        for word in word_list:
            enc_words.append([
                self.char_map[char] if char in self.char_map else self.char_map[OOV_TOKEN]
                for char in word
            ])
        return enc_words

    def get_num_chars(self):
        return len(self.char_map)

    def decode(self, enc_word_list):
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                if (char_enc != self.char_map[OOV_TOKEN] and
                        char_enc != self.char_map[CTC_BLANK] and
                        not (idx > 0 and char_enc == word[idx - 1])):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words

# 4. Трансформы для изображений
class ImageResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width
    def __call__(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

class Normalize:
    def __call__(self, img):
        return img.astype(np.float32) / 255

class ToTensor:
    def __call__(self, arr):
        return torch.from_numpy(arr)

class MoveChannels:
    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first
    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        return np.moveaxis(image, 0, -1)

def get_train_transforms(height, width):
    print("Setting up training transforms...")
    return torchvision.transforms.Compose([
        ImageResize(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])

def get_val_transforms(height, width):
    print("Setting up validation transforms...")
    return torchvision.transforms.Compose([
        ImageResize(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])

# 5. Dataset и DataLoader
class HandwritingDataset(Dataset):
    def __init__(self, df, img_dir, tokenizer, transform=None):
        print(f"Initializing dataset with {len(df)} samples...")
        start_time = time.time()
        self.df = df[df['text'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        # Check image existence
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, row['image_path'])
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
            else:
                print(f"Warning: Image not found: {img_path}")
        self.df = self.df.loc[self.valid_indices]
        self.texts = self.df['text'].tolist()
        self.enc_texts = self.tokenizer.encode(self.texts)
        print(f"Dataset initialized with {len(self.df)} valid samples in {time.time() - start_time:.2f} seconds")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_path']
        text = self.df.iloc[idx]['text']
        enc_text = torch.LongTensor(self.enc_texts[idx])
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if self.transform:
            image = self.transform(image)
        return image, text, enc_text

def collate_fn(batch):
    images, texts, enc_texts = zip(*batch)
    images = torch.stack(images, 0)
    text_lens = torch.LongTensor([len(text) for text in texts])
    enc_pad_texts = pad_sequence(enc_texts, batch_first=True, padding_value=0)
    return images, texts, enc_pad_texts, text_lens

def get_data_loader(df, img_dir, tokenizer, transforms, batch_size, shuffle=True, drop_last=False):
    print(f"Creating DataLoader with batch_size={batch_size}, shuffle={shuffle}, drop_last={drop_last}...")
    dataset = HandwritingDataset(
        df=df,
        img_dir=img_dir,
        tokenizer=tokenizer,
        transform=transforms
    )
    return DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=drop_last
    )

# 6. Модель CRNN
def get_resnet34_backbone(pretrained=True):
    print("Initializing ResNet34 backbone...")
    m = torchvision.models.resnet34(pretrained=pretrained)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out

class CRNN(nn.Module):
    def __init__(self, number_class_symbols, time_feature_count=256, lstm_hidden=256, lstm_len=2):
        super().__init__()
        print("Initializing CRNN model...")
        self.feature_extractor = get_resnet34_backbone(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x

# 7. Функции обучения и валидации
def get_accuracy(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        scores.append(true == pred)
    return np.mean(scores)

def predict(images, model, tokenizer, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    return tokenizer.decode(pred)

def val_loop(data_loader, model, tokenizer, device):
    print("Running validation...")
    start_time = time.time()
    acc_avg = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (images, texts, _, _) in enumerate(data_loader):
            batch_size = len(texts)
            text_preds = predict(images, model, tokenizer, device)
            acc_avg.update(get_accuracy(texts, text_preds), batch_size)
            print(f"Validation batch {i+1}/{len(data_loader)}, batch accuracy: {acc_avg.avg:.4f}")
    print(f"Validation completed in {time.time() - start_time:.2f} seconds, average accuracy: {acc_avg.avg:.4f}")
    return acc_avg.avg

def train_loop(data_loader, model, criterion, optimizer, epoch):
    print(f"Starting training epoch {epoch+1}...")
    start_time = time.time()
    loss_avg = AverageMeter()
    model.train()
    for i, (images, _, enc_pad_texts, text_lens) in enumerate(data_loader):
        model.zero_grad()
        images = images.to(device)
        batch_size = len(text_lens)
        output = model(images)
        output_lengths = torch.full(
            size=(output.size(1),),
            fill_value=output.size(0),
            dtype=torch.long
        )
        loss = criterion(output, enc_pad_texts, output_lengths, text_lens)
        loss_avg.update(loss.item(), batch_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        if (i + 1) % 10 == 0:  # Log every 10 batches
            print(f"Epoch {epoch+1}, batch {i+1}/{len(data_loader)}, batch loss: {loss_avg.avg:.5f}")
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f} seconds, average loss: {loss_avg.avg:.5f}, LR: {lr:.7f}")
    return loss_avg.avg

# 8. Инференс
class InferenceTransform:
    def __init__(self, height, width):
        self.transforms = get_val_transforms(height, width)
    def __call__(self, images):
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor

class OcrPredictor:
    def __init__(self, model_path, config, device='cuda'):
        print(f"Initializing OcrPredictor with model: {model_path}")
        self.tokenizer = Tokenizer(config.alphabet)
        self.device = torch.device(device)
        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.transforms = InferenceTransform(
            height=config.image_height,
            width=config.image_width,
        )
        print("OcrPredictor initialized")

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(f"Input must contain np.ndarray, tuple, or list, found {type(images)}.")
        images = self.transforms(images)
        pred = predict(images, self.model, self.tokenizer, self.device)
        if one_image:
            return pred[0]
        return pred

# 9. Основная функция обучения и инференса
def train_and_predict(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = Tokenizer(config.alphabet)

    # Создаем DataLoader'ы
    print("Creating training DataLoader...")
    train_loader = get_data_loader(
        df=train_df,
        img_dir=config.train_dir,
        tokenizer=tokenizer,
        transforms=get_train_transforms(config.image_height, config.image_width),
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=True
    )
    print("Creating validation DataLoader...")
    val_loader = get_data_loader(
        df=val_df,
        img_dir=config.train_dir,
        tokenizer=tokenizer,
        transforms=get_val_transforms(config.image_height, config.image_width),
        batch_size=config.val_batch_size,
        shuffle=False,
        drop_last=False
    )

    # Инициализируем модель
    model = CRNN(number_class_symbols=tokenizer.get_num_chars())
    model.to(device)

    # Критерий и оптимизатор
    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5, patience=5)

    # Обучение с early stopping
    print("Starting training...")
    start_training_time = time.time()
    best_acc = -np.inf
    epochs_no_improve = 0
    for epoch in range(config.num_epochs):
        train_loss = train_loop(train_loader, model, criterion, optimizer, epoch)
        val_acc = val_loop(val_loader, model, tokenizer, device)
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            model_save_path = os.path.join(config.save_dir, f'model-{epoch}-{val_acc:.4f}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= config.early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1} after {epochs_no_improve} epochs without improvement')
            break
    print(f"Training completed in {(time.time() - start_training_time)/60:.2f} minutes")

    # Инференс
    print("Starting inference on test set...")
    start_inference_time = time.time()
    test_df = load_tsv(config.test_labels)
    predictor = OcrPredictor(
        model_path=model_save_path,  # Use the best model
        config=config,
        device=device
    )
    pred_json = {}
    for idx, row in test_df.iterrows():
        img_path = os.path.join(config.test_dir, row['image_path'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Image not found: {img_path}")
            continue
        pred = predictor(img)
        pred_json[row['image_path']] = pred
        if idx % 100 == 0:  # Log every 100 images
            print(f"Inference on image {idx+1}/{len(test_df)}, path: {img_path}, prediction: {pred}")
    with open('data/prediction_HTR.json', 'w') as f:
        json.dump(pred_json, f)
    print(f"Inference completed in {(time.time() - start_inference_time):.2f} seconds, predictions saved to data/prediction_HTR.json")

# 10. Запуск программы
if __name__ == '__main__':
    print("Starting program...")
    start_program_time = time.time()
    train_and_predict(config)
    print(f"Program completed in {(time.time() - start_program_time)/60:.2f} minutes")