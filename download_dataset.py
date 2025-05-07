import os
import zipfile
import shutil
import kagglehub

def prepare_dataset():
    """Подготавливает датасет, сохраняя нужную структуру папок"""
    print("Preparing dataset...")

    # Пути из вашего Config класса
    data_dir = "data/cyrillic_handwriting"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    train_labels = os.path.join(data_dir, "train.tsv")
    test_labels = os.path.join(data_dir, "test.tsv")

    # Если данные уже есть - пропускаем загрузку
    if os.path.exists(train_labels) and os.path.exists(test_labels):
        print("Dataset already exists, skipping download")
        return

    # Создаем структуру папок
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Определяем temp_dir заранее
    temp_dir = os.path.abspath("temp_dataset")

    try:
        # Скачиваем датасет
        print("Downloading dataset from Kaggle Hub...")
        dataset_path = kagglehub.dataset_download("constantinwerner/cyrillic-handwriting-dataset")
        print(f"Downloaded dataset path: {dataset_path}")

        # Проверяем, является ли dataset_path директорией
        if os.path.isdir(dataset_path):
            # Проверяем, содержит ли директория уже распакованные файлы
            if os.path.exists(os.path.join(dataset_path, "train.tsv")) and \
               os.path.exists(os.path.join(dataset_path, "test.tsv")) and \
               os.path.exists(os.path.join(dataset_path, "train")) and \
               os.path.exists(os.path.join(dataset_path, "test")):
                print("Found pre-extracted dataset files, using them...")
                # Копируем файлы напрямую
                shutil.copy(os.path.join(dataset_path, "train.tsv"), train_labels)
                shutil.copy(os.path.join(dataset_path, "test.tsv"), test_labels)

                def copy_images(src_folder, dst_folder):
                    for img in os.listdir(src_folder):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            src_path = os.path.join(src_folder, img)
                            dst_path = os.path.join(dst_folder, img)
                            shutil.copy(src_path, dst_path)

                copy_images(os.path.join(dataset_path, "train"), train_dir)
                copy_images(os.path.join(dataset_path, "test"), test_dir)
                print("Dataset preparation complete!")
                return

            # Если нет распакованных файлов, ищем .zip
            zip_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.zip')]
            if not zip_files:
                raise FileNotFoundError(f"No .zip file or dataset files found in directory: {dataset_path}")
            if len(zip_files) > 1:
                raise ValueError(f"Multiple .zip files found in directory: {zip_files}")
            zip_path = os.path.join(dataset_path, zip_files[0])
            print(f"Found zip file: {zip_path}")
        else:
            zip_path = dataset_path

        # Проверяем, что zip_path - это файл .zip
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found at: {zip_path}")
        if not zip_path.lower().endswith('.zip'):
            raise ValueError(f"Downloaded file is not a zip file: {zip_path}")

        # Создаем временную папку для распаковки
        os.makedirs(temp_dir, exist_ok=True)

        # Распаковываем
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Переносим файлы в нужную структуру
        print("Organizing files...")
        shutil.copy(os.path.join(temp_dir, "train.tsv"), train_labels)
        shutil.copy(os.path.join(temp_dir, "test.tsv"), test_labels)

        def copy_images(src_folder, dst_folder):
            for img in os.listdir(src_folder):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(src_folder, img)
                    dst_path = os.path.join(dst_folder, img)
                    shutil.copy(src_path, dst_path)

        copy_images(os.path.join(temp_dir, "train"), train_dir)
        copy_images(os.path.join(temp_dir, "test"), test_dir)

        print("Dataset preparation complete!")

    except Exception as e:
        print(f"Error preparing dataset: {e}")
        raise
    finally:
        # Удаляем временные файлы
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

prepare_dataset()
print("You can now run your main script")