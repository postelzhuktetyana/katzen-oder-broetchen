# Katzen oder Brötchen (Кіт чи булочка?)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/postelzhuktetyana/katzen-oder-broetchen/blob/main/cat_or_bun_classifier.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/postelzhuktetyana/katzen-oder-broetchen/blob/main/cat_or_bun_classifier.ipynb)

1. **Курсова робота**: Класифікатор зображень "Кіт чи булочка?".
2. **Методологія**: Побудовано на базі навчального курсу [Fast.AI](https://course.fast.ai/) (синтез уроку Jeremy Howard *"Is it a bird? Creating a model from your own data"* та розділів *Fastbook Ch. 1 & 2*).
3. **Особливості**:
   - **Адаптивний профіль обчислень (Cloud GPU vs. Local CPU)**:
     - 🚀 **Google Colab / Kaggle (GPU)**: Автоматично задіює прискорювач NVIDIA T4, архітектуру **ResNet-34**, розмір батчу `bs=32`. Навчання займає **~15 секунд**.
     - 💻 **Локальний CPU (ноутбуки з 8 ГБ RAM)**: Автоматично перемикається на ультралегкий профіль **ResNet-18**, `bs=8`, однопотокове завантаження та активний колбек `MemoryGuardCallback`, що запобігає зависанню ОС.
   - **Небінарна класифікація**: Відображення нормалізованого розподілу ймовірностей (**Softmax**) для кожної категорії (наприклад, `84.2% Cat, 15.8% Bun`) з візуалізацією горизонтальної стовпчикової діаграми замість спрощеного "так/ні".
   - **Надійний веб-пошук зображень**: Пошук через асинхронний Bing image endpoint із браузерними заголовками (та резервним DDGS), що не потребує API-ключів та виключає помилки лімітів (HTTP 202 Ratelimit).
   - **Розумне кешування (Smart Caching)**: Автоматичний пропуск повторного завантаження, якщо папка вже містить $\ge 120$ зображень (із тумблером `force_redownload = False`).
   - **Криптографічна дедуплікація (SHA-256)**: Перевірка хеш-сум файлів, що гарантує відсутність однакових зображень у вибірці.

---

## 📁 Структура проєкту (Project Structure)

- 📓 [`cat_or_bun_classifier.ipynb`](cat_or_bun_classifier.ipynb): Основний навчальний Jupyter Notebook: завантаження даних, Presizing-аугментація, тонке налаштування ResNet та діагностика помилок.
- 🧪 [`cat_or_bun_tester.ipynb`](cat_or_bun_tester.ipynb): Інтерактивний ноутбук для тестування навченої моделі: завантаження `models/cat_or_bun_model.pkl`, вибір зображень через інтерактивний файловий браузер, drag-and-drop віджет завантаження з комп'ютера та візуалізація Softmax-розподілу ймовірностей.
- 📋 [`DESIGN.md`](DESIGN.md): Повний оновлений технічний дизайн-документ проекту.
- 📦 [`requirements.txt`](requirements.txt): Залежності для локального віртуального середовища (із сумісністю `ipywidgets` та `bing-image-downloader`).
- 📂 `data/cats_and_buns/`: Каталог із завантаженими зображеннями (`cat/` та `bun/`), створюється автоматично.
- 📂 `testing_images/`: Каталог із зовнішніми тестовими зображеннями для перевірки узагальнення моделі (`katzebroetchen.jpg`).
- 📂 `models/`: Каталог для збереження експортованої навченої моделі `cat_or_bun_model.pkl`.

---

## 🚀 Запуск проєкту (Getting Started)

### Варіант 1: Локальне середовище (Local Virtualenv)

1. Створіть та активуйте віртуальне середовище (рекомендовано Python 3.10 - 3.12):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Встановіть оновлені залежності:
   ```bash
   pip install -r requirements.txt
   ```
3. Запустіть Jupyter Lab або Notebook:
   ```bash
   jupyter lab
   ```
4. Відкрийте та виконайте `cat_or_bun_classifier.ipynb`.

### Варіант 2: Google Colab / Kaggle

1. Завантажте файл `cat_or_bun_classifier.ipynb` у Google Colab або Kaggle Notebooks.
2. Переконайтеся, що активовано апаратний прискорювач (GPU: T4).
3. Розкоментуйте та виконайте першу клітинку для встановлення залежностей:
   ```python
   !pip install -Uqq fastai bing-image-downloader duckduckgo_search
   ```
4. Виконайте всі клітинки по порядку (`Runtime -> Run all`).
