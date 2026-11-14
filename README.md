# Katzen oder Brötchen (Кіт чи булочка?)

1. **Курсова робота**: Класифікатор зображень "Кіт чи булочка?".
2. **Методологія**: Побудовано на базі навчального курсу [Fast.AI](https://course.fast.ai/) (синтез уроку Jeremy Howard *"Is it a bird? Creating a model from your own data"* та розділів *Fastbook Ch. 1 & 2*).
3. **Особливість**: Небінарна класифікація з відображенням нормалізованого розподілу ймовірностей (**Softmax**) для кожної категорії (наприклад, `84.2% Cat, 15.8% Bun`) замість спрощеного "так/ні".

---

## 📁 Структура проєкту (Project Structure)

- 📓 [`cat_or_bun_classifier.ipynb`](cat_or_bun_classifier.ipynb): Основний навчальний Jupyter Notebook із детальним 3-компонентним поясненням до кожного блоку коду (Практична мета, ML-концепція, Деталі реалізації в Python/FastAI).
- 📋 [`DESIGN.md`](DESIGN.md): Повний технічний дизайн-документ проекту.
- 📦 [`requirements.txt`](requirements.txt): Залежності для локального віртуального середовища.
- 📂 `data/cats_and_buns/`: Каталог із завантаженими зображеннями котів та булочок (створюється автоматично під час виконання).
- 📂 `models/`: Каталог для збереження експортованої навченої моделі `cat_or_bun_model.pkl`.

---

## 🚀 Запуск проєкту (Getting Started)

### Варіант 1: Локальне середовище (Local Virtualenv)

1. Створіть та активуйте віртуальне середовище (рекомендовано Python 3.10 - 3.12):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Встановіть залежності:
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
   !pip install -Uqq fastai duckduckgo_search
   ```
4. Виконайте всі клітинки по порядку (`Runtime -> Run all`).

---

## 🧠 Ключові етапи пайплайну (Pipeline Highlights)

1. **Resilient Web Scraping**: Завантаження зображень через DuckDuckGo (`DDGS`) із затримками проти блокування запитів (`time.sleep`) та retry-логікою.
2. **Data Sanitization**: Перевірка та видалення пошкоджених файлів через `verify_images`.
3. **Fastbook Presizing**: Двоетапна обробка — масштабування на CPU (`Resize(460)`) та комбінована GPU-аугментація (`aug_transforms(224)`) без втрати чіткості.
4. **ResNet-34 Transfer Learning**: Тонке налаштування (`fine_tune`) з 1cycle-політикою та дискримінативними швидкостями навчання.
5. **Diagnostics & Interpretation**: Аналіз матриці помилок (Confusion Matrix) та візуалізація Top Losses.
6. **Multi-Class Probability Output**: Експорт моделі та інференс із побудовою горизонтальної діаграми розподілу ймовірностей для нових зображень.
