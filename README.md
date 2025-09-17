## Глубокое обучение (практика) МКН СПбГУ

Преподаватель: Георгий Сарапулов
- Telegram: https://t.me/g_eos
- Email: g-eos@yandex.ru

### Содержание

| Тема              | Материалы | Задания |
| :---------------- | :------ | :---- |
| 1. Введение в глубокое обучение <br><br> | [Запись 02.09.2025](https://disk.yandex.ru/i/C6QWtdfa5peEmQ) <br> [Слайды](slides/01_dl_intro.pdf) |  |
| 2. Введение в Pytorch (часть 1) <br><br> | [Запись 09.09.2025](https://disk.yandex.ru/i/dbfWNNl1oQgpfQ) <br> [Тензоры](notebooks/tensors.ipynb)| [Операции над тензорами](assignments/tensor_ops.ipynb) <br> [Градиентный спуск на NumPy](assignments/gradient_descent.ipynb) <br> Дедлайн: 24.09|
| 3. Введение в Pytorch (часть 2) <br>Инициализация параметров<br><br><br><br><br><br> | [Запись 16.09.2025](https://disk.yandex.ru/i/cpxycbi74FO6Dg) <br> [Autograd в PyTorch](notebooks/autograd.ipynb) <br> [Пример: логистическая регрессия на Pytorch](notebooks/backprop.ipynb) <br> [Оптимизаторы, параметры и модули](notebooks/parameters_and_modules.ipynb) <br> [Работа с данными](notebooks/dataloader.ipynb) <br> [Пример: MLP на датасете MNIST](notebooks/mnist.ipynb) <br> [Инициализация параметров](notebooks/initialization.ipynb) (самостоятельное изучение)| [Стабилизация обучения и регуляризация](assignments/training_tricks.ipynb) <br> Дедлайн: 01.10 <br><br><br><br><br><br> |

### Создание окружения
Версии пакетов в списке зависимостей зафиксированы под python 3.13 и pytorch 2.8.0. Можно использовать другие версии, но в этом случае будьте готовы самостоятельно решать проблемы с зависимостями (впрочем, это полезный опыт).

Создаём и активируем виртуальное окружение:
```
python3.13 -m venv .venv
source .venv/bin/activate
```

Далее устанавливаем зависимости, они немного отличаются в зависимости от операционной системы, архитектуры процессора и модели видеокарты.

Для linux без использования CUDA (легковесная версия, подойдёт для большей части заданий)
```
pip install -r requirements-linux-cpu.txt
```
Для linux с использованием CUDA (у вас должна быть видеокарта NVIDIA и до 15 Гб места на диске):
```
pip install -r requirements-linux-cuda.txt
```
Для MacOS с процессором серии M:
```
pip install -r requirements-macos.txt
```
Для более ранних Macbook pytorch новых версий (2.2.0+) больше не поставляется. Если вы хотите использовать такой Macbook, придётся использовать более старые версии python и pytorch.
