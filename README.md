## Глубокое обучение (практика) МКН СПбГУ

Преподаватель: Георгий Сарапулов
- Telegram: https://t.me/g_eos
- Email: g-eos@yandex.ru

### Содержание

| Тема              | Материалы | Задания |
| :---------------- | :------ | :---- |
| 1. Введение в глубокое обучение <br><br> | [Запись 02.09.2025](https://disk.yandex.ru/i/C6QWtdfa5peEmQ) <br> [Слайды](slides/01_dl_intro.pdf) |  |
| 2. Введение в Pytorch (часть 1) <br><br><br> | [Запись 09.09.2025](https://disk.yandex.ru/i/dbfWNNl1oQgpfQ) <br> [Тензоры](notebooks/tensors.ipynb) <br><br>| [Операции над тензорами](assignments/tensor_ops.ipynb) <br> [Градиентный спуск на NumPy](assignments/gradient_descent.ipynb) <br> Дедлайн: 24.09 23:59|
| 3. Введение в Pytorch (часть 2) <br> Инициализация параметров<br><br><br><br><br><br><br><br> | [Запись 16.09.2025](https://disk.yandex.ru/i/cpxycbi74FO6Dg) <br> [Autograd в PyTorch](notebooks/autograd.ipynb) <br> [Пример: логистическая регрессия на Pytorch](notebooks/backprop.ipynb) <br> [Оптимизаторы, параметры и модули](notebooks/parameters_and_modules.ipynb) <br> [Работа с данными](notebooks/dataloader.ipynb) <br> [Пример: MLP на датасете MNIST](notebooks/mnist.ipynb) <br> [Инициализация параметров](notebooks/initialization.ipynb) (самостоятельное изучение)| [Стабилизация обучения и регуляризация](assignments/training_tricks.ipynb) <br> Дедлайн: 01.10 23:59 <br><br><br><br><br><br><br> |
| 4. Свёрточные нейронные сети. Классификация изображений. <br><br><br><br> | [Запись 23.09.2025](https://disk.yandex.ru/i/inAnB1LREV1SNg) <br>[Слайды](slides/dl_architectures.pdf)<br> [Блоки для свёрточных сетей](notebooks/cnn_blocks.ipynb) <br> [Пример: классификация CIFAR10](notebooks/cifar10.ipynb) <br> [Pytorch Lightning](notebooks/lightning_etc.ipynb) (самостоятельное изучение)| [Классификация изображений. Дообучение. Аугментация.](assignments/finetuning_augmentation.ipynb) <br> Дедлайн: 08.10 23:59 <br><br><br>|
| 5. Сегментация изображений <br><br><br> | [Запись 07.10.2025](https://disk.yandex.ru/i/Ud6KwcuSdsnlcg) <br> [Датасет с дрожжевыми клетками](notebooks/yeast_cell_in_microstructures.ipynb) <br> [Сегментация](notebooks/segmentation.ipynb)<br>| [Сегментация дрожжевых клеток](assignments/cell_segmentation.ipynb)<br> Дедлайн: 22.10 23:59 <br><br>|
| 6. Детекция объектов <br><br> | [Запись 14.10.2025](https://disk.yandex.ru/i/Q0oi5nx_C7fxZw) <br> [YOLO для детекции объектов](notebooks/object_detection.ipynb) <br> | [Детекция дрожжевых клеток](assignments/cell_detection.ipynb)<br> Дедлайн: 29.10 23:59 <br>|
| 7. Рекуррентные сети.<br> Механизм внимания. <br> Блок трансформера. <br><br> | [Запись 21.10.2025](https://disk.yandex.ru/i/QLdDq8DNc3mLqg) <br> [Слайды](slides/dl_architectures.pdf) <br> [Рекуррентная сеть для генерации имён](notebooks/rnn.ipynb) <br> [Механизм внимания](notebooks/transformer_attention.ipynb) | [Структурная классификация белков](assignments/protein_folds.ipynb)<br> Дедлайн: 05.11 23:59 <br><br><br>|
| 8. Трансформер: варианты архитектуры.<br> Введение в графовые нейронные сети.<br> | [Запись 28.10.2025](https://disk.yandex.ru/i/lhK8KTUFKTpCBg) <br> [Слайды](slides/dl_architectures.pdf) <br> [Трансформер: варианты](notebooks/transformer_model.ipynb) <br><br> | [TinyStories: генерация текстов](assignments/tiny_stories.ipynb) <br> [Multi30k: перевод с немецкого на английский](assignments/translation.ipynb) <br>  Дедлайн: 12.11 23:59 <br>|
| 9. Графовые нейронные сети.<br><br><br> | [Запись 11.11.2025](https://disk.yandex.ru/i/44PmP6r5Q2jV8A) <br> [GNN и Pytorch Geometric](notebooks/gnn.ipynb) <br><br> | [Предсказание свободной энергии связывания белков](assignments/binding_affinity.ipynb) <br>  Дедлайн: 26.11 23:59 <br>|

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
