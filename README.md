## Материалы курса Глубокое обучение (практика) для 3 курса НоД МКН СПбГУ

### Содержимое
| Тема              | Ноутбук и запись | Задания |
| :---------------- | :------ | :---- |
| 1. Знакомство с Pytorch: <br> тензоры, autograd, обучение перцептрона | [01_pytorch_intro.ipynb](workshops/01_pytorch_intro.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/la2EklKdychyEA)  | [01_basics.ipynb](assignments/01_basics.ipynb) <br> [01_mnist.ipynb](assignments/01_mnist.ipynb) |
| 2. Как не потерять градиент: <br> функции активации, инициализация, нормализация | [02_initialization_batchnorm.ipynb](workshops/02_initialization_batchnorm.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/xkfF5dY5UvaDlg)| [03_init_act_norm_optim.ipynb](assignments/03_init_act_norm_optim.ipynb)|
| 3. Регуляризация. <br> Введение в свёрточные сети | [03_regularization.ipynb](workshops/03_regularization.ipynb) <br> [04_cnn_intro.ipynb](workshops/04_cnn_intro.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/lXRfl467B1J8Ug)| |
| 4. Pytorch Lightning <br> (самостоятельное изучение) | [05_lightning_etc.ipynb](workshops/05_lightning_etc.ipynb) | [04_finetuning_augmentation.ipynb](assignments/04_finetuning_augmentation.ipynb)|
| 5. Семантическая сегментация | [06_segmentation.ipynb](workshops/06_segmentation.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/uEJGaAHgcHzTIg) | [05_cell_segmentation.ipynb](assignments/05_cell_segmentation.ipynb) |
| 6. Обнаружение объектов | [07_object_detection.ipynb](workshops/07_object_detection.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/7QEID5-6F7TwxQ) | [06_cell_detection.ipynb](assignments/06_cell_detection.ipynb) |
| 7. Рекуррентные сети | [08_rnn.ipynb](workshops/08_rnn.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/45b_pk0YXrjYEw) |  |

Записи практик 3 курса СП можно найти [по ссылке](https://disk.yandex.ru/d/EG-JuOGOdiyYmw)

### Создание окружения conda

Для Linux и Windows:

```bash
# создаём окружение из файла
PIP_EXISTS_ACTION=w conda env create -f environment-linux.yaml
# активируем окружение
conda activate dl-mcs
```

Если вы добавили новые зависимости в `.yaml` файл, среду можно обновить командой
```bash
conda env update -f environment-linux.yaml --prune
```

Для MacOS команды аналогичные, но указываем `environment-macos.yaml`
