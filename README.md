## Материалы курса Глубокое обучение (практика) для 3 курса НоД МКН СПбГУ

### Содержимое
| Тема              | Ноутбук и запись | Задания |
| :---------------- | :------ | :---- |
| 1. Знакомство с Pytorch: <br> тензоры, autograd, обучение перцептрона | [01_pytorch_intro.ipynb](workshops/01_pytorch_intro.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/la2EklKdychyEA)  | [01_basics.ipynb](assignments/01_basics.ipynb) <br> [01_mnist.ipynb](assignments/01_mnist.ipynb) |
| 2. Как не потерять градиент: <br> функции активации, инициализация, нормализация | [02_initialization_batchnorm.ipynb](workshops/02_initialization_batchnorm.ipynb) | [03_init_act_norm_optim.ipynb](assignments/03_init_act_norm_optim.ipynb)|

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
