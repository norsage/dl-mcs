## Материалы курса Глубокое обучение (практика) для 3 курса НоД МКН СПбГУ

### Содержимое
| Тема              | Ноутбук и запись | Задания |
| :---------------- | :------ | :---- |
| 1. Знакомство с Pytorch: <br> тензоры, autograd, обучение перцептрона | [01_pytorch_intro.ipynb](workshops/01_pytorch_intro.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/la2EklKdychyEA)  | [01_basics.ipynb](assignments/01_basics.ipynb) <br> [02_mnist.ipynb](assignments/02_mnist.ipynb) |
| 2. Как не потерять градиент: <br> функции активации, инициализация, нормализация | [02_initialization_batchnorm.ipynb](workshops/02_initialization_batchnorm.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/xkfF5dY5UvaDlg)| [03_init_act_norm_optim.ipynb](assignments/03_init_act_norm_optim.ipynb)|
| 3. Регуляризация. <br> Введение в свёрточные сети | [03_regularization.ipynb](workshops/03_regularization.ipynb) <br> [04_cnn_intro.ipynb](workshops/04_cnn_intro.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/lXRfl467B1J8Ug)| |
| 4. Pytorch Lightning <br> (самостоятельное изучение) | [05_lightning_etc.ipynb](workshops/05_lightning_etc.ipynb) | [04_finetuning_augmentation.ipynb](assignments/04_finetuning_augmentation.ipynb)|
| 5. Семантическая сегментация | [06_segmentation.ipynb](workshops/06_segmentation.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/uEJGaAHgcHzTIg) | [05_cell_segmentation.ipynb](assignments/05_cell_segmentation.ipynb) |
| 6. Обнаружение объектов | [07_object_detection.ipynb](workshops/07_object_detection.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/7QEID5-6F7TwxQ) | [06_cell_detection.ipynb](assignments/06_cell_detection.ipynb) |
| 7. Рекуррентные сети | [08_rnn.ipynb](workshops/08_rnn.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/45b_pk0YXrjYEw) | [07_amino_acid_rnn.ipynb](assignments/07_amino_acid_rnn.ipynb) |
| 8. Трансформер: механизм внимания | [09_transformer_attention.ipynb](workshops/09_transformer_attention.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/K3QN07Jt3MrA9w) |  |
| 9. Трансформер: архитектуры | [10_transformer_model.ipynb](workshops/10_transformer_model.ipynb) <br> [Запись практики](https://disk.yandex.ru/d/YWUBE4M52gVENA) | [08_translation.ipynb](assignments/08_translation.ipynb) |
| 10. Глубокое обучение в структурной биологии | [Запись лекции](https://disk.yandex.ru/i/TNHkYiOaAfIQOQ) <br> [Слайды](workshops/ml_in_structural_biology.pdf) |  |
| 11. Графовые сети | [11_gnn.ipynb](workshops/11_gnn.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/dzNXCBOHc9f5pw) | [09_binding_affinity.ipynb](assignments/09_binding_affinity.ipynb) |
| 12. Генеративные модели: GAN, VAE | [12_gan.ipynb](workshops/12_gan.ipynb) <br> [13_vae.ipynb](workshops/13_vae.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/fLV_UU1hM-IJ9w) | [10_gan.ipynb](assignments/10_gan.ipynb) <br> [11_vae.ipynb](assignments/11_vae.ipynb) |
| 13. Генеративные модели: DDPM | [14_ddpm.ipynb](workshops/14_ddpm.ipynb) <br> [Запись практики](https://disk.yandex.ru/i/6kCIvqFESpt8Fw) | [12_ddpm_ddim.ipynb](assignments/12_ddpm_ddim.ipynb) |

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
