from collections import defaultdict
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


def run_experiment(
    model_gen: Callable[[], nn.Module],
    optim_gen: Callable[[nn.Module], torch.optim.Optimizer],
    train_loader: DataLoader,
    test_loader: DataLoader,
    seed: int,
    n_epochs: int = 10,
    max_batches: int | None = None,
    verbose: bool = False,
    device: str = "cpu",
) -> dict[str, list[float] | int]:
    """Функция для запуска экспериментов.

    Args:
        model_gen (Callable[[], nn.Module]): Функция для создания модели
        optim_gen (Callable[[nn.Module], torch.optim.Optimizer]): Функция для создания оптимизатора для модели
        train_loader (DataLoader): Загрузчик обучающих данных
        test_loader (DataLoader): Загрузчик тестовых данных
        seed (int): random seed
        n_epochs (int, optional): Число эпох обучения. Defaults to 10.
        max_batches (int | None, optional): Если указано, только `max_batches` минибатчей
            будет использоваться при обучении и тестировании. Defaults to None.
        verbose (bool, optional): Выводить ли информацию для отладки. Defaults to False.
        device (str, optional): Устройство для обучения (cpu, cuda, mps). Defaults to "cpu".

    Returns:
        dict[str, list[float] | int]: словарь с метриками обучения
    """
    torch.manual_seed(seed)
    # создадим модель и выведем значение ошибки после инициализации
    model = model_gen()
    optim = optim_gen(model)
    metrics: dict[str, list[float] | int] = defaultdict(list)
    # сохраняем число параметров в модели
    metrics["n_parameters"] = sum(p.numel() for p in model.parameters())

    # обучаем модель
    for i in range(n_epochs):
        train_dict = train_epoch(
            train_loader, model, optim, max_batches=max_batches, device=device
        )
        test_dict = test_epoch(
            test_loader, model, max_batches=max_batches, device=device
        )
        train_loss, train_accuracy = train_dict["loss"], train_dict["accuracy"]
        test_loss, test_accuracy = test_dict["loss"], test_dict["accuracy"]
        if verbose:
            print(
                f"Epoch {i} train: loss = {train_loss:.4f}, accuracy = {train_accuracy:.4f}"
            )
            print(
                f"Epoch {i} test: loss = {test_loss:.4f}, accuracy = {test_accuracy:.4f}"
            )

        metrics["train_losses"].append(train_loss)
        metrics["train_accuracies"].append(train_accuracy)
        metrics["test_losses"].append(test_loss)
        metrics["test_accuracies"].append(test_accuracy)

    return metrics


def create_report(metrics: dict[str, list[float]]) -> None:
    """Функция для создания мини-отчёта об обучении модели.

    Args:
        metrics (dict[str, list[float]]): Словарь с метриками, выход из `run_experiment`.
    """
    print(f'Число параметров в модели: {metrics["n_parameters"]}')
    print()
    print(f'Минимальная ошибка:        {min(metrics["test_losses"]):.4f}')
    print(f'Максимальная точность:     {max(metrics["test_accuracies"]):.4f}')
    print()
    print(f'Ошибка в конце обучения:   {metrics["test_losses"][-1]:.4f}')
    print(f'Точность в конце обучения: {metrics["test_accuracies"][-1]:.4f}')
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 2.5))

    loss_ax.set_title("Loss")
    loss_ax.plot(metrics["train_losses"])
    loss_ax.plot(metrics["test_losses"])
    loss_ax.legend(("train", "test"), loc="upper right")

    acc_ax.set_title("Accuracy")
    acc_ax.plot(metrics["train_accuracies"])
    acc_ax.plot(metrics["test_accuracies"])
    acc_ax.legend(("train", "test"), loc="upper right")
    plt.show()


def training_step(
    batch: tuple[Tensor, Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    model.to(device=device)
    # прогоняем батч через модель
    x, y = batch
    logits = model(x.to(device=device))
    # оцениваем значение ошибки
    loss = F.cross_entropy(logits, y.to(device=device))
    # обновляем параметры
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # возвращаем значение функции ошибки для логирования
    return loss, logits


def train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_batches: int = 100,
    device: str = "cpu",
) -> dict[str, float]:
    model.train()
    loss_total = 0
    n_correct = 0
    n_total = 0
    for i, batch in enumerate(dataloader):
        x, y = batch
        loss, logits = training_step(batch, model, optimizer, device)

        # save stats
        n_total += y.size(0)
        n_correct += (y.to(device=device) == logits.argmax(dim=1)).sum().item()
        loss_total += y.size(0) * loss.item()
        if i == max_batches:
            break

    return {
        "loss": loss_total / n_total,
        "accuracy": n_correct / n_total,
    }


@torch.no_grad()
def test_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    max_batches: int = 100,
    device: str = "cpu",
) -> Tensor:
    model.eval()
    model.to(device=device)
    loss_total = 0
    n_correct = 0
    n_total = 0
    for i, batch in enumerate(dataloader):
        x, y = batch
        logits = model(x.to(device=device))
        # оцениваем значение ошибки
        loss = F.cross_entropy(logits, y.to(device=device))
        # save stats
        n_total += y.size(0)
        n_correct += (y.to(device=device) == logits.argmax(dim=1)).sum().item()
        loss_total += y.size(0) * loss.item()
        if i == max_batches:
            break

    return {
        "loss": loss_total / n_total,
        "accuracy": n_correct / n_total,
    }
