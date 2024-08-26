import torch
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Subset
from torch.nn.functional import mse_loss
from sys import argv
from os import makedirs
from tqdm import tqdm
from time import time
from dataset import SlidingWindow, PollutionDataset, TARGET_COLUMNS
from models import ModelGRU, ModelLSTM
from typing import Union

SEED = 42
gen = torch.Generator()
gen.manual_seed(SEED)

models_dict = {
    'lstm': ModelLSTM,
    'bilstm': ModelLSTM,
    'gru': ModelGRU
}

def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    SS_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    SS_tot = torch.sum((y_true - y_true.mean(dim=0)) ** 2, dim=0)
    return 1 - SS_res / SS_tot

def training_loop(train_data: Subset[PollutionDataset], model: Union[ModelLSTM, ModelGRU],
                  optimizer: Adam, month, year, epochs: int = 10, batch_size: int = 256) -> tuple[float, float]:
    model.train()
    train_time = time()

    losses = []
    pbar = tqdm(range(epochs), desc=f'Training for {month}/{year}', position=1, leave=False)
    for _ in pbar:
        for X, y in DataLoader(train_data, batch_size, shuffle=True, generator=gen):
            optimizer.zero_grad()
            y_hat = model(X)
            #loss = mse_loss(y_hat, y)
            loss = (y_hat - y).pow(2).mean(0).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        pbar.set_postfix({'mse_loss': sum(losses) / len(losses)})

    train_time = time() - train_time
    return train_time, sum(losses) / len(losses)

def evaluate(test_data: Subset[PollutionDataset], model: Union[ModelLSTM, ModelGRU]) -> dict:
    model.eval()
    pred = torch.empty((len(test_data),6))
    target = torch.empty((len(test_data),6))

    test_time = time()
    for i, (X, y) in enumerate(DataLoader(test_data, batch_size=1024)):
        with torch.no_grad():
            y_hat = model(X)

        pred[i * 1024:(i + 1) * 1024,:] = y_hat
        target[i * 1024:(i + 1) * 1024,:] = y

    rmse = torch.sqrt(((pred - target) ** 2).mean(dim=0))
    mae = (torch.abs(pred - target)).mean(dim=0)
    r2 = r2_score(pred, target)
    test_time = time() - test_time

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'test_time': test_time}

def main(model_name: str = 'lstm', window_size: int = 6, forget: bool = False):
    assert model_name in models_dict.keys(), f'Invalid model: {model_name}'

    model_path = f'output/model_{model_name}_window_size={window_size}_forget={forget}'
    file_path = f'output/results_{model_name}_window_size={window_size}_forget={forget}.csv'
    makedirs(model_path, exist_ok=True)

    f = open(file_path, 'w')
    f.write('month,year,')
    for x in TARGET_COLUMNS:
        f.write(f'{x}_R2,{x}_RMSE,{x}_MAE,')
    f.write('R2,RMSE,MAE,train_exec_time,test_exec_time\n')
    f.close()

    current_window = SlidingWindow('partial.csv', window_size)

    print(f'Seed: {SEED}')
    print(f'Model: {model_name}')
    print(f'Window size: {window_size} months')
    print(f'Forget: {forget}')
    print('Starting training...')

    start = True
    pbar = tqdm(current_window, total=len(current_window), position=0, leave=False)
    for train_data, test_data in pbar:
        month, year = current_window.get_month_year()

        # Initialize model and optimizer
        if start or forget:
            model = models_dict[model_name](current_window.n_features, 6, bidirectional=model_name == 'bilstm')
            optimizer = Adam(model.parameters(), lr=0.001)
            start = False

        # Train and evaluate
        train_time, loss = training_loop(train_data, model, optimizer, month, year)
        metrics = evaluate(test_data, model)

        # Save results
        f = open(file_path, 'a')
        f.write(f'{month},{year},')
        for i, x in enumerate(TARGET_COLUMNS):
            f.write(f'{metrics["R2"][i]},{metrics["RMSE"][i]},{metrics["MAE"][i]},')
        f.write(f'{metrics["R2"].mean().item()},{metrics["RMSE"].mean().item()},{metrics["MAE"].mean().item()},{train_time},{metrics["test_time"]}\n')
        f.close()

        # Save model
        torch.save(model.state_dict(), f'{model_path}/{model_name}_{month}-{year}.pt')

        pbar.set_postfix({'curr_loss': loss})

if __name__ == '__main__':
    if len(argv) > 3:
        main(argv[1], int(argv[2]), bool(argv[3]))
    else:
        print('Usage: python train.py <model_name> <window_size> <forget>')
