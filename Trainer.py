import time
import torch


class Trainer():
    def __init__(self, loss_fn, optimizer,
                 train_dataloader, valid_dataloader,
                 unpack_batch_fn):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.unpack_batch_fn = unpack_batch_fn
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

    def train_an_epoch(self, model, log_interval):
        model.train()
        for idx, (data) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            inputs = self.unpack_batch_fn(data)
            predictions = model(*inputs)

            label = data['labels']
            loss = self.loss_fn(predictions, label)
            loss.backward()
            self.optimizer.step()
            if idx % log_interval == 0 and idx > 0:
                print(f'At iteration {idx} the loss is {loss:.3f}.')

    def train(self, model, evaluator, num_epochs, log_interval):

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            time_taken = self.train_an_epoch(model, log_interval)
            time_taken = time.time() - epoch_start_time
            with torch.no_grad():
                metric = evaluator.after_epoch(
                    epoch, model, time_taken)
            print(
                f'Epoch: {epoch}, time taken: {time_taken:.1f}s, validation accuracy: {metric:.3f}.')
