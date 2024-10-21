import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tokenizers.decoders import WordPiece
from tqdm import tqdm

decoder = WordPiece()
class DataGenerator(torch.nn.Module):
    def __init__(self, vocab_size, test_dataloader, embed_dim, num_layers, pad_token_idx, device):
        super().__init__()
        self.__embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_idx)
        self.__lstm = torch.nn.LSTM(embed_dim, embed_dim, num_layers=num_layers, batch_first=True)
        self.__fc = torch.nn.Linear(embed_dim, vocab_size)
        self.__pad_token_idx = pad_token_idx
        self.__optimizer = None
        self.__criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_idx)
        self.__device = device
        self.__test_dataloader = test_dataloader
        self.__train_losses = []
        self.__eval_losses = []

        # Move the model to the correct device (cuda or cpu)
        self.to(self.__device)

    def forward(self, x):
        embedded = self.__embedding(x)
        lstm_out, _ = self.__lstm(embedded)
        logits = self.__fc(lstm_out)
        return logits

    def __callback(self, train_loss):
        eval_loss = self.eval_model()
        self.__train_losses.append(train_loss)
        self.__eval_losses.append(eval_loss)
        print(f'Train Loss: {train_loss:.5f}, Eval Loss: {eval_loss:.5f}')

    def __train_epoch(self, train_dataloader, lr):
        self.__optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.train()
        losses = []

        for batch in tqdm(train_dataloader):
            src = batch[:, :-1].to(self.__device)
            trg = batch[:, 1:].to(self.__device)

            self.__optimizer.zero_grad()
            logits = self(src)
            loss = self.__criterion(logits.transpose(1, 2), trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.__optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        self.__callback(avg_loss)

    def train_loop(self, epochs, train_dataloader, lr, save_path):
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            self.__train_epoch(train_dataloader, lr)
            if save_path and (epoch + 1) % 10 == 0:
                self.save_model(save_path, epoch)
                self.draw_losses(save_path, epoch)

    def eval_model(self):
        self.eval()
        eval_losses = []

        with torch.no_grad():
            for batch in tqdm(self.__test_dataloader):
                src = batch[:, :-1].to(self.__device)
                trg = batch[:, 1:].to(self.__device)
                logits = self(src)
                loss = self.__criterion(logits.transpose(1, 2), trg)
                eval_losses.append(loss.item())
        return np.mean(eval_losses)

    def continues_sentence(self, sentence, tokenizer, max_len=30):
        self.eval()
        tokens = tokenizer.encode(sentence.lower(), add_special_tokens=False)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.__device)

        for _ in range(max_len):
            with torch.no_grad():
                output = self(tokens)
                next_token_logits = output[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                tokens = torch.cat((tokens, torch.tensor([[next_token]]).to(self.__device)), dim=1)
                if next_token == tokenizer.sep_token_id:
                    break

        token_ids = tokens.squeeze().tolist()
        token_list = tokenizer.convert_ids_to_tokens(token_ids)
        return decoder.decode(token_list)

    def draw_losses(self, save_path, epoch):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), self.__train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, epoch + 2), self.__eval_losses, label='Eval Loss', marker='o')

        plt.title('Loss during Training and Evaluation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, f'loss_plot_epoch_{epoch + 1}.png')
        plt.savefig(save_filename)
        plt.close()
        print(f'Loss plot saved to {save_filename}')

    def save_model(self, save_path, epoch):
        model_state = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
            'epoch': epoch,
            'train_losses': self.__train_losses,
            'eval_losses': self.__eval_losses,
        }
        save_filename = os.path.join(save_path, f'model_epoch_{epoch + 1}.pth')
        torch.save(model_state, save_filename)
        print(f'Model saved to {save_filename} at epoch {epoch + 1}')

    def load_model(self, save_path, specific_epoch=None):
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Save path '{save_path}' does not exist.")

        model_files = [f for f in os.listdir(save_path) if f.startswith('model_epoch_') and f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError(f"No saved models found in '{save_path}'.")

        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if specific_epoch is None:
            model_file = model_files[-1]
        else:
            model_file = f"model_epoch_{specific_epoch}.pth"
            if model_file not in model_files:
                raise FileNotFoundError(f"Model for epoch {specific_epoch} not found in '{save_path}'.")
        model_path = os.path.join(save_path, model_file)
        checkpoint = torch.load(model_path, map_location=self.__device)

        self.load_state_dict(checkpoint['model_state_dict'])
        if self.__optimizer:
            self.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.__train_losses = checkpoint.get('train_losses', [])
        self.__eval_losses = checkpoint.get('eval_losses', [])

        print(f"Loaded model from {model_file}")