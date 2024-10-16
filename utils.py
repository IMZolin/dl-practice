import torch


def remove_consecutive_spaces(string):
    return ' '.join(string.split())


def chars(f, t):
    return list(map(chr, range(ord(f), ord(t)+1)))

valid_chars = set(chars('а', 'я') + ['ё'] + chars('a', 'z') + chars('0', '9')
                  + list("<>(){}" + ".,\"!?;:-*—\'") )


def clean_text(text):
    text = text.lower()
    text = ''.join([char if char in valid_chars else ' ' for char in text])
    text = remove_consecutive_spaces(text)
    return text


def read_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


def dist(X1, X2):
    d = (X1 - X2) ** 2
    return d.mean()


def gradient_descent(x_noisy, w, h, num_iterations, learning_rate):
    x_noisy = x_noisy.float()
    losses = []
    for i in range(num_iterations):
        x_pred = w.matmul(h)
        grad_w = 2 * (x_pred - x_noisy).matmul(h.T)
        grad_h = 2 * w.T.matmul(x_pred - x_noisy)
        w -= learning_rate * grad_w
        h -= learning_rate * grad_h
        if i % 100 == 0:
            current_loss = dist(x_pred, x_noisy).item()
            losses.append(current_loss)
    return losses


def sgd(params, loss_fn, num_iterations, learning_rate, stop_loss_value=-0.007, fix_ends=True, callback=None):
    for i in range(num_iterations):
        for param in params:
            if param.grad is not None:
                param.grad.zero_()
        loss = loss_fn(*params)
        loss.backward()

        with torch.no_grad():
            for param in params:
                if fix_ends:
                    param[1:-1] -= learning_rate * param.grad[1:-1]
                else:
                    param -= learning_rate * param.grad
        if callback is not None:
            callback(i, loss.item())
        if stop_loss_value is not None and loss.item() < stop_loss_value:
            print(f"Iteration {i}: Reached stop loss value = {stop_loss_value}. Stopping optimization.")
            break

    return params


if __name__ == '__main__':
    print(clean_text(read_file('./data/books_txt/AKSAKOW/bagrov.txt')))