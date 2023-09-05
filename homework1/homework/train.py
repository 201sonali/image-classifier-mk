try:
    from .utils import accuracy, load_data
    from .models import ClassificationLoss, model_factory, save_model
except:
    from utils import accuracy, load_data
    from models import ClassificationLoss, model_factory, save_model
import torch.optim as optim


def train(args):
    # model
    model = model_factory[args.model]()

    # loss
    loss = ClassificationLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # load data
    train_data = load_data('data/train')

    # running SGD
    runs = 10
    for run in range(runs):
        model.train()
        total_loss = 0.0  # float

        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)

            # update total loss
            run_loss = loss(outputs, labels)
            total_loss += run_loss.item()

            # backpropagate and optimize
            run_loss.backward()
            optimizer.step()

    # save model
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')

    args = parser.parse_args()
    train(args)
