import os
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as Data

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model import get_data, set_tokenizer_dict, Reviews_Dataset, RNN, YourModel
from utils import set_seed


def train(model, optimizer, loss_fn, dataloader, writer, current_epoch_index, device):
    """The training process of training the model.

    Args:
        - model (torch.nn.Module): the neural network
        - optimizer (torch.optim): optimizer for parameters of model
        - loss_fn: a function that takes batch_output and batch_labels and 
        computes the loss for the batch
        - dataloader (Data.DataLoader): a generator that generates batches 
        of data and labels
        - writer (SummaryWriter): a writer for recording results in tensorboard.
        - current_epoch_index (int): epoch counter for writer
        - device (str): sent Tensor to the specified device

    """
    model.train()

    # `n_iters` corresponds to the update frequency of the tqdm bar description. 
    # `epoch_losses` is used to store the loss of all data throughout the entire 
    # epoch, which is used to calculate the average loss of the epoch.
    n_iters = 0
    epoch_losses = []

    with tqdm.tqdm(total=len(dataloader)) as pbar:
        # `line_tensor` represents the sentence embedding, `category_tensor` represents
        # the ground label
        for line_tensor, category_tensor in dataloader:
            # Sent Tensor to the specified device. To delve deeper into this process, 
            # there is a question. why is there a [0] here?
            category_tensor = category_tensor[0].to(device)
            line_tensor = line_tensor[0].to(device)

            # The core of training process, including model prediction, loss calculation, 
            # and backpropagation.
            optimizer.zero_grad()
            output = model(line_tensor)
            loss = loss_fn(output, category_tensor)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # Update the tqdm bar description for every 10 batches, then update tqdm 
            # process bar and batch counter `n_iters`
            if n_iters % 10 == 0:
                pbar.set_description(
                    f"Train| Epoch: {current_epoch_index}, loss: {loss:.3f}"
                )
            pbar.update(1)
            n_iters += line_tensor.shape[0]

        # Appending the loss value of the current batch to the epoch_losses list, 
        # calculating the average loss, and passing it to the writer
        epoch_average_loss = torch.tensor(epoch_losses).mean().item()
        writer.add_scalar("train/loss", epoch_average_loss, current_epoch_index)


def eval(model, loss_fn, dataloader, threshold, device, bar_description: str, current_epoch_index=0):
    """The eval process of training the model. Used for validation() and
    test() function

    Args:
        - model (torch.nn.Module): The neural network
        - loss_fn: A function that takes batch_output and batch_labels and 
        computes the loss for the batch
        - dataloader (Data.DataLoader): A generator that generates batches 
        of data and labels
        - threshold (float): Convert network predicted probabilities into 
        positive (greater than threshold) and negative (less than threshold).
        - device (str): Sent Tensor to the specified device
        - bar_description (str): Set tqdm processing bar description
        - current_epoch_index (int): Epoch counter for writer
        
    Returns:
        Dict: The metrics calculated for all data, including precision, 
        recall, and F1 score.
    """
    model.eval()

    # `preds` and `gts` respectively store model outputs and ground label, 
    # used for calculating the final precision, recall, and F1 score.
    # `epoch_losses` is used to store the loss of all data throughout the entire 
    # epoch, which is used to calculate the average loss of the epoch.
    preds = torch.zeros(len(dataloader), dtype=torch.int32)
    gts = torch.zeros(len(dataloader), dtype=torch.int32)
    epoch_losses = []

    with tqdm.tqdm(total=len(dataloader)) as pbar:
        pbar.set_description(f"{bar_description}| Epoch: {current_epoch_index}")
        for idx, (line_tensor, category_tensor) in enumerate(dataloader):
            # Sent Tensor to the specified device. To delve deeper into this process, 
            # there is a question. why is there a [0] here?
            category_tensor = category_tensor[0].to(device)
            line_tensor = line_tensor[0].to(device)
            
            # The core of evaluation process, including model prediction and
            # store results and labels in `preds` and `gts`
            output = model(line_tensor)
            pred = (output > threshold).int().item()
            preds[idx] = pred
            gts[idx] = category_tensor.item()
            
            """
            TODO: Calculate the loss using appropriate loss function and sent to
            writer
            
            Input:
                `output` (tensor): The model's predicted results
                `category_tensor` (tensor): The ground truth label
            Output:
                `loss` (tensor|float)
            
            Hint: You may want to use a loss function like 
            `torch.nn.BCEWithLogitsLoss()` for binary classification tasks
            """
            # =============================================================================
            # Your Code Here
            loss = loss_fn(output, category_tensor)
            epoch_losses.append(loss)
            # =============================================================================
            # Update tqdm process bar
            pbar.update(1)

    # Compute the average loss of the epoch
    epoch_average_loss = torch.tensor(epoch_losses).mean().item()

    """
    TODO: Calculate the precision, recall, and F1 score using appropriate function 
    and sent to writer, the result could be 1-dim tensor or float 
    
    Input:
        `pred` (int): the predict result (1 for Positive, 0 for negative)
        `category_tensor` (tensor): The ground truth label
    Output:
        `precision`, `recall`, `f1_score`: the performance of model evaluate on 
        the evaluation data.
    
    Hint: Make good use of `preds` and `gts` for storing predictions (preds) 
    and ground truths (gts) for each batch.
    """
    # =============================================================================
    # Your Code Here
    # category_tensor.to(device)
    # preds.to(device)
    TP = torch.sum((preds == 1) & (gts == 1))
    TN = torch.sum((preds == 0) & (gts == 0))
    FP = torch.sum((preds == 1) & (gts == 0))
    FN = torch.sum((preds == 0) & (gts == 1))
    
    print(TP,TN, FP,FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2*(precision*recall)/(precision+recall)
    print(f"f1_score:{f1_score}")
    # =============================================================================
    
    return {
        "f1_score": f1_score.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "loss": epoch_average_loss,
    }

@torch.no_grad()
def validation(model, loss_fn, dataloader, threshold, writer, current_epoch_index, device):
    """The validation process of training the model.

    Args:
        - model (torch.nn.Module): the neural network
        - dataloader (Data.DataLoader): a generator that generates batches 
        of data and labels
        - threshold (float): Convert network predicted probabilities into 
        positive (greater than threshold) and negative (less than threshold).
        - writer (SummaryWriter): a writer for recording results in tensorboard.
        - current_epoch_index (int): epoch counter for writer
        - device (str): sent Tensor to the specified device

    Returns:
        Dict: The metrics calculated for all data, including precision, 
        recall, and F1 score.
    """
    valid_result = eval(
        model=model,
        loss_fn=loss_fn,
        dataloader=dataloader,
        threshold=threshold,
        # writer=writer,
        device=device,
        bar_description='Valid',
        current_epoch_index=current_epoch_index,
    )
    
    # print(f'f1_socre:{valid_result["f1_score"]}')
    writer.add_scalar("Valid/f1 score", valid_result["f1_score"], current_epoch_index)
    writer.add_scalar("Valid/loss", valid_result["loss"], current_epoch_index)
    
    return valid_result


@torch.no_grad()
def test(model, loss_fn, dataloader, threshold, writer, device):
    """The test process of training the model.

    Args:
        - model (torch.nn.Module): the neural network
        - dataloader (Data.DataLoader): a generator that generates batches 
        of data and labels
        - threshold (float): Convert network predicted probabilities into 
        positive (greater than threshold) and negative (less than threshold).
        - writer (SummaryWriter): a writer for recording results in tensorboard.
        - device (str): sent Tensor to the specified device

    Returns:
        Dict: The metrics calculated for all data, including precision, 
        recall, and F1 score.
    """
    test_result = eval(
        model=model,
        loss_fn=loss_fn,
        dataloader=dataloader,
        threshold=threshold,
        # writer=writer,
        device=device,
        bar_description='Test',
    )
    
    # print(f'f1_socre:{test_result["f1_score"]}')
    writer.add_scalar("Test/f1 score", test_result["f1_score"], 0)
    writer.add_scalar("Test/loss", test_result["loss"], 0)
    
    return test_result


def main():
    # Setting up experimental environment parameters
    set_seed(2024)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    # Setting up the database Dataset class and DataLoader class
    data = get_data("data/reviews.csv")
    n_letters, all_letters = set_tokenizer_dict(data)
    dataset = Reviews_Dataset(data, all_letters)
    training_dataset, validation_dataset, test_dataset = Data.random_split(
        dataset, [4000, 500, 500]
    )
    train_dataloader = Data.DataLoader(training_dataset, shuffle=True)
    valid_dataloader = Data.DataLoader(validation_dataset, shuffle=False)
    test_dataloader = Data.DataLoader(test_dataset, shuffle=False)
    
    # for item in training_dataset:
    #     print(item)
    # Setting the threshold for binary classification prediction probabilities
    training_categories = torch.tensor([item[1] for item in training_dataset])
    training_categories_0_proportion = 2*(training_categories == 0).sum() / len(
        training_categories
    )
    # training_categories_0_proportion = 0.9
    # print('######')
    # print(training_categories_0_proportion)

    """
    TODO: Instantiate the class YourModel from model.py, replace the given RNN model
    
    Hint: args please refer to the class YourModel in model.py file 
    and the network architecture you designed
    """
    # =============================================================================
    # Replace the code here
    model = RNN(vocab_size=n_letters, embedding_size=256, hidden_size=256,)
    model.to(device)
    # =============================================================================

    # Setting up the parameters required for the training process, including 
    # optimizer, loss function, and logger
    num_epoch = 30
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    writer = SummaryWriter(
        # log_dir=f"2024_4_12"
        log_dir=f"output/logs/{datetime.now().strftime('%b%d_%H-%M-%S')}"
    )
    
    # The actual training and testing section for the model, setting up for 
    # 'num_epoch' epochs, where each epoch includes training with train() and 
    # validation(), followed by testing with test() after all epochs are completed.
    for current_epoch in range(num_epoch):
        # validation(
        #     model=model,
        #     loss_fn=criterion,
        #     dataloader=valid_dataloader,
        #     threshold=training_categories_0_proportion,
        #     writer=writer,
        #     current_epoch_index=current_epoch,
        #     device=device,
        # )
        train(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            dataloader=train_dataloader,
            writer=writer,
            current_epoch_index=current_epoch,
            device=device,
        )
        validation(
            model=model,
            loss_fn=criterion,
            dataloader=valid_dataloader,
            threshold=training_categories_0_proportion,
            writer=writer,
            current_epoch_index=current_epoch,
            device=device,
        )
    test(
        model=model,
        loss_fn=criterion,
        dataloader=test_dataloader,
        threshold=training_categories_0_proportion,
        writer=writer,
        device=device,
    )


if __name__ == "__main__":
    main()