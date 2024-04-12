# Sentence Sentiment Classification with Deep Learning
Note: All scripts must be run in the `project1_sequence_classification` folder.

## Requirements

1. Create a new conda environment:
```sh
conda create --name <myenv>
```
2. Activate the environment:
```sh
conda activate <myenv>
```
3. Install PyTorch:
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
4. Install the other libraries by pip:
```sh
pip install -r requirements.txt
```

If you're interested in the relationship between [conda](https://docs.conda.io/en/latest/) and [pip](https://pip.pypa.io/en/stable/), you can find more information at [here](https://www.anaconda.com/blog/understanding-conda-and-pip).


## Task
The goal is to predict the expressed sentiment in a given sentence. **In this project, we only consider positive as 1 and negative as 0.**
```
太棒了!太棒了!简直棒得不能言语了，国内动漫电影, 1
```

## Code Flow
1. Initiate the environment parameter
```python
set_seed(2024)
device = torch.device("cuda")
```

2. Pre-processing text

Normally, this part involves text cleaning, but here we're starting directly with building the tokenizer dictionary.
```python
n_letters, all_letters = set_tokenizer_dict(data)
```
`n_letters` represents the size of the tokenizer dictionary, used for model setting; `all_letters` is the tokenizer dictionary. In project 1, we simply use a `str` as a dictionary; the position index represents the character's token.

3. Set the dataloader

Set `Reviews_dataset` for data loading, including the "text->token" preprocess.
```python
dataset = Reviews_Dataset(data, all_letters)
```
Then, split the training and validation data. We recommend splitting train:valid as 8:1, since we took 10% of the data as test data.
```python
training_dataset, validation_dataset, test_dataset = Data.random_split(dataset, [4000, 500, 500])
```
Set dataloader: You must modify this step based on the GPU utilization method.
```python
train_dataloader = Data.DataLoader(training_dataset, shuffle=True)
...
```

4. Initiate the model
```python
rnn = RNN(vocab_size=n_letters, embedding_size=256, hidden_size=256)
rnn.to(device)
```

5. Set the training hyperparameter
```python
num_epoch = ...
criterion = ...
optimizer = ...
...
```

6. Training loop
```python
for epoch in epochs:
    train()
    validation()
test()
```

## Code block you need to fill

1. `__getitem__()` method in the dataset class, in `model.py`.

    - Implement the `__getitem__` function to return a tuple of tensors
The document includes the text review and sentiment categories.

2. Metirc calculation in `eval()`, in `train.py`.

    - Calculate the loss using an appropriate loss function and send it to
    writer

    - Calculate the precision, recall, and F1 score using the appropriate function
    and sent to the writer, the result could be a 1-dim tensor or float

3. Design or use others' models; replace class `RNN()` in `model.py`.

## After filling in the block

1. Open a command-line interface (Terminal or Command Prompt).
2. Use the cd command to navigate to the directory containing the train.py file.
```bash
cd /path/to/project_1/directory
```
3. Enter the following command to execute the training program:
```python
python train.py
```
This will run the training program with default parameters.

4. During training, enter the following command to check the test result on the [tensorboard](https://pytorch.org/docs/stable/tensorboard.html):
```bash
tensorboard --logdir=<output path> --port <available ports>
```


# Guidelines for more advanced use
- change suitable hyperparameters for the model and training loop.
- change the simple tokenizer into a suitable Chinese tokenizer.
- change the ratio between the training set and the validation set (previously 8:1).

# Resources
We hope this guide helps you successfully run the neural network training program! If you encounter any issues during usage, feel free to reach out to us for assistance.

- [PyTorch documentation](https://pytorch.org/docs/1.2.0/)
- Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems 26 (2013).
- Yoon Kim. 2014. Convolutional Neural Networks for Sentence Classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746–1751, Doha, Qatar. Association for Computational Linguistics.

