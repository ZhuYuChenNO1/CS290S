import csv
import os
import torch
import torch.nn as nn

def get_data(path):
    csv_data = []
    with open(path, encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            csv_data.append(row)
    data = csv_data[1:]
    return data

def set_tokenizer_dict(data):
    all_letters = []
    for item in data:
        all_letters += item[0]
    all_letters = sorted(list(set(all_letters)))
    all_letters = ''.join(all_letters)
    n_letters = len(all_letters)
    
    return n_letters, all_letters

# Find letter index from all_letters
def letterToIndex(all_letters, letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1> Tensor
def letterToTensor(all_letters, letter):
    tensor = torch.zeros(1, dtype=torch.long)
    tensor[0] = letterToIndex(all_letters, letter)
    return tensor

# Turn a line into a <line_length x 1> index vectors,
def lineToTensor(all_letters, line):
    tensor = torch.zeros(len(line), 1, dtype=torch.long)
    for idx, letter in enumerate(line):
        tensor[idx] = letterToIndex(all_letters, letter)
    return tensor
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Reviews_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, all_letters):
        self.data = data
        self.all_letters = all_letters

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        TODO: Implement the __getitem__ function to return a tuple of tensors 
        containing the text review and its sentiment category.
        
        Input:
            `index`: Index of the data sample to retrieve.
        Output:
            `(line_tensor, category_tensor)`: Tuple containing two tensors:
                - line_tensor: Tensor representing the text review, 
                encoded based on all characters in `reviews.csv`.
                - category_tensor: Tensor representing the sentiment category 
                of the review, should be a float tensor indicating sentiment 
                (e.g., 1.0 for positive sentiment, 0.0 for negative sentiment).
              
        Hint: Make good use of `lineToTensor()`. The completed functionality 
        is similar to a simple tokenizer.
        """
        # =============================================================================
        # Your Code Here
        line_tensor, category_tensor = lineToTensor(self.all_letters, self.data[index][0]),\
                                        torch.tensor([[int(self.data[index][1])]], dtype=torch.float32)
        # =============================================================================
        return (line_tensor, category_tensor)
    
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.i2h = MLP(embedding_size + hidden_size, hidden_size, hidden_size, act_layer=nn.ReLU, drop=0)
        self.i2h = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        hidden = self.initHidden().to(input.device)
        embeds = self.embedding(input)
        for i in range(embeds.shape[0]):
            combined = torch.cat((embeds[i], hidden), 1)
            hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.sigmoid(output)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

"""
# TODO: Design or use others custom model for project_1 task.
# """
# # =============================================================================
# # Replace with your designed model
# class YourModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         """
#         Initialize the custom neural network
        
#         Args:
#             input_size (int): Dimensionality of the input features
#             hidden_size (int): Size of the hidden layer
#             output_size (int): Number of output classes
#         """
#         super(YourModel, self).__init__()
#         # Fully connected layer from input to hidden
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         # ReLU activation function
#         self.relu = nn.ReLU()
#         # Fully connected layer from hidden to output
#         self.fc2 = nn.Linear(hidden_size, output_size)
        
#     def forward(self, input):
#         """
#         Define the forward propagation process
        
#         Args:
#             x (torch.Tensor): Input data tensor
        
#         Returns:
#             torch.Tensor: Output of the neural network
#         """
#         # Linear transformation from input to hidden
#         x = self.fc1(x)
#         # ReLU activation function
#         x = self.relu(x)
#         # Linear transformation from hidden to output  
#         x = self.fc2(x)
        
#         return x
# # =============================================================================