# StoryGAN Implementation

## Dataset Preparation

First importing necessary libraries

```
torch matplotlib
```

Using CLEVR-SV and Pororo-SV Dataset. 

Text Processing, we use ```sentence-transformers``` or Universal Sentence Encoder for 128-dimensional embeddings. 

```
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('$your-sentence')
st_vector = encoder.encode(sentence)
```

Story Encoder

Aim is to encode whole story into a vector $h_0$ thats of a distribution.

The input story has $T$ sentence embeddings. Concatenate them into a single vector.
Pass the concatenated vector through two separate MLPs to compute mean and the variance. Then, we apply the parameterization trick.
We use KL Divergene to measure the loss function in the normal distribution generation.

So as of now we have distribution and we sample $h_0$ from that.

We create minibatches of ${s_t, S, x_t}$ and $(S,X)$ using 
importation and Dataloader of pytorch.utils.data

```
train_dataset = TensorDataset(arg1, arg2)
train_loader = DataLoader(trainingdataset,batch_size=size_of_batch, shuffle=bool)
```
then we define the neural netwrok models as a class with parameter as nn.Module
then create the constructor and the forward pass fuction.

```
class <name>(nn.Module):
    def __init__(self):
        super().__init__()
        self.<layer1> = nn.Linear(input, output) or other nn. classes
        <include other layers here and then initialize with weights>
        nn.init.xavier_normal_
        nn.init.zeros_(self.<layerno.>.<weight/bias>)

    def forward(self, x):
        <do the things>
        x=torch.relu(self.fc1(x))
        return x
```

Move the model to the GPU and then create an object
define the criterion and optimizer that we are going to use 

```
model = SimpleNN().to(device)
criterion = nn.<which loss>()
optimizer = optim.Adam(model.parameters(), lr = <amt>)
```

Next we train the model, we move the batch to the same device as the model

```
def train(model, loader, epoch=<num>)
    model.train()
    for epoch_num in range(epoch):
        total_loss = 0
        for x,y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs=model(x)
            loss=criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        predictions=model(X)
        test_loss = criterion(predictions,y)
```