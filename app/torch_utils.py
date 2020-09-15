import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
# hyper parameters
input_size = 784 # 28x28
hidden_size = 500
output_size = 10

# load model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # out = self.sigmoid(out) 
        # multiple classisication usually use softmax

        return out
PATH = "app/mnist_ffn.pth"
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(PATH))
model.eval()

# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
                                    transforms.Resize((28, 28)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])
    
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_predication(image_tensor):
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)

    _, predications = torch.max(outputs, 1)
    return predications