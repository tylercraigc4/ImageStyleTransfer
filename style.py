import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.models.vgg import vgg19
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

model = vgg19(pretrained=True).features # gives us all the conv layers we need
print(model)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28'] # features of VGG network that we want
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
image_width = 1080
image_height = 720

loader = transforms.Compose(
    [
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor()
    ]
)

original_img = load_image('san_diego_temple.jpg')
style_img = load_image('modern_art.jpg')

print('===================')
print(original_img.shape)
print(style_img.shape)
print('===================')

model = VGG().to(device).eval() # so we can freeze the weights of the model
generated = original_img.clone().requires_grad_(True) # we're only optimizing on the generated image

# hyperparameters
total_steps = 6000
learning_rate = 0.001
alpha = 1 # for content loss
beta = 0.01 # how much style in the image
optimizer = optim.Adam([generated], lr=learning_rate) # only optimizing on the generated image

for step in range(total_steps): # how many times the image will be modified
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = 0
    original_loss = 0

    # iterate through all the features for the chosen layers
    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)
        
        # compute Gram Matrix
        # multiply every pixel value from each channel with every other channel for 
        # generated features which will yield a matrix with shape channel x channel
        # this is going to be subtracted from the style Gram matrix
        # the Gram matrix calculates some sort of correlation matrix
        # if the pixel colors are similar across the generated image AND style image 
        # then that results in the pictures having a similar style to each other
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )

        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )

        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated, 'generated.png')
