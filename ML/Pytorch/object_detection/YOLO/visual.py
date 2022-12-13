import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import Yolov1
from loss import YoloLoss
from PIL import Image

LEARNING_RATE = 2e-5
DEVICE = "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "./outputs/Zee_YOLO.pth.tar"
IMG_DIR = "data/data/images"
LABEL_DIR = "data/data/labels"
counter = 0
conv_layers = []
conv_weights = []
outputs = []
names = []
processed = []

model = Yolov1().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = YoloLoss()

checkpoint = torch.load(LOAD_MODEL_FILE, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["optimizer"])

model_children = list(model.children())
for i in range(len(model_children)):
    for child in model_children[i].children():
        if type(child) == nn.Conv2d:
            conv_layers.append(child)
            conv_weights.append(child.weight)
            counter += 1

#print(f"Total Convolutional Layers: {counter}")

plt.figure(figsize=(20, 17))
for i, filter in enumerate(conv_weights[0]):
  plt.subplot(8, 8, i+1)
  plt.imshow(filter[0, :, :].detach(), cmap='gray')
  plt.axis('off')
  plt.savefig('./outputs/filter_layer1.png')
plt.show()


transform = transforms.Compose([
transforms.Resize((448, 448)),
transforms.ToTensor(),
transforms.Normalize(mean=0., std=1.)])

image = Image.open("cat.jpg")
model = model.to(DEVICE)

image = transform(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(DEVICE)

for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))

#print(len(outputs))

#for feature_map in outputs:
#    print(feature_map.shape)

for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

fig = plt.figure(figsize=(100, 100))
for i in range(len(processed)):
    a = fig.add_subplot(5, 5, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str('./outputs/feature_maps.jpg'), bbox_inches='tight')

# for weight, conv in zip(conv_weights, conv_layers):
#     print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

#print("Success!")
