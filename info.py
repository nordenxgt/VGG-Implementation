import torch
from torchinfo import summary

from utils import get_model

def main():
    vggs = ["vgg11", "vgg11_LRN", "vgg13", "vgg16_1", "vgg16", "vgg19"]
    for vgg in vggs:
        model = get_model(vgg)
        print("=" * 90)
        print(vgg)
        print("=" * 90)
        summary(model, input_size=[1, 3, 224, 224])
        X = torch.rand((1, 3, 224, 224)).to("cuda")
        print(model(X).shape)
        print("\n\n")

if __name__ == "__main__":
    main()