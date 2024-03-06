from model import VGG

def vgg11(): return VGG("A")
def vgg11_LRN(): return VGG("A-LRN")
def vgg13(): return VGG("B")
def vgg16_1(): return VGG("C")
def vgg16(): return VGG("D")
def vgg19(): return VGG("E")

def get_model(vgg: str):
    models = {
        "vgg11": vgg11,
        "vgg11_LRN": vgg11_LRN,
        "vgg13": vgg13,
        "vgg16_1": vgg16_1,
        "vgg16": vgg16,
        "vgg19": vgg19
    }
    
    if vgg in models:
        return models[vgg]()
    else:
        raise ValueError(f"Invalid VGG: {vgg}")