import torchvision
def wide_resnet50_2():
    model = torchvision.models.wide_resnet50_2(pretrained=True)
    model.eval()
    return model
