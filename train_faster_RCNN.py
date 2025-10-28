from voc_dataset import VOCDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def train():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    transform = ToTensor()
    train = VOCDataset(root='./VOC2012', year='2012', image_set='train', download=False, transform=transform)
    
    train_dataloader = DataLoader(
        dataset=train,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    for images, labels in train_dataloader:
        images = [image.to(device) for image in images]
        labels = [{k: v.to(device) for k, v in target.items()} for target in labels]

        # Lan truyền tiến
        losses = model(images, labels)
        final_loss = sum(loss for loss in losses.values())
        print(final_loss)
        
        # Lan truyền lui
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
if __name__ == '__main__':
    train()
