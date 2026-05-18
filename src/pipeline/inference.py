import torch
import cv2
import numpy as np


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess(cell, transform):
    return transform(image=cell)["image"].unsqueeze(0).float()


def predict_cell(model, cell, transform, device):
    x = preprocess(cell, transform).to(device)

    with torch.no_grad():
        pred = model(x)
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    return pred