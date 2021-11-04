import numpy as np
import torch
import cv2
import blazeface

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

front_net = blazeface.BlazeFace().to(gpu)
front_net.load_weights("blazeface.pth")
front_net.load_anchors("anchors.npy")

def cv_point(point):
    return [int(p) for p in point]


def show_detection(image, detection):
    detection = detection.detach().cpu().numpy()
    confidence = detection[..., -1:]
    positions = detection[..., :-1].reshape(detection.shape[0], 8, 2) * image.shape[1::-1]

    for face in positions:
        cv2.rectangle(image, cv_point(face[0]), cv_point(face[1]), (0, 255, 0))
        for p in range(2, len(face)):
            cv2.circle(image, cv_point(face[p]), 3, (0, 255, 0))


def predict(image_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (128, 128))

    detection = front_net.predict_on_image(rgb)

    show_detection(image, detection)

    cv2.imshow('blaze-face', image)
    cv2.waitKey(0)


images = [
    r"D:\blob\voxceleb1-fomm\train\id10376#mk1UmZlF7XU#007079#007762.mp4\0000174.png",
    r"D:\blob\voxceleb1-fomm\train\id10715#pCkz5N0a0RY#002578#002823.mp4\0000082.png",
    r"D:\blob\voxceleb1-fomm\train\id11117#JSzJC7Ytf_8#007300#007647.mp4\0000071.png",
    r"D:\blob\voxceleb1-fomm\train\id11181#K0UVl6K5FxY#000531#000626.mp4\0000095.png",
    r"D:\blob\voxceleb1-fomm\train\id10036#NKuVgEn0sr8#004165#004265.mp4\0000098.png",
    r"D:\blob\voxceleb1-fomm\train\id11211#zWE8NG5oSOA#001852#002027.mp4\0000208.png",
    r"D:\blob\voxceleb1-fomm\train\id10231#PyZ7LqrciCs#003683#003783.mp4\0000096.png",
    r"D:\blob\voxceleb1-fomm\train\id10311#ntfEFDmQ8Bk#002852#003248.mp4\0000083.png"
]

for image in images:
    predict(image)
