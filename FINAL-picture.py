import argparse
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from config import config as conf
from model import FaceMobileNet
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from utils.torch_utils import select_device


def group_image(images, batch) -> list:
    """Group image paths by batch size"""
    size = len(images)
    res = []
    for i in range(0, size, batch):
        end = min(batch + i, size)
        res.append(images[i: end])
    return res


def _preprocess(images: list, transform) -> torch.Tensor:
    res = []
    for img in images:
        im = Image.open(img)
        im = transform(im)
        res.append(im)
    data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    data = data[:, None, :, :]  # shape: (batch, 1, 128, 128)
    return data


def featurize(images: list, transform, net, device) -> dict:
    """featurize each image and save into a dictionary
    Args:
        images: image paths
        transform: test transform
        net: pretrained model
        device: cpu or cuda
    Returns:
        Dict (key: imagePath, value: feature)
    """
    data = _preprocess(images, transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data)
    res = {img: feature for (img, feature) in zip(images, features)}
    return res


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def dynamic_resize(shape, stride=64):
    max_size = max(shape[0], shape[1])
    if max_size % stride != 0:
        max_size = (int(max_size / stride) + 1) * stride
    return max_size


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def show_results(img, xywh, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl * 2, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl + 5, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    # label = str(int(class_num)) + ': ' + str(conf)[:5]
    label = class_num + ': ' + str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl, [225, 255, 255], thickness=tf * 2, lineType=cv2.LINE_AA)
    return img


def detect(model, img0):
    stride = int(model.stride.max())  # model stride
    imgsz = opt.img_size
    if imgsz <= 0:  # original size
        imgsz = dynamic_resize(img0.shape)
    imgsz = check_img_size(imgsz, s=64)  # check img_size
    img = letterbox(img0, imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]
    # Apply NMS
    pred = non_max_suppression_face(pred, opt.conf_thres, opt.iou_thres)[0]
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
    gn_lks = torch.tensor(img0.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
    boxes = []
    h, w, c = img0.shape
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        pred[:, 5:15] = scale_coords_landmarks(img.shape[2:], pred[:, 5:15], img0.shape).round()
        for j in range(pred.size()[0]):
            xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()
            conf = pred[j, 4].cpu().numpy()
            landmarks = (pred[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
            class_num = pred[j, 15].cpu().numpy()
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            # boxes.append([x1, y1, x2-x1, y2-y1, conf])
            boxes.append([xywh, conf, landmarks, class_num])
    return boxes


def compute_accuracy(feature_dict, frame, test_root, model, transform, device):
    pairs = ["cao", "minnie", "song", "tian", "ye"]

    similarities = []
    labels = []

    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = transform(frame)

    data = torch.cat([res], dim=0)  # shape: (batch, 128, 128)
    data = data[:, None, :, :]  # shape: (batch, 1, 128, 128)

    data = data.to(device)
    model = model.to(device)

    with torch.no_grad():
        feature1 = model(data)
    feature1 = feature1.cpu().numpy()

    for pair in pairs:
        a_similarities = []

        for i in range(7):
            img2 = f"{test_root}\\{pair}/{pair}_{i}.jpg"
            feature2 = feature_dict[img2].cpu().numpy()
            similarity = cosin_metric(feature1, feature2)
            a_similarities.append(similarity.tolist()[0])

        # print(a_similarities)
        # similarities.append((sum(a_similarities) - min(a_similarities) - max(a_similarities)) / (len(a_similarities) - 2))
        similarities.append(sum(a_similarities) / len(a_similarities))
    # accuracy, threshold, label = threshold_search(similarities, labels)
    # return accuracy, label

    return max(similarities), similarities.index(max(similarities))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s-face.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    # load recognition model
    model1_list = ["mian", "ni", "qi", "juan", "shua"]

    model1 = FaceMobileNet(conf.embedding_size)
    model1 = nn.DataParallel(model1)
    model1.load_state_dict(torch.load(conf.test_model, map_location=conf.device))
    model1.eval()

    with open(conf.test_list, 'r') as fd:
        pairs = fd.readlines()
    images = []
    for pair in pairs:
        id1, _ = pair.split()
        images.append(id1)

    images = [osp.join(conf.test_root, img) for img in images]
    groups = group_image(images, conf.test_batch_size)

    feature_dict = dict()
    for group in groups:
        d = featurize(group, conf.test_transform, model1, conf.device)
        feature_dict.update(d)

    # Load detection model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    with torch.no_grad():
        for name in ["mian", "ni", "qi", "juan", "shua","test"]:
            frame = cv2.imread(f"{name}.jpg")

            # frame = cv2.resize(frame, (960, 540))

            boxes = detect(model, frame)
            for box in boxes:
                # -------------- add recognition --------------------------
                # box[0] = xywh
                h, w, _ = frame.shape

                length = 0.6 * max(box[0][2] * w, box[0][3] * h)

                x1 = int(box[0][0] * w - length)
                y1 = int(box[0][1] * h - length)
                x2 = int(box[0][0] * w + length)
                y2 = int(box[0][1] * h + length)

                face_frame = frame[max(0, y1):min(y2, h), max(0, x1):min(x2, w)]
                # cv2.imshow("face", face_frame)

                cv2.rectangle(frame, (max(0, x1), max(0, y1)), (min(x2, w), min(y2, h)), (0, 0, 255), thickness=3,
                              lineType=cv2.LINE_AA)

                accuracy, label = compute_accuracy(feature_dict, face_frame, conf.test_root, model1,
                                                   conf.test_transform,
                                                   conf.device)

                box[1] = accuracy  # conf
                box[3] = model1_list[label]  # class

                # if accuracy < 0.4:
                #     box[3] = "unknown"  # class
                # else:
                #     box[3] = model1_list[label]  # class
                # -------------- add recognition --------------------------
                frame = show_results(frame, box[0], box[1], box[2], box[3])

            cv2.imshow('Frame', frame)
            cv2.waitKey(2000)
        cv2.destroyAllWindows()
