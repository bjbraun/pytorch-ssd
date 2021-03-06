from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
import cv2
import sys
import os
import pathlib
from os.path import expanduser
import torch

def isImgFile(file):
    return (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".JPG")) \
           and not(file.endswith(".cs.png")) and not(file.endswith(".depth.png")) \
           and not(file.startswith("."))

if len(sys.argv) < 6:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path> <output path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
root = pathlib.Path(sys.argv[4])
output = pathlib.Path(sys.argv[5])


class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

filenames_img = [os.path.join(path, file) for path, directories, filenames in os.walk(root) for
                      file in filenames if isImgFile(file)]
filenames_img.sort()

for counter in range(len(filenames_img)):
    orig_image = cv2.imread(filenames_img[counter])
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.1)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(orig_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    path = expanduser(output / ("%s%s%s" % ("ssd_visualize_output_", counter, ".jpg")))
    print(path)
    cv2.imwrite(path, orig_image)
    print(f"Found {len(probs)} objects. The output image is {path}")
