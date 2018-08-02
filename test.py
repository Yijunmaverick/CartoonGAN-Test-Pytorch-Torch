import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default = 'test_img')
parser.add_argument('--load_size', default = 450)
parser.add_argument('--model_path', default = './pretrained_model')
parser.add_argument('--style', default = 'Hayao')
parser.add_argument('--output_dir', default = 'test_output')
parser.add_argument('--gpu', type=int, default = 0)

opt = parser.parse_args()

valid_ext = ['.jpg', '.png']

if not os.path.exists(opt.output_dir): os.mkdir(opt.output_dir)

# load pretrained model
model = Transformer()
model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.style + '_net_G_float.pth')))
model.eval()

if opt.gpu > -1:
	print('GPU mode')
	model.cuda()
else:
	print('CPU mode')
	model.float()

for files in os.listdir(opt.input_dir):
	ext = os.path.splitext(files)[1]
	if ext not in valid_ext:
		continue
	# load image
	input_image = Image.open(os.path.join(opt.input_dir, files)).convert("RGB")
	# resize image, keep aspect ratio
	h = input_image.size[0]
	w = input_image.size[1]
	ratio = h *1.0 / w
	if ratio > 1:
		h = opt.load_size
		w = int(h*1.0/ratio)
	else:
		w = opt.load_size
		h = int(w * ratio)
	input_image = input_image.resize((h, w), Image.BICUBIC)
	input_image = np.asarray(input_image)
	# RGB -> BGR
	input_image = input_image[:, :, [2, 1, 0]]
	input_image = transforms.ToTensor()(input_image).unsqueeze(0)
	# preprocess, (-1, 1)
	input_image = -1 + 2 * input_image 
	if opt.gpu > -1:
		input_image = Variable(input_image, volatile=True).cuda()
	else:
		input_image = Variable(input_image, volatile=True).float()
	# forward
	output_image = model(input_image)
	output_image = output_image[0]
	# BGR -> RGB
	output_image = output_image[[2, 1, 0], :, :]
	# deprocess, (0, 1)
	output_image = output_image.data.cpu().float() * 0.5 + 0.5
	# save
	vutils.save_image(output_image, os.path.join(opt.output_dir, files[:-4] + '_' + opt.style + '.jpg'))

print('Done!')
