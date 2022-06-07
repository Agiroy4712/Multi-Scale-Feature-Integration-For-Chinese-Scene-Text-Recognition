from __future__ import absolute_import

import sys

sys.path.append('./')

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.backends import cudnn
import torchvision
from torchvision import transforms

from config import get_args
from lib.models.model_builder import ModelBuilder
from lib.utils.serialization import load_checkpoint
from lib.evaluation_metrics.metrics import get_str_list
from lib.utils.labelmaps import get_vocabulary

global_args = get_args(sys.argv[1:])


def image_process(image_path, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    img = Image.open(image_path).convert('RGB')

    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img


class DataInfo(object):
    """
  Save the info about the dataset.
  This a code snippet from dataset.py
  """

    def __init__(self, voc_type):
        super(DataInfo, self).__init__()
        self.voc_type = voc_type

        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS', 'CHINESE_7715']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print('using cuda.')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (32, 100)

    dataset_info = DataInfo(args.voc_type)

    # Create model
    model = ModelBuilder(arch=args.arch, rec_num_classes=dataset_info.rec_num_classes,
                         sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=args.max_len,
                         eos=dataset_info.char2id[dataset_info.EOS], STN_ON=args.STN_ON)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    if args.cuda:
        device = torch.device("cuda")
        model = model.to(device)
        model = nn.DataParallel(model)

    # Evaluation
    model.eval()
    img = image_process(args.image_path)
    with torch.no_grad():
        img = img.to(device)
    input_dict = {}
    input_dict['images'] = img.unsqueeze(0)
    # TODO: testing should be more clean.
    # to be compatible with the lmdb-based testing, need to construct some meaningless variables.
    rec_targets = torch.IntTensor(1, args.max_len).fill_(1)
    rec_targets[:, args.max_len - 1] = dataset_info.char2id[dataset_info.EOS]
    input_dict['rec_targets'] = rec_targets
    input_dict['rec_lengths'] = [args.max_len]
    output_dict = model(input_dict)
    pred_rec = output_dict['output']['pred_rec']
    pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info)
    torchvision.utils.save_image(output_dict['output']['rectified_images'],
                                 "/home/liaohaiqing/Chinese_Scene_Text_Rec_multi_se/data/rec.png")
    print('Recognition result: {0}'.format(pred_str[0]))


if __name__ == '__main__':
    # parse the config
    args = get_args(sys.argv[1:])
    main(args)
