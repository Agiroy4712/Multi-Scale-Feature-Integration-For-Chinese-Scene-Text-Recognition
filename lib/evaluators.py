from __future__ import print_function, absolute_import
import time
from time import gmtime, strftime
from datetime import datetime
from collections import OrderedDict

import torch
import json
import numpy as np
from random import randint
from PIL import Image
import sys
from tqdm import tqdm
# for test part
from lib.evaluation_metrics.metrics import get_str_list

from . import evaluation_metrics
from .evaluation_metrics import Accuracy, EditDistance, RecPostProcess
from .utils.meters import AverageMeter
from .utils.visualization_utils import recognition_vis, stn_vis

metrics_factory = evaluation_metrics.factory()

from config import get_args
# for test part
from .utils.labelmaps import get_vocabulary

# for edit_distance
from nltk.metrics.distance import edit_distance

global_args = get_args(sys.argv[1:])


class DataInfo(object):
    """
Save the info about the dataset.
This a code snippet from dataset.py
"""

    def __init__(self, voc_type):
        super(DataInfo, self).__init__()
        self.voc_type = voc_type

        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS', 'CHINESE', 'CHINESE_syn', 'CHINESE_7715']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)


class BaseEvaluator(object):
    def __init__(self, model, metric, use_cuda=True):
        super(BaseEvaluator, self).__init__()
        self.model = model
        self.metric = metric
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def evaluate(self, data_loader, step=1, print_freq=1, tfLogger=None, dataset=None, vis_dir=None):
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        # forward the network
        images, outputs, targets, losses = [], {}, [], []
        images2 = 0
        file_names = []

        end = time.time()

        for i, inputs in enumerate(data_loader):
            # 验证集太大设置
            # if i >= 2:
            # break
            data_time.update(time.time() - end)

            input_dict = self._parse_data(inputs)
            output_dict = self._forward(input_dict)

            batch_size = input_dict['images'].size(0)

            total_loss_batch = 0.
            for k, loss in output_dict['losses'].items():
                loss = loss.mean(dim=0, keepdim=True)
                total_loss_batch += loss.item() * batch_size

            temp = len(input_dict['images'])
            images2 += temp
            # images.append(input_dict['images'])

            targets.append(input_dict['rec_targets'])
            losses.append(total_loss_batch)
            if global_args.evaluate_with_lexicon:
                file_names += input_dict['file_name']
            for k, v in output_dict['output'].items():
                if k not in outputs:
                    outputs[k] = []
                outputs[k].append(v.cpu())

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('[{}]\t'
                      'Evaluation: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      # .format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                      .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))
            torch.cuda.empty_cache()
        # if not global_args.keep_ratio:
        # images = torch.cat(images)
        # num_samples = images.size(0)
        # else:
        # num_samples = sum([subimages.size(0) for subimages in images])

        num_samples = images2
        targets = torch.cat(targets)
        losses = np.sum(losses) / (1.0 * num_samples)
        for k, v in outputs.items():
            outputs[k] = torch.cat(outputs[k])

        # save info for recognition
        if 'pred_rec' in outputs:
            # evaluation with metric
            if global_args.evaluate_with_lexicon:
                eval_res = metrics_factory[self.metric + '_with_lexicon'](outputs['pred_rec'], targets, dataset,
                                                                          file_names)
                print('lexicon0: {0}, {1:.3f}'.format(self.metric, eval_res[0]))
                print('lexicon50: {0}, {1:.3f}'.format(self.metric, eval_res[1]))
                print('lexicon1k: {0}, {1:.3f}'.format(self.metric, eval_res[2]))
                print('lexiconfull: {0}, {1:.3f}'.format(self.metric, eval_res[3]))
                eval_res = eval_res[0]
            else:
                eval_res = metrics_factory[self.metric](outputs['pred_rec'], targets, dataset)
                print('lexicon0: {0}: {1:.3f}'.format(self.metric, eval_res))
            pred_list, targ_list, score_list = RecPostProcess(outputs['pred_rec'], targets, outputs['pred_rec_score'],
                                                              dataset)

            # =====save the predict str====

            self._saveresult(pred_list, targ_list)

            if tfLogger is not None:
                # (1) Log the scalar values
                info = {
                    'loss': losses,
                    self.metric: eval_res,
                }
                for tag, value in info.items():
                    tfLogger.scalar_summary(tag, value, step)

            self._calc_norm_ED(pred_list, targ_list, num_samples)

        # ====== Visualization ======#
        # if vis_dir is not None:
        # recognition_vis(images, outputs['pred_rec'], targets, score_list, dataset, vis_dir)
        # stn_vis(images, outputs['rectified_images'], outputs['ctrl_points'], outputs['pred_rec'],
        # targets, score_list, outputs['pred_score'] if 'pred_score' in outputs else None, dataset, vis_dir)
        return eval_res

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs):
        raise NotImplementedError

    def _saveresult(self, pred_list, targ_list):
        raise NotImplementedError

    def _calc_norm_ED(self, pred_list, targ_list, num_samples):
        raise NotImplementedError


class Evaluator(BaseEvaluator):
    def _parse_data(self, inputs):
        input_dict = {}
        if global_args.evaluate_with_lexicon:
            imgs, label_encs, lengths, file_name = inputs
        else:
            imgs, label_encs, lengths = inputs

        with torch.no_grad():
            images = imgs.to(self.device)
            if label_encs is not None:
                labels = label_encs.to(self.device)

        input_dict['images'] = images
        input_dict['rec_targets'] = labels
        input_dict['rec_lengths'] = lengths
        if global_args.evaluate_with_lexicon:
            input_dict['file_name'] = file_name
        return input_dict

    def _forward(self, input_dict):
        self.model.eval()
        with torch.no_grad():
            output_dict = self.model(input_dict)
        return output_dict

    def _saveresult(self, pred_list, targ_list):
        # pred_rec = output['pred_rec']
        # rec_targets = torch.IntTensor(1, global_args.max_len).fill_(1)
        # dataset_info = DataInfo(global_args.voc_type)
        # rec_targets[:, global_args.max_len - 1] = dataset_info.char2id[dataset_info.EOS]
        # pred_str, _ = get_str_list(pred_rec, rec_targets, dataset=dataset_info)
        dir = global_args.test_save
        # for rects

        for index, pre in enumerate(pred_list):
            # if pre == "":
            #     pred_list[index] = ""
            pred_list[index] += "\n"

        with open(dir, 'w', encoding='utf-8') as f:
            f.writelines(pred_list)

        for index, pre in enumerate(targ_list):
            # if pre == "":
            #     pred_list[index] = ""
            targ_list[index] += "\n"

        with open("targ_list.txt", 'w', encoding='utf-8') as f:
            f.writelines(targ_list)

        # for art
        # result_dict = {}
        # for i in range(len(pred_list)):
        # key = 'res_' + str(i+1+35086)
        # value = [{"transcription": ''}]
        # value[0]["transcription"] = pred_list[i]
        # result_dict[key] = value
        # with open(dir, 'w', encoding='utf-8') as f:
        # json.dump(result_dict, f)

    def _calc_norm_ED(self, pred_list, targ_list, length_of_data):
        num = len(pred_list)
        norm_ED = 0
        print("计算编辑距离：")
        for i in tqdm(range(num)):
            if len(pred_list[i]) == 0 or len(targ_list[i]) == 0:
                norm_ED += 0
            elif len(targ_list[i]) > len(pred_list[i]):
                norm_ED += 1 - edit_distance(pred_list[i], targ_list[i]) / len(targ_list[i])
            else:
                norm_ED += 1 - edit_distance(pred_list[i], targ_list[i]) / len(pred_list[i])

        norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance
        print(norm_ED)
