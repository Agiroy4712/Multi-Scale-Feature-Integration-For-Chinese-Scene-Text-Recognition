#coding=utf-8
#************************************************
# *Author*        :huangzhicai
# *Created Time*  : 2020-08-06 23:26:16
#**Desc**:
#**Analyse**:
#**Get**:
#**Code**:
#************************************************

import os 
import cv2
import sys 

# sys.path.append( os.path.abspath( os.path.dirname(__file__)) )
from ipdb import set_trace

import numpy as np
# import seaborn as sns
from sklearn.metrics import confusion_matrix 
import pandas as pd 
# import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
from sklearn.metrics import classification_report

from Levenshtein import editops

import torch
import torch.nn.functional as F
import numpy as np 

from utils.file_utils import add_start_docstrings, add_end_docstrings, mklogsdir


class MetricUtils(object):
    """MetricUtils 
    
    """
    def __init__(self):

        pass 
    
    @classmethod
    # val_compare=(pred label) pred is torch, label is list
    def getAccuracy(cls, val_compare, tokenizer):
        # pred size
        total = 0
        nsame = 0
        for pred, label, size_of_this_batch in val_compare:
            total += len(label)

            pred_size = torch.IntTensor([pred.size(0)] * size_of_this_batch)
            # cnn -> ctcLen, batch, nclass
            _, predmax = pred.max(dim=2)
            predspro = predmax.transpose(1,0).contiguous().view(-1)
            pred_text = tokenizer.decode(predspro.data, pred_size.data)

            pred_text_np  = np.array(pred_text)
            label_text_np = np.array(label )

            nsame += np.sum(pred_text_np==label_text_np)
        acc = nsame / total
        # print("total is {}, accuracy is {}\n".format(total, acc))
        # return acc
        return nsame, total, acc

    @classmethod
    # val_compare=(pred label) pred is torch, label is list
    def getAccuracyBySent(cls, val_compare, tokenizer):
        # pred size
        total = 0
        nsame = 0

        # size_of_this_batch is useless
        for pred_text, label, size_of_this_batch in val_compare:
            total += len(label)

            pred_text_np  = np.array(pred_text)
            label_text_np = np.array(label )

            nsame += np.sum(pred_text_np==label_text_np)
        acc = nsame / total
        # print("total is {}, accuracy is {}\n".format(total, acc))
        # return acc
        return nsame, total, acc

    
    @classmethod
    # pred is model out
    def getAccOfDocu(cls, preds, targets):
        totalNum = targets.shape[0]
        pred_val, pred_idx = preds.max(dim=1)
        tn = (pred_idx==targets).sum()
        acc = tn.item()/float(totalNum)
        # print("tn is {}, acc is: {:.4f}".format(tn, acc) )
        return acc 

    @classmethod 
    def getEditDistance(cls, val_compare, tokenizer):
        pred, label, size_of_this_batch = val_compare[0]

        mean_distance, length = 0, len(label)
        pred_size = torch.IntTensor([pred.size(0)] * size_of_this_batch)
        # cnn -> ctcLen, batch, nclass
        _, predmax = pred.max(dim=2)
        predspro = predmax.transpose(1,0).contiguous().view(-1)
        pred_text = tokenizer.decode(predspro.data, pred_size.data)

        for y0, y in zip(pred_text, label):
            mean_distance += levenshtein(y0, y) / length
        return mean_distance

    @classmethod 
    def getEditDistanceBySent(cls, val_compare, tokenizer):
        pred_text, label, size_of_this_batch = val_compare[0]

        mean_distance, length = 0, len(label)

        for y0, y in zip(pred_text, label):
            mean_distance += levenshtein(y0, y) / length
        return mean_distance


    @classmethod 
    def getNormEditDistance(cls, val_compare, tokenizer):
        pred, label, size_of_this_batch = val_compare[0]

        mean_distance, length = 0, len(label)
        pred_size = torch.IntTensor([pred.size(0)] * size_of_this_batch)
        # cnn -> ctcLen, batch, nclass
        _, predmax = pred.max(dim=2)
        predspro = predmax.transpose(1,0).contiguous().view(-1)
        pred_text = tokenizer.decode(predspro.data, pred_size.data)

        for y0, y in zip(pred_text, label):
            mean_distance += levenshtein(y0, y) / (len(y) * length)
        return mean_distance

    @classmethod 
    def getNormEditDistanceBySent(cls, val_compare, tokenizer):
        pred_text, label, size_of_this_batch = val_compare[0]

        mean_distance, length = 0, len(label)

        for y0, y in zip(pred_text, label):
            mean_distance += levenshtein(y0, y) / (len(y) * length)
        return mean_distance


    @classmethod 
    def getNormEditDistanceOfICDAR19(cls, val_compare, tokenizer):
        '''
        (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
        if len(gt) == 0:
            norm_ED += 1
        else:
            norm_ED += edit_distance(pred, gt) / len(gt)
        '''

        pred, label, size_of_this_batch = val_compare[0]

        pred_size = torch.IntTensor([pred.size(0)] * size_of_this_batch)
        # cnn -> ctcLen, batch, nclass
        _, predmax = pred.max(dim=2)
        predspro = predmax.transpose(1,0).contiguous().view(-1)
        pred_text = tokenizer.decode(predspro.data, pred_size.data)
        gt   = label
        pred = pred_text
        # ICDAR2019 Normalized Edit Distance
        if len(gt) == 0 or len(pred) == 0:
            norm_ED += 0
        elif len(gt) > len(pred):
            norm_ED += 1 - edit_distance(pred, gt) / len(gt)
        else:
            norm_ED += 1 - edit_distance(pred, gt) / len(pred)

        return norm_ED

    @classmethod 
    def getCRAR(cls, predtext, labeltext, verse=False):
        ar_list=list()
        cr_list=list()

        # process empty string
        if predtext=='':
            predtext=['']

        mul=1
        for step_i in range(len(labeltext)):
            label = labeltext[step_i]
            pred  = predtext[step_i]
            nTruthNum=len(label)
            temps = editops(label, pred)
            insert_num = 0
            delete_num = 0
            replace_num = 0
            insertionError=0
            deletionError=0
            substituteError=0
            for temp in temps:
                if temp[0] == 'insert':
                    insert_num += 1 * mul
                    insertionError += 1 * mul
                elif temp[0] == 'delete':
                    delete_num += 1 * mul
                    deletionError += 1 * mul
                elif temp[0] == 'replace':
                    replace_num += 1 * mul
                    substituteError += 1 * mul
            if verse:
                temp_CR = (nTruthNum - delete_num - replace_num) / nTruthNum
                temp_AR = (nTruthNum - delete_num - replace_num - insert_num) / nTruthNum
                print("CR = " + str(temp_CR), "; AR = " + str(temp_AR) + '-----')
                print('-'*30)
                print("真实文本数 N = ", nTruthNum)
                print("删除错误 D = ", deletionError)
                print("替换错误 S = ", substituteError)
                print("插入错误 I = ", insertionError)
            # NOTE
            if nTruthNum:
                CorrectRate = (nTruthNum - deletionError - substituteError) / nTruthNum
                AccuracyRate = (nTruthNum - deletionError - substituteError - insertionError) / nTruthNum
                cr_list.append(CorrectRate)
                ar_list.append(AccuracyRate)
            else:
                cr_list.append(0.0)
                ar_list.append(0.0)
        
            if verse:
                print("CR = " + str(CorrectRate))
                print("AR = " + str(AccuracyRate))
        return np.array(cr_list).mean(), np.array(ar_list).mean()

    @classmethod 
    def getCRAROverall(cls, predtext, labeltext, verse=False):
        ar_list=list()
        cr_list=list()

        nTruthNumList       = list()
        insertionErrorList  = list()
        deletionErrorList   = list()
        substituteErrorList = list()

        mul=1
        for step_i in range(len(labeltext)):
            label = labeltext[step_i]
            pred  = predtext[step_i]
            nTruthNum=len(label)
            temps = editops(label, pred)
            insert_num = 0
            delete_num = 0
            replace_num = 0
            insertionError=0
            deletionError=0
            substituteError=0
            for temp in temps:
                if temp[0] == 'insert':
                    insert_num += 1 * mul
                    insertionError += 1 * mul
                elif temp[0] == 'delete':
                    delete_num += 1 * mul
                    deletionError += 1 * mul
                elif temp[0] == 'replace':
                    replace_num += 1 * mul
                    substituteError += 1 * mul
            if verse:
                temp_CR = (nTruthNum - delete_num - replace_num) / nTruthNum
                temp_AR = (nTruthNum - delete_num - replace_num - insert_num) / nTruthNum
                print("CR = " + str(temp_CR), "; AR = " + str(temp_AR) + '-----')
                print('-'*30)
                print("真实文本数 N = ", nTruthNum)
                print("删除错误 D = ", deletionError)
                print("替换错误 S = ", substituteError)
                print("插入错误 I = ", insertionError)
            # NOTE
            if nTruthNum:
                CorrectRate = (nTruthNum - deletionError - substituteError) / nTruthNum
                AccuracyRate = (nTruthNum - deletionError - substituteError - insertionError) / nTruthNum
                cr_list.append(CorrectRate)
                ar_list.append(AccuracyRate)
            else:
                cr_list.append(0.0)
                ar_list.append(0.0)


            nTruthNumList.append(nTruthNum)
            deletionErrorList.append(deletionError)
            substituteErrorList.append(substituteError)
            insertionErrorList.append(insertionError)

            if verse:
                print("CR = " + str(CorrectRate))
                print("AR = " + str(AccuracyRate))
        nTruthNumTotal = sum(nTruthNumList)
        tl = sum(nTruthNumList)
        dl = sum(deletionErrorList)
        sl = sum(substituteErrorList)
        il = sum(insertionErrorList)
        # CorrectRate = (tl - dl - sl ) / tl
        # AccuracyRate = (tl - dl - sl - il) / tl
        # return CorrectRate, AccuracyRate
        return tl,dl,sl,il


def edit_distance(y_pred, y_true):
    mean_distance, length = 0, len(y_true)
    for y0, y in zip(y_pred, y_true):
        mean_distance += levenshtein(y0, y) / length
    return mean_distance

def normalized_edit_distance(y_pred, y_true):
    mean_distance, length = 0, len(y_true)
    for y0, y in zip(y_pred, y_true):
        mean_distance += levenshtein(y0, y) / (len(y) * length)
    return mean_distance

# edit diatance
def levenshtein(seq1, seq2):  
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1

    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                   )
            else:
                matrix [x,y] = min(
                        matrix[x-1,y] + 1,
                        matrix[x-1,y-1] + 1,
                        matrix[x,y-1] + 1
                    )
                                                                                                                
    return (matrix[size_x - 1, size_y - 1])



"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""
# TODO
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
                         
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
                                                              
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc
                                                                                                                      
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
                                                        
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]    
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


class recogGlobalARCR():
    def __init__(self, args, tokenizer, dataloader, inputParams):
        self.args=args 
        self.inputParams=inputParams
        self.tokenizer = tokenizer
        
        #########################
        # init params
        #########################
        self.n_count=0
        self.nsameList = list()
        self.totalList = list()
        self.batch_cr_list = []
        self.batch_ar_list = []
        self.med      = 0
        self.med_norm = 0

        self.tloverList = list()
        self.dloverList = list()
        self.sloverList = list()
        self.iloverList = list()

        self.dataloader_len = len(dataloader)


    def update(self, labels, pred_list, printable=False):
        batch_size = len(labels)
        val_compare = [(pred_list, labels, batch_size)]

        mean_distance = MetricUtils.getEditDistanceBySent(val_compare, self.tokenizer)
        mean_distance_norm = MetricUtils.getNormEditDistanceBySent(val_compare, self.tokenizer)

        self.med      += mean_distance
        self.med_norm += mean_distance_norm

        nsame, total, acc = MetricUtils.getAccuracyBySent(val_compare, self.tokenizer)
        self.nsameList.append(nsame)
        self.totalList.append(total)
        cr, ar = MetricUtils.getCRAR(pred_list, labels)

        tl,dl,sl,il = MetricUtils.getCRAROverall(pred_list, labels)
        self.tloverList.append(tl)
        self.dloverList.append(dl)
        self.sloverList.append(sl)
        self.iloverList.append(il)
        self.batch_cr_overall = (tl - dl - sl ) / tl
        self.batch_ar_overall = (tl - dl - sl - il) / tl
        self.batch_cr_list.append(cr)
        self.batch_ar_list.append(ar)

        if printable:
            print("acc is {:.4f}, ar is {:.4f}, cr is {:.4f}, arover {:.4f}, crover {:.4f}".format(acc, ar, cr, \
                    self.batch_ar_overall, self.batch_cr_overall))

    # write test log to resultfile
    def writeLog(self, resultfile, num, preds, data, pred_text):
        args = self.args
        image_paths, imagesTensor, labels = data
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for cn,( fn, lt, pt, pred_max_prob) in enumerate( zip(image_paths, labels, pred_text, preds_max_prob) ):
            # fn = fn.name
            fn = str(fn)
            cn = args.batch_size*num+cn

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            resultfile.write("filena: \t{}\t{}\n".format(cn, fn) )
            resultfile.write("confsc: \t{:.4f}\n".format(confidence_score) )
            resultfile.write("ground: \t{}\n".format(lt) )
            resultfile.write("predtx: \t{}\n\n".format(pt) )

    def writeErrorLog(self, resultfile, num, preds, data, pred_text):
        args = self.args
        image_paths, imagesTensor, labels = data
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for cn,( fn, lt, pt, pred_max_prob) in enumerate( zip(image_paths, labels, pred_text, preds_max_prob) ):
            # fn = fn.name
            fn = str(fn)
            cn = args.batch_size*num+cn

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            if lt!=pt:
                resultfile.write("filena: \t{}\t{}\n".format(cn, fn) )
                resultfile.write("confsc: \t{:.4f}\n".format(confidence_score) )
                resultfile.write("ground: \t{}\n".format(lt) )
                resultfile.write("predtx: \t{}\n\n".format(pt) )

    def writeLowScore(self, resultfile, num, preds, data, pred_text):
        args = self.args
        image_paths, imagesTensor, labels = data
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for cn,( fn, lt, pt, pred_max_prob) in enumerate( zip(image_paths, labels, pred_text, preds_max_prob) ):
            # fn = fn.name
            fn = str(fn)
            cn = args.batch_size*num+cn

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            if confidence_score<0.1:
                resultfile.write("filena: \t{}\t{}\n".format(cn, fn) )
                resultfile.write("confsc: \t{:.4f}\n".format(confidence_score) )
                resultfile.write("ground: \t{}\n".format(lt) )
                resultfile.write("predtx: \t{}\n\n".format(pt) )

    # get final result
    def getResult(self, resultfile, testDataName, printable=False, output=True, use_LM=False):
        args = self.args
        nsamesum = np.array(self.nsameList).sum()
        totalsum = np.array(self.totalList).sum()
        # global sum
        acc = nsamesum/totalsum
        final_ar = np.array(self.batch_ar_list).mean()
        final_cr = np.array(self.batch_cr_list).mean()
        
        tl = sum(self.tloverList)
        dl = sum(self.dloverList)
        sl = sum(self.sloverList)
        il = sum(self.iloverList)
        final_cr_overall = (tl - dl - sl ) / tl
        final_ar_overall = (tl - dl - sl - il) / tl

        med      = self.med/self.dataloader_len
        med_norm = self.med_norm/self.dataloader_len

        if output:
            if use_LM:
                resultfile.writelines(f"testDataName: {testDataName} LM\n")
            else:
                resultfile.writelines(f"testDataName: {testDataName}\n")

            resultfile.writelines("acc is {:.4f}, same is {}, total is {}, final_ar is {:.4f}, final_cr is {:.4f}, ar_over {:.4f}, cr_global {:.4f}\n".format(
                acc, nsamesum, totalsum, final_ar, final_cr, final_ar_overall, final_cr_overall))
            resultfile.writelines("mean_distance is {}, mean_distance_norm is {}\n".format(med, med_norm))

            # test result
            testResFile = os.path.join(args.project_dir, args.logdir, "testResFile.txt")
            with open(testResFile, "a") as trf:
                trf.writelines(f"{testDataName}    {args.use_unified_time}\n")
                trf.writelines(f"{args.archive_name} {args.desc} {args.pretrained_model_suffix} test result is:\n")
                trf.writelines("acc is {:.4f}, same is {}, total is {}, final_ar is {:.4f}, final_cr is {:.4f}, ar_over {:.4f}, cr_over {:.4f}\n\n".format(
                    acc, nsamesum, totalsum, final_ar, final_cr, final_ar_overall, final_cr_overall))

        if printable:
            print("{} {} {}".format(testDataName, args.archive_name, args.desc))
            print("acc {:.4f}, same is {}, total is {}, final_ar is {:.4f}, final_cr is {:.4f}, ar_over {:.4f}, cr_over {:.4f}\n".format(
                acc, nsamesum, totalsum, final_ar, final_cr, final_ar_overall, final_cr_overall))
            print("mean_distance is {}, mean_distance_norm is {}\n".format(med, med_norm))
        return acc


class Acc_Average():
    def __init__(self, config):
        #########################
        # init params
        #########################
        self.n_count=0
        self.accList = list()
        self.predTargetArray={}
        self.y_pred = []
        self.y_true = []
        self.labelDict = config.labelDict
        # storage error info
        self.infoDict = {self.labelDict[k]:{"count":0, "same":0} for k in self.labelDict.keys()}

    def initPredTargetArray(self):
        labelDictLen = len(self.labelDict)
        self.predTargetArray=np.zeros( (labelDictLen, labelDictLen ) , np.uint8)

    def saveHeatMapOfSeaborn(self, savePath="confusion.pdf"):
        labelDictLen = len(self.labelDict)
        keys = [v for v in self.labelDict.values()]
        idxs = [str(i) for i in range(labelDictLen)]

        conf_matrix = pd.DataFrame(self.predTargetArray, index=idxs, columns=idxs)
        # TODO
        # fig, ax = plt.subplots(figsize = (100,100))

        # TODO sns occur signal 11 error
        # sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 29}, cmap="Blues")

        # plt.ylabel('True label', fontsize=48)
        # plt.xlabel('Predicted label', fontsize=48)
        # TODO
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.savefig(savePath, bbox_inches='tight')

    def calculateF1(self, resultfile):
        labelDictLen = len(self.labelDict)
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        outList = []

        outList.append("------Weighted------")
        wp = precision_score(y_true, y_pred, average='weighted')
        wr = recall_score(y_true, y_pred, average='weighted')
        wf = f1_score(y_true, y_pred, average='weighted')
        outList.append(f'Weighted precision: {wp:.6}' )
        outList.append(f'Weighted recall: {wr:.6f}' )
        outList.append(f'Weighted f1-score: {wf:.6f}' )

        outList.append( '------Macro------')
        macp = precision_score(y_true, y_pred, average='macro')
        macr = recall_score(y_true, y_pred, average='macro')
        macf = f1_score(y_true, y_pred, average='macro')
        outList.append(f'Macro precision: {macp:.6f}' )
        outList.append(f'Macro recall: {macr:.6f}' )
        outList.append(f'Macro f1-score: {macf:.6f}' )

        outList.append('------Micro------')
        micp = precision_score(y_true, y_pred, average='micro')
        micr = recall_score(y_true, y_pred, average='micro')
        micf = f1_score(y_true, y_pred, average='micro')
        outList.append(f'Micro precision: {micp:.6f}' )
        outList.append(f'Micro recall: {micr:.6f}' )
        outList.append(f'Micro f1-score: {micf:.6f}' )

        resultfile.write('\n'.join(outList))
        print("\n".join(outList))

        # support is total of y_true
        report = classification_report(y_true, y_pred)
        resultfile.write("\n"+report)
        print(report)

    def saveHeatMapOfPredTarget(self, savePath="foo.png"):
        margin = 10
        labelDictLen = len(self.labelDict)
        newHeatMap = np.ones((labelDictLen*margin, labelDictLen*margin), np.uint8)
        newHeatMap.fill(255)

        # scale to large img
        for ti in range(labelDictLen):
            for pj in range(labelDictLen):
                val = self.predTargetArray[ti][pj]
                if val>0:
                    newHeatMap[ti*margin : (ti+1)*margin][:, pj*margin: (pj+1)*margin]=val 
        newheatmap = cv2.applyColorMap(newHeatMap, cv2.COLORMAP_HOT)
        # newheatmap = cv2.applyColorMap(newHeatMap, cv2.COLORMAP_JET)
        # newheatmap = cv2.applyColorMap(newHeatMap, cv2.COLORMAP_BONE)
        cv2.imwrite(savePath, newheatmap)

    def add(self, val):
        self.accList.append(val)
        self.n_count += 1

    def getAvgAcc(self):
        return np.array(self.accList).mean()

    def clear(self):
        self.accList = []
        self.n_count = 0

    def writeLog(self, resultfile):
        resultfile.write("acc: \t{:.6f}\n".format(self.getAvgAcc()) )
        for k,v in self.infoDict.items():
            resultfile.write(f"{k}:\t{str(v)}\n")


