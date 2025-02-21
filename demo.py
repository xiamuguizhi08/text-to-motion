import os
import codecs as cs
import numpy as np
import random
from collections import OrderedDict
from datetime import datetime
from torch.utils.data._utils.collate import default_collate
import os
import torch
from os.path import join as pjoin
from torch.utils import data
from torch.utils.data import DataLoader
from utils.get_opt import get_opt
from utils.metrics import *
from utils.word_vectorizer import WordVectorizer
from utils.word_vectorizer import POS_enumerator

from tqdm import tqdm
from networks.modules import *
import sys
sys.path.append("/liujinxin/code/Hu/dataset/HumanML3D_sample/")  
from data_converter_15joints import process_npy_files_263  



def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, data_path, mean, std, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        # self.max_length = 20
        self.max_length=20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        # min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        data_dict = {}
        id_list = []
        for item in os.listdir(os.path.join(opt.data_root, data_path)):
            if item.endswith('.npy'):
                id_list.append(item.split('.')[0])
        new_name_list = []
        length_list = []
        print('====id_list',len(id_list))
        for name in tqdm(id_list):
            #try:
            motion = np.load(pjoin(opt.data_root, data_path, name + '.npy'))
            if (len(motion)) < min_motion_len or (len(motion) >= 200):
                continue
            
            # if len(motion)<min_motion_len:
            #     pad_length = min_motion_len - len(motion)
            #     padding = np.zeros((pad_length, motion.shape[1]))  # 形状：(pad_length, 179)
            #     motion = np.concatenate([motion, padding], axis=0)

            # elif len(motion) > 196:
            #     motion = motion[:196] 
            text_data = []
            flag = False
            with cs.open(pjoin(opt.text_dir,  name+ '.txt')) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                       

                    else:
                        try:
                            n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                            if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                continue
                            # if len(n_motion) >= 197:
                            #     n_motion = motion[:196] 
                            # elif len(n_motion)<min_motion_len:
                            #     pad_length = min_motion_len - len(motion)
                                padding = np.zeros((pad_length, motion.shape[1]))  # 形状：(pad_length, 179)
                                n_motion = np.concatenate([motion, padding], axis=0)
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            data_dict[new_name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict]}
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))
                        except Exception as e:
                            # print("发生异常:", e)
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag, to_tag, name)
                            # break
            if flag:
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data}
                new_name_list.append(name)
                length_list.append(len(motion))

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        print('===name_list',len(name_list))
        print('===length_list',len(length_list))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]  #[T, 251]

        "Z Normalization"
        # motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)



def get_dataset_motion_loader(opt_path, data_path, batch_size, device):
    opt = get_opt(opt_path, device)
    opt.data_root = './dataset/'
    opt.text_dir ='/liujinxin/code/text-to-motion/dataset/test/test_txt_1000'

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m':
        print('Loading dataset %s ...' % opt.dataset_name)
        mean = np.load(pjoin(opt.meta_dir, 'mean_15joints.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std_15joints.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        dataset = Text2MotionDatasetV2(opt, data_path, mean, std, w_vectorizer )
        print('0000000len(dataset)',len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)


    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset 


def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match_179', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 179 
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
   
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach() #[3, 49, 512]
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens) #[3, 512]

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            # torch.Size([30, 49, 512])     torch.Size([30, 196, 175])
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            # torch.Size([30, 512])   torch.Size([30, 49, 512])
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding


def evaluate_matching_score(motion_loader):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    all_motion_embeddings = []
    score_list = []
    all_size = 0
    matching_score_sum = 0
    top_k_count = 0
    # print(motion_loader_name)
    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
            word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch #word_embeddings [3, 22, 300], motions [3, 196, 251]
            text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=motions,
                m_lens=m_lens
            )  #motion_embeddings [3,512]
            dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                    motion_embeddings.cpu().numpy())
            matching_score_sum += dist_mat.trace()
            argsmax = np.argsort(dist_mat, axis=1)
            top_k_mat = calculate_top_k(argsmax, top_k=3)
            top_k_count += top_k_mat.sum(axis=0)

            all_size += text_embeddings.shape[0]

            all_motion_embeddings.append(motion_embeddings.cpu().numpy())

        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        matching_score = matching_score_sum / all_size
        R_precision = top_k_count / all_size
        match_score_dict[motion_loader_name] = matching_score
        R_precision_dict[motion_loader_name] = R_precision
        activation_dict[motion_loader_name] = all_motion_embeddings

    print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')

    line = f'---> [{motion_loader_name}] R_precision: '
    for i in range(len(R_precision)):
        line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)

    return match_score_dict, R_precision_dict, activation_dict

def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)
    print('#############', gt_motion_embeddings.shape)
    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict



def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        print('##########motion embeddings ', motion_embeddings.shape)
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loader, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    mm_motion_embeddings = []
    with torch.no_grad():
        for idx, batch in enumerate(mm_motion_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(motions, m_lens)
            mm_motion_embeddings.append(motion_embeddings.unsqueeze(0))
    if len(mm_motion_embeddings) == 0:
        multimodality = 0
    else:
        mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy() #358,3,512
        multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
    print(f'---> [{motion_loader_name}] Multimodality: {multimodality:.4f}')
    print(f'---> [{motion_loader_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
    eval_dict[motion_loader_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})

        pred_dirs = os.listdir(os.path.join('./dataset', pred_root))
        replication_times = len(pred_dirs)
        for replication in range(replication_times):
            pred_dir = pred_dirs[replication]
            print('========',f'{pred_root}/{pred_dir}')
            pred_loader, mm_motion_loader= get_dataset_motion_loader(dataset_opt_path, f'{pred_root}/{pred_dir}', batch_size, device)

            print(f'==================== Replication {replication} ====================')
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(pred_loader)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(pred_loader, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)


import os
import torch



def process_files(input_folder, output_folder):
    """处理 .npy 文件"""
    for dir_name in filter(lambda d: not d.endswith(('.yaml', '.log')), os.listdir(input_folder)):
        print('Processing:', dir_name)
        input_folder_new = os.path.join(input_folder, dir_name)
        output_folder_new = os.path.join(output_folder, dir_name)
        os.makedirs(output_folder_new, exist_ok=True)
        
   
        process_npy_files_263(input_folder_new, output_folder_new)






 
if __name__ == '__main__':
    
    pred_name = "hu_finetune_EXP_amass_not_abs_2-19-13-53-49"
    dataset_opt_path = './checkpoints/t2m/opt.txt'
    motion_loader_name = f'{pred_name}'

    device_id = 0
    device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)


    mm_num_samples = 30 
    mm_num_repeats = 30
    mm_num_times = 10

    diversity_times = 200
    replication_times = 5
    batch_size = 30

    input_folder = "/liujinxin/code/Hu/dataset/HumanML3D_sample/amass_15_1000/hu_finetune_EXP_amass_not_abs_2-19-13-53-49/checkpoint-30000"
    output_folder = "/liujinxin/code/text-to-motion/dataset/hu_finetune_EXP_amass_not_abs_2-19-13-53-49/checkpoint-30000"

    # process_files(input_folder, output_folder)

    pred_root=f'./hu_finetune_EXP_amass_not_abs_2-19-13-53-49/checkpoint-30000'
    gt_loader, gt_dataset= get_dataset_motion_loader(dataset_opt_path, 'test/gt_joint_vecs/test_npy_1000', batch_size, device)
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    log_file = f'./log/{pred_root[2:]}.log'

    log_dir = os.path.dirname(log_file)

   
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    evaluation(log_file)
    print(pred_root)
 