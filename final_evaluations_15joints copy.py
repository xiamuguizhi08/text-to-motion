import os
import numpy as np
import random
from collections import OrderedDict
from datetime import datetime
from torch.utils.data._utils.collate import default_collate

from os.path import join as pjoin
from torch.utils import data
from torch.utils.data import DataLoader
from utils.get_opt import get_opt
from utils.metrics import *


from tqdm import tqdm
from networks.modules import *


def collate_fn(batch):
    batch.sort(key=lambda x: x[1], reverse=True)
    return default_collate(batch)

'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, data_path, mean, std):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        for item in os.listdir(os.path.join(opt.data_root, data_path)):
            if item.endswith('.npy'):
                id_list.append(item.split('.')[0])
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.data_root, data_path, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                   }
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

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
        motion, m_length = data['motion'], data['length']
        # Randomly select a caption
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
        return  motion, m_length


def get_dataset_motion_loader(opt_path, data_path, batch_size, device):
    opt = get_opt(opt_path, device)
    opt.data_root = './dataset/'

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m':
        print('Loading dataset %s ...' % opt.dataset_name)
        mean = np.load(pjoin(opt.meta_dir, 'mean_15joints.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std_15joints.npy'))

        dataset = Text2MotionDatasetV2(opt, data_path, mean, std )
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
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return  motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 179 
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.motion_encoder.eval()
        self.movement_encoder.eval()


    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

def evaluate_matching_score(motion_loader, file):
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    all_motion_embeddings = []
    with torch.no_grad():
        print("len motion loader",len(motion_loader))
        for idx, batch in enumerate(motion_loader):
            motions, m_lens  = batch #word_embeddings [3, 22, 300], motions [3, 196, 251]
            print('motions ', idx, motions.shape)
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )  #motion_embeddings [3,512]
            print("motion_embedding", motion_embeddings.shape)

            all_motion_embeddings.append(motion_embeddings.cpu().numpy())

        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        print('all_motion_embeddings ....', all_motion_embeddings.shape)
        activation_dict[motion_loader_name] = all_motion_embeddings

    return activation_dict

def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            motions, m_lens = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict

def evaluate_diversity(activation_dict):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        print('model name ,', model_name, 'motion embedding, ', motion_embeddings.shape)
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
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
        for replication in range(replication_times):
            pred_dirs = os.listdir(os.path.join('./dataset', 'test/pred_test'))
            if len(pred_dirs) == replication_times:
                pred_dir = pred_dirs[replication]
            elif len(pred_dirs) == 1:
                pred_dir = pred_dirs[0]
            if 'pred' not in pred_dir: continue
            print('pred dir   ', pred_dir )
            pred_loader, pred_dataset = get_dataset_motion_loader(dataset_opt_path, f'test/pred_test/{pred_dir}', batch_size, device)

            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            acti_dict = evaluate_matching_score(pred_loader, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)
            div_score_dict = evaluate_diversity(acti_dict )

            

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            
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


if __name__ == '__main__':
    dataset_opt_path = './checkpoints/t2m/opt.txt'
    motion_loader_name = 'youtube'

    device_id = 0
    device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    #mm_num_samples = 100
    mm_num_samples = 3 
    # mm_num_samples = 0

    mm_num_repeats = 30
    mm_num_times = 10

    diversity_times = 100
    replication_times = 20
    batch_size = 2


    # mm_num_samples = 20
    # mm_num_repeats = 1
    #
    # batch_size = 1

    gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, 'test/gt_joint_vecs/humanml3d/new_15_joint_vecs', batch_size, device)
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    log_file = './t2m_evaluation.log'
    evaluation(log_file)
    # animation_4_user_study('./user_study_t2m/')