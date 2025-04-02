import torch
import numpy as np
import os
from common.skeleton import Skeleton
from common.quaternion import *
from paramUtil import *


def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qmul(q, r):
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()


def global_position_2_representation(
    positions, feet_thre, fid_r, fid_l, face_joint_indx, n_raw_offsets, kinematic_chain
):
    global_positions = positions.copy()

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    r_rot = None

    def get_rifke(positions):
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        positions = qrot_np(
            np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions
        )
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=False
        )
        quat_params = qfix(quat_params)
        r_rot = quat_params[:, 0].copy()
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=True
        )
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    root_y = positions[:, 0, 1:2]
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    # print("###root data:  ", root_data.shape, (True in np.isnan(root_data)))
    # print("###ric data:  ",ric_data.shape, (True in np.isnan(ric_data)))
    # print("###rot data:  ",rot_data.shape, (True in np.isnan(rot_data)))
    # print("###local_vel data:  ",local_vel.shape, (True in np.isnan(local_vel)))
    # print("###feet_l data:  ",feet_l.shape, (True in np.isnan(feet_l)))
    # print("###feet_r data:  ",feet_r.shape, (True in np.isnan(feet_r)))
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    # print('data*****', data.shape)

    return data, global_positions, positions, l_velocity
def process_npy_files_263(input_path, output_path):
    import traceback

    feet_thre = 0.002
    fid_r, fid_l = [7, 8], [3, 4]
    face_joint_indx = [5, 1, 12, 9]
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain_15_joints

    os.makedirs(output_path, exist_ok=True)
    file_names = os.listdir(input_path)
    npy_files = [f for f in file_names if f.endswith(".npy")]

    for i, file_name in enumerate(npy_files):
        file_path = os.path.join(input_path, file_name)

        try:
            # 加载文件
            data = np.load(file_path)
            print(f"{i} file name: ### {file_name}, shape: {data.shape}")

            # 跳过特定文件
            if '000990' in file_name or '005836' in file_name:
                print(f"Skipping file: {file_name}")
                continue
            # joints_idx = [0,1,4,7,10,2,5,8,11,16,18,20,17,19,21]  
            # # data=data[0]
            # data = data[:, joints_idx, :]
            # 调用 global_position_2_representation
            data2, _, _, _ = global_position_2_representation(
                data,
                feet_thre,
                fid_r,
                fid_l,
                face_joint_indx,
                n_raw_offsets,
                kinematic_chain,
            )

            # 保存处理后的文件
            processed_file_path = os.path.join(output_path, file_name)
            # processed_file_path = os.path.join(output_path, "amass_" + file_name)

            np.save(processed_file_path, data2)

        except Exception as e:
            # 捕获异常并打印错误信息
            print(f"Error processing file: {file_name}")
            traceback.print_exc()
            continue  # 跳过当前文件，继续处理下一个

    print(f"All valid files processed. Output saved to: {output_path}")



if __name__ == "__main__":
    # input_folder = "/liujinxin/code/text-to-motion/dataset/test/test_50_npy/"
    # output_folder = "/liujinxin/code/text-to-motion/dataset/test/gt_joint_vecs/test_50_npy_youtube"
    #input_folder =  "/storage/liujinxin/code/text-to-motion/dataset/test/lu_samples/predictions"
    # input_folder =  "/storage/liujinxin/code/Hu2/output/benchmark/70b/hu_sft_union15_1-6-23-28-52"
    # output_folder = "/storage/liujinxin/code/text-to-motion/dataset/test/pred_joint_vecs/70b/hu_sft_union15_1-6-23-28-52"
    # name = "checkpoint-92076"
    # # input_folder =  f"/liujinxin/code/text-to-motion/dataset/test/8B-10epochs/{name}"
    # output_folder = f"/liujinxin/code/text-to-motion/dataset/test/pred_joint_vecs_8B_10epochs/{name}"
    
    # input_folder =  "/ssdwork/output/hu_sft_union15_12-18-4-39-17"
    # output_folder = "/liujinxin/code/text-to-motion/dataset/test/pred_joint_vecs/hu_sft_union15_12-18-4-39-17"

    # input_folder =  "/liujinxin/code/Hu/dataset/HumanML3D_sample/data_hu/hu_sft_union15_1-11-12-0-57/checkpoint-15346"
    # output_folder = "/liujinxin/code/text-to-motion/dataset/pred_joint_vecs/1epoch"


    # input_folder =  "/liujinxin/code/Hu/dataset/HumanML3D_sample/data_hu/hu_pretrain_union15_12-17-17-24-6"
    # output_folder = "/liujinxin/code/text-to-motion/dataset/test/pred_joint_vecs/hu_pretrain_union15_12-17-17-24-6"

    # input_folder =  "/liujinxin/code/Hu/dataset/HumanML3D_sample/test_4646/hu_pretrain_union15_12-17-17-24-6"
    # output_folder = "/liujinxin/code/text-to-motion/dataset/test/pred_joint_vecs/test_4646_hu_pretrain_union15_12-17-17-24-6"

    # input_folder =  "/liujinxin/code/Hu/dataset/HumanML3D_sample/data_vis/10"
    # output_folder = "/liujinxin/code/text-to-motion/dataset/test/pred_joint_vecs/pred_vis"


    # input_folder =  "/liujinxin/code/Hu/dataset/HumanML3D_sample/test_4646/hu_sft_union15_1-11-12-0-57/checkpoint-15346"
    # output_folder = "/liujinxin/code/text-to-motion/dataset/test/pred_joint_vecs/test_4646_epoch1"

    input_folder =  "/liujinxin/code/Hu/dataset/AMASS15/train_1000_npy"
    output_folder = "/liujinxin/code/text-to-motion/dataset/test/gt_joint_vecs/train_npy_1000"

    # /liujinxin/anaconda3/envs/T2M-GPT/bin/python data_converter_15joints.py 
    # /liujinxin/anaconda3/envs/RDT/bin/python data_converter_15joints.py
    # os.makedirs(output_folder, exist_ok=True)

    process_npy_files_263(input_folder, output_folder)

    # for dir_name in os.listd
    # for dir_name in os.listdir(input_folder):
    #         if dir_name.endswith('.yaml') or dir_name.endswith('.log'): continue
    #         print('dir_name .....', dir_name)
    #         input_folder_new = os.path.join(input_folder, dir_name)
    #         output_folder_new = os.path.join(output_folder, dir_name)
    #         if not os.path.exists(output_folder_new):
    #             os.system(f'mkdir {output_folder_new}')
    #         process_npy_files_263(input_folder_new, output_folder_new)