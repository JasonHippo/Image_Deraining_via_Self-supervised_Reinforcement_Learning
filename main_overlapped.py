import argparse
import chainer
import numpy as np
import os
from tqdm import tqdm
from brisque import BRISQUE
from mini_batch_loader import MiniBatchLoader
import State
from MyFCN import *
from pixelwise_a3c import *
from skimage.util import img_as_float
import cv2

def overlapped_process(raw_x, mask, agent, current_state, patch_size, stride):
    _, _, h, w = raw_x.shape
    output = np.zeros_like(raw_x)
    counter = np.zeros_like(raw_x)
    for x in range(0, h, stride):
        for y in range(0, w, stride):
            x_end = min(x + patch_size, h)
            y_end = min(y + patch_size, w)
            patch_image = raw_x[:, :, x:x_end, y:y_end]
            patch_mask = mask[:, :, x:x_end, y:y_end]
            patch_output = inference_patch(agent, current_state, patch_image, patch_mask)
            output[:, :, x:x_end, y:y_end] += patch_output
            counter[:, :, x:x_end, y:y_end] += 1
    output = np.divide(output, counter)
    return output

def inference_patch(agent, current_state, patch_image, mask):
    # only return the final result
    current_state.reset(patch_image)
    mask_squeeze = np.squeeze(mask, axis=1)
    for t in range(0, args.episode_len):
        action, inner_state = agent.act(current_state.tensor)
        action = np.where(mask_squeeze==0, 1, action) # if mask equal to 0, the pixel isn't a rain, so the act should be id==1:"do nothing" 
        current_state.step(action, inner_state)  
    agent.stop_episode()
    return current_state.image

def inference(agent, raw_x, mask, name):
    current_state = State.State(args.move_range)
    _, c, h, w = raw_x.shape
    output = overlapped_process(raw_x, mask, agent, current_state, patch_size=128, stride=64)
    current_state.reset(raw_x)
    p = np.maximum(0,output)
    p = np.minimum(1,p)
    p = (p*255).astype(np.uint8)
    p = np.transpose(p[0], [1,2,0])
    cv2.imwrite(args.save_dir_path+name, p)

def brisque_reward(brisque_metrics, previous_image, current_image):
    previous_image = np.transpose(previous_image[0], [1,2,0]) * 255
    current_image = np.transpose(current_image[0], [1,2,0]) * 255
    reward = brisque_metrics.score(img_as_float(cv2.cvtColor(previous_image, cv2.COLOR_BGR2RGB))) - brisque_metrics.score(img_as_float(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)))
    return reward

def main(args):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        args.data_path, 
        args.image_dir_path)
    
    brisque_metrics = BRISQUE(url=False)

    chainer.cuda.get_device_from_id(args.gpu_id).use()

    current_state = State.State(args.move_range)
    
    train_data_size = MiniBatchLoader.count_paths(args.data_path)
    for i in range(0, train_data_size):
        # train
        model = MyFcn(args.n_actions)
        optimizer = chainer.optimizers.Adam(alpha=args.lr)
        optimizer.setup(model)
        agent = PixelWiseA3C_InnerState(model, optimizer, 5, args.gamma)
        agent.act_deterministically = True
        agent.model.to_gpu()
        raw_x, pseudo_ys, mask, name = mini_batch_loader.load_training_data(index=i)
        print("===== Process for {} =====".format(name))
        for episode in tqdm(range(1, args.max_episode+1)):
            # random crop 128x128
            _, c, h, w = raw_x.shape
            rand_range_h = h-128
            rand_range_w = w-128
            x_offset = np.random.randint(rand_range_w)
            y_offset = np.random.randint(rand_range_h)
            raw_x_crop = raw_x[:, :, y_offset:y_offset+128, x_offset:x_offset+128]
            current_state.reset(raw_x_crop)
            mask_crop = mask[:, :, y_offset:y_offset+128, x_offset:x_offset+128]
            # random sample pseudo_y from 50 pseudo_ys
            rand_index = np.random.randint(0, 50)
            pseudo_y = np.expand_dims(pseudo_ys[rand_index], axis=0)
            # random crop 128x128
            pseudo_y_crop = pseudo_y[:, :, y_offset:y_offset+128, x_offset:x_offset+128]
            reward = np.zeros(pseudo_y_crop.shape, pseudo_y_crop.dtype)
            sum_reward = 0
            for t in range(0, args.episode_len):
                previous_image = current_state.image.copy()
                action, inner_state = agent.act_and_train(current_state.tensor, reward)
                current_state.step_with_mask(action, mask_crop, inner_state)
                reward = np.square(pseudo_y_crop - previous_image)*255 - np.square(pseudo_y_crop - current_state.image)*255
                reward += 0.025 * brisque_reward(brisque_metrics, previous_image.copy(), current_state.image.copy())
                sum_reward += np.mean(reward)*np.power(args.gamma,t)
            agent.stop_episode_and_train(current_state.tensor, reward, True)
            optimizer.alpha = args.lr*((1-episode/args.max_episode)**0.9)

        inference(agent, raw_x, mask, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for training') 
    # seed
    parser.add_argument('--random_seed', type=int, default=1)
    # Directories
    parser.add_argument('--image_dir_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='./dataset/Rain800/testing_pt2.txt')
    parser.add_argument('--save_dir_path', type=str, default='./Results/Rain800/test/SRL-Derain/')
    # config
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--move_range', type=int, default=3)
    parser.add_argument('--episode_len', type=int, default=15)
    parser.add_argument('--max_episode', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_actions', type=int, default=9)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    main(args)