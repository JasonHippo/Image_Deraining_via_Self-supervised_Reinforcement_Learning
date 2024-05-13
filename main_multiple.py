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
import time
import cv2

def brisque_reward(brisque_metrics, previous_image, current_image):
    previous_image = np.transpose(previous_image[0], [1,2,0]) * 255
    current_image = np.transpose(current_image[0], [1,2,0]) * 255
    reward = brisque_metrics.score(img_as_float(cv2.cvtColor(previous_image, cv2.COLOR_BGR2RGB))) - brisque_metrics.score(img_as_float(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)))
    return reward

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
    B, C, H, W = raw_x.shape
    if H*W > 535000: # for high resolurion images, we use overlapped inference due to GPU limitations
        output = overlapped_process(raw_x, mask, agent, current_state, patch_size=128, stride=64)
        p = np.maximum(0,output)
        p = np.minimum(1,p)
        p = (p*255).astype(np.uint8)
        p = np.transpose(p[0], [1,2,0])
        
    else:     
        current_state.reset(raw_x)
        mask_squeeze = np.squeeze(mask, axis=1)
        for t in range(0, args.episode_len):
            action, inner_state = agent.act(current_state.tensor)
            action = np.where(mask_squeeze==0, 1, action) # if mask equal to 0, the pixel isn't a rain, so the act should be id==1:"do nothing" 
            current_state.step(action, inner_state)
        agent.stop_episode()
        p = np.maximum(0,current_state.image)
        p = np.minimum(1,p)
        p = (p*255).astype(np.uint8)
        p = np.transpose(p[0], [1,2,0])
    
    cv2.imwrite(os.path.join(args.save_dir_path, name), p)
    
def train(args):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        args.data_path, 
        args.image_dir_path)
    
    brisque_metrics = BRISQUE(url=False)

    chainer.cuda.get_device_from_id(args.gpu_id).use()

    current_state = State.State(args.move_range)
    model = MyFcn(args.n_actions)
    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)
    agent = PixelWiseA3C_InnerState(model, optimizer, 5, args.gamma)
    agent.act_deterministically = True
    agent.model.to_gpu()
    
    os.makedirs(os.path.join(args.save_dir_path, 'model_weight'), exist_ok=True)
    
    train_data_size = MiniBatchLoader.count_paths(args.data_path)
    indices = np.random.permutation(train_data_size)
    i = 0
    for episode in tqdm(range(1, args.max_episode+1)):
        r = indices[i:i+args.batch_size]
        raw_x, pseudo_ys, mask = mini_batch_loader.load_training_batch_data(r, args.img_size)
        current_state.reset(raw_x)
        reward = np.zeros(pseudo_ys.shape, pseudo_ys.dtype)
        sum_reward = 0
        for t in range(0, args.episode_len):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act_and_train(current_state.tensor, reward)
            current_state.step_with_mask(action, mask, inner_state)
            reward = np.square(pseudo_ys - previous_image)*255 - np.square(pseudo_ys - current_state.image)*255
            reward += 0.025 * brisque_reward(brisque_metrics, previous_image.copy(), current_state.image.copy())
            sum_reward += np.mean(reward)*np.power(args.gamma,t)
        agent.stop_episode_and_train(current_state.tensor, reward, True)
        optimizer.alpha = args.lr*((1-episode/args.max_episode)**0.9)

        if i+args.batch_size >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:        
            i += args.batch_size

        if i+2*args.batch_size >= train_data_size:
            i = train_data_size - args.batch_size

        agent.save(os.path.join(args.save_dir_path, 'model_weight', 'last'))

def test(args):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        args.data_path, 
        args.image_dir_path)
    
    chainer.cuda.get_device_from_id(args.gpu_id).use()

    model = MyFcn(args.n_actions)
    optimizer = chainer.optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)
    agent = PixelWiseA3C_InnerState(model, optimizer, 5, 0.99)
    chainer.serializers.load_npz(args.model_weight_path, agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu()
    
    os.makedirs(args.save_dir_path, exist_ok=True)
    
    test_data_size = MiniBatchLoader.count_paths(args.data_path)
    total_process_time = 0
    for i in tqdm(range(0, test_data_size, 1)):
        r = np.array(range(i, i+1))
        raw_x, mask, name = mini_batch_loader.load_testing_batch_data(r)
        begin = time.process_time()
        inference(agent, raw_x, mask, name)   
        end = time.process_time()
        total_process_time += end-begin
    
    print("average process time: ", total_process_time/test_data_size)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters') 
    # seed
    parser.add_argument('--random_seed', type=int, default=1)
    # mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # Directories
    parser.add_argument('--image_dir_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='./dataset/Rain100L/training.txt')
    parser.add_argument('--save_dir_path', type=str, default='./Results/Rain100L/test/SRL-Derain_multiple/')
    parser.add_argument('--model_weight_path', type=str, default='', help='only for inference(test)')
    # config
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--move_range', type=int, default=3)
    parser.add_argument('--episode_len', type=int, default=15)
    parser.add_argument('--max_episode', type=int, default=1000)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_actions', type=int, default=9)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)