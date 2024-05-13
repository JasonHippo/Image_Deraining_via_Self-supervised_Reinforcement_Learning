import argparse
import chainer
import numpy as np
from tqdm import tqdm
from brisque import BRISQUE
from mini_batch_loader import MiniBatchLoader
import State
from MyFCN import *
from pixelwise_a3c import *
from skimage.util import img_as_float
import cv2

def inference(agent, raw_x, mask, name):
    current_state = State.State(args.move_range)
    current_state.reset(raw_x)
    mask_squeeze = np.squeeze(mask, axis=1)
    for t in range(0, args.episode_len):
        action, inner_state = agent.act(current_state.tensor)
        action = np.where(mask_squeeze==0, 1, action) # if mask equal to 0, the pixel isn't a rain, so the act should be id==1:"do nothing" 
        current_state.step(action, inner_state)
    agent.stop_episode()
    # SVAE DERAIN RESULT
    p = np.maximum(0,current_state.image)
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
    # DATA LOADER
    mini_batch_loader = MiniBatchLoader(
        args.data_path, 
        args.image_dir_path)
    # METRICS DEFINITION
    brisque_metrics = BRISQUE(url=False)
    # SETTING GPU
    chainer.cuda.get_device_from_id(args.gpu_id).use()
    # STATE
    current_state = State.State(args.move_range)
    # MAIN PROCESS FOR EVERY IMAGES IN DATASET
    train_data_size = MiniBatchLoader.count_paths(args.data_path)
    for i in range(0, train_data_size):
        # INIT MODEL
        model = MyFcn(args.n_actions)
        optimizer = chainer.optimizers.Adam(alpha=args.lr)
        optimizer.setup(model)
        # INIT AGENT
        agent = PixelWiseA3C_InnerState(model, optimizer, 5, args.gamma)
        agent.act_deterministically = True
        agent.model.to_gpu()
        # LOAD IMAGES FROM DATALOADER
        # raw_x: Rainy image (I)
        # pseudo_ys: Pseudo Reference (y^pr)
        # mask: Rain mask (M^r)
        raw_x, pseudo_ys, mask, name = mini_batch_loader.load_training_data(index=i)
        print("===== Process for {} =====".format(name))
        # TRAINING
        for episode in tqdm(range(1, args.max_episode+1)):
            current_state.reset(raw_x)
            # random sample pseudo_y from 50 pseudo_ys
            rand_index = np.random.randint(0, 50)
            pseudo_y = np.expand_dims(pseudo_ys[rand_index], axis=0)
            reward = np.zeros(pseudo_y.shape, pseudo_ys.dtype)
            sum_reward = 0
            for t in range(0, args.episode_len):
                previous_image = current_state.image.copy()
                # FEED TO MODEL
                action, inner_state = agent.act_and_train(current_state.tensor, reward)
                # APPLY ACTION
                current_state.step_with_mask(action, mask, inner_state)
                # COMPUTE REWARDS
                reward = np.square(pseudo_y - previous_image)*255 - np.square(pseudo_y - current_state.image)*255
                reward += 0.025 * brisque_reward(brisque_metrics, previous_image.copy(), current_state.image.copy())
                sum_reward += np.mean(reward)*np.power(args.gamma,t)
            # UPDATE NETWORKS
            agent.stop_episode_and_train(current_state.tensor, reward, True)
            # SCHEDULE LR RATE
            optimizer.alpha = args.lr*((1-episode/args.max_episode)**0.9)
        # TEST AND SAVE DERAIN RESULTS
        inference(agent, raw_x, mask, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for training') 
    # seed
    parser.add_argument('--random_seed', type=int, default=1)
    # Directories
    parser.add_argument('--image_dir_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='./dataset/Rain100L/testing.txt')
    parser.add_argument('--save_dir_path', type=str, default='./Results/Rain100L/test/SRL-Derain/')
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