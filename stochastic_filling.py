import cv2
import os
import numpy as np
import time

dataset_path = './dataset/Rain100L/'
save_path = './dataset/Rain100L/'
target_path = 'Pseudo_Reference_RDP5/'

def make_folder(path):
   try:
      os.makedirs(path)
   except:
      pass

def all_rain(grid):
   return np.all(grid>0)

def compute_similarity(rainy_image, rdp_image, j, i):

   ## return format ######
   # neighbor = [(i_1,j_1), (i_2,j_2)...... (i_n,j_n)]
   # probability = [p_1, p_2, ... p_n]
   #######################
   # initial
   neighbor, probability = list(), list()
   big_window_size = 9
   Height, Width = rainy_image.shape[0], rainy_image.shape[1]
   # determin
   big_window_size = max(big_window_size, 0)
   big_padding_size = big_window_size//2
   # compute neighbor grid
   for n_j in range(max(j-big_padding_size,0), min(j+big_padding_size+1, Height)):
      for n_i in range(max(i-big_padding_size,0), min(i+big_padding_size+1, Width)):
         if rdp_image[n_j, n_i] == 0: # if is non-rain > candidate
            neighbor.append((n_j, n_i))
            probability.append(1)
   # normalize probability   
   probability = np.array(probability).astype(np.float32)
   probability = probability/ probability.sum()
   return neighbor, probability

### MAIN PROCESS GOES HERE ###
rainy_path = os.path.join(dataset_path, "input")
rdp_path = os.path.join(dataset_path, "RDP5")
rainy_folder = os.listdir(rainy_path)
print(rainy_folder)
print(len(rainy_folder))

# pr setting
pr_num = 50
rdp_intensity_threshold = 10
total_time = 0

## SDR
for image_name in rainy_folder:
   make_folder(os.path.join(save_path, target_path, image_name[:-4]))
   
   start = time.process_time()
   rainy_image    = cv2.imread(os.path.join(rainy_path, image_name))
   rdp_image     = cv2.imread(os.path.join(rdp_path, image_name[:-4]+'.png'), cv2.IMREAD_GRAYSCALE)
   return_images  = np.zeros((pr_num, rainy_image.shape[0], rainy_image.shape[1], rainy_image.shape[2]))
   return_images[:,:,:,:] = rainy_image[:,:,:]
   count = 0
   Height, Width = rainy_image.shape[0], rainy_image.shape[1]
   for j in range(Height):
      for i in range(Width):
         if rdp_image[j,i] > rdp_intensity_threshold:
            count += 1
            neighbor, probability = compute_similarity(rainy_image, rdp_image.copy(), j, i)
            try:
               np.random.seed(0)
               sample = np.random.choice(len(neighbor), pr_num, p = probability)
            except:
               pass
            
            for num in range(pr_num):
               try:
                  pix = neighbor[sample[num]]
                  return_images[num,j,i,:] = rainy_image[pix[0],pix[1],:]
               except:
                  pass
   stop = time.process_time()    
   # save pseudo references
   for i in range(pr_num):
      cv2.imwrite(os.path.join(save_path, target_path, image_name[:-4], str(i)+".png"), return_images[i])
   
   duration = stop - start
   print("SDR: ", image_name, '/Time: ',duration ) 
   total_time += duration
print("Average Time: ", total_time/len(rainy_folder))