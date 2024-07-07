from PIL import Image, ImageChops
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
import matplotlib.pyplot as plt
import json

root_path = 'f-mnist/eval/1model'
m_prefix = ''
save_heatmaps = True
title = False
x = 10
y = 2
overwite_heatmaps = True
save_class_figure = True
save_to_heatmap_folder = True


f_mnist = {0: 'T-shirt/top',
  1: 'Trouser',
  2: 'Pullover',
  3: 'Dress',
  4: 'Coat',
  5: 'Sandal',
  6: 'Shirt',
  7: 'Sneaker',
  8: 'Bag',
  9: 'Ankle boot'
  }


ssim_distances = []
mse_distances = []
psnr_distances = []
nrmse_distances = []
l2_distances = []
l1_distances = []
non_zero_pixels = []
pixel_difference = []
stylemix_layers = []
stylemix_seeds = [[] for _ in range(11)]
seeds = [[] for _ in range(11)]
l2_comparison = []
con_classes = [[] for _ in range(10)]

def save_stats(stats_path,
                stylemix_layers,
                ssim_distances,
                mse_distances,
                psnr_distances,
                nrmse_distances,
                l2_distances,
                l1_distances,
                non_zero_pixels,
                pixel_difference,
                seeds,
                stylemix_seeds,
                l2_comparison,
                con_classes = None
              ):

  os.makedirs(stats_path, exist_ok=True)
  boxprops = dict(facecolor='lightgray', color='black', linewidth=1)

  # plt.figure(figsize=(20, 6))
  layers, l_counts = np.unique(stylemix_layers, return_counts=True)
  print(f'layers: {layers}')
  print(f'l_counts: {l_counts}')
  plt.grid(True, alpha=0.25)
  plt.bar(layers, l_counts, align='center', zorder=3)
  plt.gca().set_xticks(layers)
  if title:
    plt.title('Histogram of stylemix layers')
  plt.xlabel('Layer index')
  plt.ylabel('Frequency')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/stylemix_layer.png')
  plt.close()

  if con_classes is not None:
    num_classes = 10

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14, 10))
    axes = axes.ravel()  # Flatten the array of axes

    for i in range(num_classes):
        labels, counts = np.unique(con_classes[i], return_counts=True)
        print(np.max(counts))
        axes[i].grid(True, alpha=0.25)
        axes[i].bar(range(len(counts)), counts, color='skyblue', zorder=3)
        axes[i].set_title(f"Class {i}")
        axes[i].set_xticks(range(len(counts)))
        axes[i].set_xticklabels(labels)
        axes[i].set_ylim(0, 45)  # setting a fixed y-axis limit for better comparison
        # axes[i].set_ylim(0, 65)  # setting a fixed y-axis limit for better comparison

    fig.tight_layout()
    plt.savefig(f'{stats_path}/con_classes.png')
    plt.close()

  plt.figure(figsize=(x, y))
  plt.grid(True, alpha=0.25)
  plt.boxplot(ssim_distances, vert=False, meanline=True, showmeans=True, patch_artist=True, labels=["SSIM"], boxprops=boxprops, zorder=3)
  # ssim_mean = np.mean(ssim_distances)
  # ssim_median = np.median(ssim_distances)
  # # Annotate mean
  # plt.annotate(f'Mean: {ssim_mean:.3f}',
  #             xy=(ssim_mean, 1),
  #             xytext=(ssim_mean, 1.2))

  # # Annotate median
  # plt.annotate(f'Median: {ssim_median:.3f}',
  #             xy=(ssim_median, 1),
  #             xytext=(ssim_median, 1.1))
  if title:
    plt.title('Structural similarity index between original and modified images')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/SSIM.png')
  plt.close()

  plt.figure(figsize=(x, y))
  plt.grid(True, alpha=0.25)
  plt.boxplot(mse_distances, vert=False, meanline=True, showmeans=True, patch_artist=True, boxprops=boxprops, zorder=3)
  if title:
    plt.title('Mean squared error between original and modified images')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/MSE.png')
  plt.close()

  plt.figure(figsize=(x, y))
  plt.grid(True, alpha=0.25)
  plt.boxplot(psnr_distances, vert=False, meanline=True, showmeans=True, patch_artist=True, boxprops=boxprops, zorder=3)
  if title:
    plt.title('Peak signal-to-noise ratio between original and modified images')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/PSNR.png')
  plt.close()

  plt.figure(figsize=(x, y))
  plt.grid(True, alpha=0.25)
  plt.boxplot(nrmse_distances, vert=False, meanline=True, showmeans=True, patch_artist=True, boxprops=boxprops, zorder=3)
  if title:
    plt.title('Normalized root mean squared error between original and modified images')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/NRMSE.png')
  plt.close()

  plt.figure(figsize=(x, y))
  plt.grid(True, alpha=0.25)
  plt.boxplot(l2_distances, vert=False, meanline=True, showmeans=True, patch_artist=True, labels=["  L2"], boxprops=boxprops, zorder=3)
  if title:
    plt.title('L2 of original and modified images difference  L2(original - modified)\L2(original)')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/L2.png')
  plt.close()

  plt.figure(figsize=(x, y))
  plt.grid(True, alpha=0.25)
  plt.boxplot(l1_distances, vert=False, meanline=True, showmeans=True, patch_artist=True, boxprops=boxprops, zorder=3)
  if title:
    plt.title('L1 of original and modified images difference  L1(original - modified)')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/L1.png')
  plt.close()

  plt.figure(figsize=(x, y))
  plt.grid(True, alpha=0.25)
  plt.boxplot(non_zero_pixels, vert=False, meanline=True, showmeans=True, patch_artist=True, boxprops=boxprops, zorder=3)
  if title:
    plt.title('Number of mutated pixels')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/non_zero_pixels.png')
  plt.close()

  plt.figure(figsize=(x, y))
  plt.grid(True, alpha=0.25)
  plt.boxplot(pixel_difference, vert=False, meanline=True, showmeans=True, patch_artist=True, boxprops=boxprops, zorder=3)
  if title:
    plt.title('Pixel value difference between original and modified images')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/pixel_difference.png')
  plt.close()
  # print(f'min pixel difference: {min(pixel_difference)}')
  # print(f'max pixel difference: {max(pixel_difference)}')
  # print(f'average pixel difference: {round(np.mean(pixel_difference), 3)}')
  # print(f'median pixel difference: {np.median(pixel_difference)}')
  # print(f'mean pixel difference: {np.mean(pixel_difference)}')

  plt.figure(figsize=(x, y))
  plt.grid(True, alpha=0.25)
  plt.hist(pixel_difference, bins='auto', zorder=3)
  if title:
    plt.title('Histogram of stylemix_layer')
  plt.xlabel('Value')
  plt.ylabel('Frequency')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/pixel_difference_hist.png')
  plt.close()


  plt.figure(figsize=(15 , 9))
  plt.grid(True, alpha=0.25)
  if isinstance(stylemix_seeds[0], list):
    labels = [f'Class: {i}' for i in range(10)]
    labels.append('Class: 0-9')
    plt.boxplot(stylemix_seeds, vert=False, meanline=True, showmeans=True, labels=labels, patch_artist=True, boxprops=boxprops, zorder=3)
  else:
    plt.boxplot(stylemix_seeds, vert=False, meanline=True, showmeans=True, patch_artist=True, boxprops=boxprops, zorder=3)
  if title:
    plt.title('Number of mutations')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/mutations.png')
  plt.close()

  plt.figure(figsize=(15 , 9))
  plt.grid(True, alpha=0.25)
  if isinstance(seeds[0], list):
    labels = [f'Class: {i}' for i in range(10)]
    labels.append('Class: 0-9')
    plt.boxplot(seeds, vert=False, meanline=True, showmeans=True, labels=labels, patch_artist=True, boxprops=boxprops, zorder=3)
  else:
    plt.boxplot(seeds, vert=False, meanline=True, showmeans=True, patch_artist=True, boxprops=boxprops, zorder=3)
  if title:
    plt.title('Number of Seeds')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/seeds.png')
  plt.close()

  plt.figure(figsize=(x, y))
  plt.grid(True, alpha=0.25)
  plt.boxplot(l2_comparison, vert=False, meanline=True, showmeans=True, patch_artist=True, boxprops=boxprops, zorder=3)
  if title:
    plt.title('L2 difference between original and modified images  (L2(original) - L2(modified))')
  plt.tight_layout()
  plt.savefig(f'{stats_path}/l2_comparison.png')
  plt.close()

model_folder = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
for class_folder in model_folder:
  if not class_folder in ['stats', 'heatmaps']:

    class_ssim_distances = []
    class_mse_distances = []
    class_psnr_distances = []
    class_nrmse_distances = []
    class_l2_distances = []
    class_l1_distances = []
    class_non_zero_pixels = []
    class_pixel_difference = []
    class_stylemix_layers = []
    class_seeds = []
    class_l2_comparison = []

    class_path = os.path.join(root_path, class_folder)
    seed_class = [f for f in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, f))]
    for seed in seed_class:
      seed_path = os.path.join(class_path, seed)
      content = os.listdir(seed_path)
      files = [f for f in content if os.path.isfile(os.path.join(seed_path, f))]
      subfolders = [subfolder for subfolder in content if os.path.isdir(os.path.join(seed_path, subfolder))]
      # class_seeds.append(int(seed))
      seeds[int(class_folder)].append(int(seed))
      seeds[10].append(int(seed))



      for file in files:
        if file.endswith('.png') and not file.startswith('heatmap'):
            img_path = os.path.join(seed_path, file)
            for subfolder in subfolders:
                m_path = os.path.join(seed_path, subfolder, m_prefix)
                if os.path.exists(m_path):
                    m_pngs = sorted([m_png for m_png in os.listdir(m_path) if os.path.isfile(os.path.join(m_path, m_png)) and m_png.endswith('.png')])
                    for m_png in m_pngs[:1]:

                        img = Image.open(img_path)
                        img_array = np.array(img)

                        m_img_path = os.path.join(m_path, m_png)
                        m_img = Image.open(m_img_path)
                        m_img_array = np.array(m_img)

                        m_img_json_path = os.path.join(m_path, m_png.replace('.png', '.json'))
                        with open(m_img_json_path, 'r') as f:
                          data = json.load(f)
                          stylemix_layer = data['stylemix_idx']
                          stylemix_layer = f'{stylemix_layer}'

                          con_class = data["predicted-class"][0]
                          stylemix_seed = data['stylemix_seed']
                          con_classes[int(class_folder)].append(con_class)
                          stylemix_layers.append(stylemix_layer)
                          stylemix_seeds[int(class_folder)].append(stylemix_seed)
                          stylemix_seeds[10].append(stylemix_seed)
                          class_stylemix_layers.append(stylemix_layer)

                        diff = ImageChops.difference(img, m_img)
                        non_zero_elements = np.array(diff)[np.nonzero(diff)]
                        pixel_difference.extend(non_zero_elements)
                        class_pixel_difference.extend(non_zero_elements)
                        non_zero_pixel = np.count_nonzero(diff)
                        non_zero_pixels.append(non_zero_pixel)
                        class_non_zero_pixels.append(non_zero_pixel)
                        ssim_distance = ssim(img_array, m_img_array, data_range=255)
                        ssim_distances.append(ssim_distance)
                        class_ssim_distances.append(ssim_distance)
                        mse_distance = mse(img_array, m_img_array)
                        mse_distances.append(mse_distance)
                        class_mse_distances.append(mse_distance)
                        psnr_distance = psnr(img_array, m_img_array, data_range=255)
                        psnr_distances.append(psnr_distance)
                        class_psnr_distances.append(psnr_distance)
                        nrmse_distance = nrmse(img_array, m_img_array, normalization='euclidean')
                        nrmse_distances.append(nrmse_distance)
                        class_nrmse_distances.append(nrmse_distance)
                        l2_diff = np.linalg.norm(img_array - m_img_array)
                        l2_distance = np.linalg.norm(img_array - m_img_array)/np.linalg.norm(img_array)
                        # if stylemix_layer == 3:
                        #   print(img_path)
                        #   print(round(l2_distance,2))
                        l2_distances.append(l2_distance)
                        class_l2_distances.append(l2_distance)
                        l1_distance = np.linalg.norm(img_array - m_img_array, 1)
                        l1_distances.append(l1_distance)
                        class_l1_distances.append(l1_distance)

                        img_l2 = int(np.linalg.norm(img_array))
                        m_img_l2 = int(np.linalg.norm(m_img_array))
                        l2_comparison.append(img_l2 - m_img_l2)
                        class_l2_comparison.append(img_l2 - m_img_l2)

                        if save_heatmaps:
                          if save_to_heatmap_folder:
                            os.makedirs(f'{root_path}/heatmaps/{class_folder}', exist_ok=True)
                            heatmap_path = os.path.join(f'{root_path}/heatmaps/{class_folder}', f'heatmap-{seed}-{subfolder}.png')
                            # print(heatmap_path)
                          else:
                            heatmap_path = os.path.join(seed_path, f'heatmap-{subfolder}.png')

                          if overwite_heatmaps or not os.path.exists(heatmap_path):
                            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

                            # Display original image
                            axs[0].imshow(img, cmap='gray')

                            if img_l2 < m_img_l2:
                              color = 'red'
                              m_color = 'black'
                            else:
                              color = 'black'
                              m_color = 'red'

                            axs[0].set_title(f'Original Image - Class {f_mnist[int(class_folder)]} - L2: {img_l2}', color=color)

                            # Display modified image
                            axs[1].imshow(m_img, cmap='gray')
                            axs[1].set_title(f'Modified Image - Class {f_mnist[int(subfolder)]} - L2: {m_img_l2}', color=m_color)

                            # Display difference heatmap
                            divider = make_axes_locatable(axs[2])
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            im = axs[2].imshow(diff, cmap='jet', interpolation='nearest')
                            fig.colorbar(im, cax=cax, orientation="vertical")
                            axs[2].set_title(f'Difference Jetmap - D-L2: {int(l2_diff)}, SSIM: {round(ssim_distance *100, 1)}')
                            # im = axs[2].imshow(np.square(diff), cmap='gray', interpolation='nearest')
                            # fig.colorbar(im, cax=cax, orientation="vertical")
                            # axs[2].set_title(f'Difference² - D-L2: {int(l2_distance)}, SSIM: {round(ssim_distance *100, 1)}')
                            print(f'Heatmap saved to {heatmap_path}')
                            plt.tight_layout()
                            plt.savefig(heatmap_path)
                            plt.close()
                          else:
                            print(f'Heatmap already exists at {heatmap_path}')

    if save_class_figure and not class_folder.startswith('heatmap'):
      stats_path = f'{root_path}/stats/{class_folder}'
      save_stats(stats_path,
                  class_stylemix_layers,
                  class_ssim_distances,
                  class_mse_distances,
                  class_psnr_distances,
                  class_nrmse_distances,
                  class_l2_distances,
                  class_l1_distances,
                  class_non_zero_pixels,
                  class_pixel_difference,
                  seeds[int(class_folder)],
                  stylemix_seeds[int(class_folder)],
                  class_l2_comparison
                )

    # print(f'Average SSIM: {round(np.mean(class_ssim_distances), 3)}')
    # print(f'Average MSE: {round(np.mean(class_mse_distances), 3)}')
    # print(f'Average PSNR: {round(np.mean(class_psnr_distances), 3)}')
    # print(f'Average NRMSE: {round(np.mean(class_nrmse_distances), 3)}')
    # print(f'Average L2: {round(np.mean(class_l2_distances), 3)}')
    # print(f'Average L1: {round(np.mean(class_l1_distances), 3)}')
    # print(f'Average non-zero pixels: {round(np.mean(class_non_zero_pixels), 3)}')


stats_path = f'{root_path}/stats/'
save_stats(stats_path,
            stylemix_layers,
            ssim_distances,
            mse_distances,
            psnr_distances,
            nrmse_distances,
            l2_distances,
            l1_distances,
            non_zero_pixels,
            pixel_difference,
            seeds,
            stylemix_seeds,
            l2_comparison,
            con_classes
          )

print(f'Average Seeds: {round(np.mean(seeds[10]), 3)}')
print(f'Max Seeds: {np.max(seeds[10])}')
print(f'Average StyleMix Seeds: {round(np.mean(stylemix_seeds[10]), 3)}')
print(f'Average L2: {round(np.mean(l2_distances), 3)}')
print(f'Average SSIM: {round(np.mean(ssim_distances), 3)}')