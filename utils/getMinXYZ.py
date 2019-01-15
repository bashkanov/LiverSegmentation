import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

rootdir = os.getcwd()

img_dir = os.path.join(rootdir, 'data/train/img/')
seg_dir = os.path.join(rootdir, 'data/train/seg/')

widths = []
heights = []
depths = []
paths = []
spaces = []

for filename in sorted(os.listdir(img_dir)):
    if filename.endswith(".nii.gz") or filename.endswith(".nii"):
        # print(os.path.join(img_dir, filename))
        image = sitk.ReadImage(os.path.join(img_dir, filename))
        paths.append(str(filename))
        widths.append(image.GetWidth())
        heights.append(image.GetHeight())
        depths.append(image.GetDepth())
        spaces.append(image.GetSpacing()[0])

print("max_widths: {}, path: {} \n"
      "max_heights: {}, path: {} \n"
      "max_depths: {}, path: {}\n"
      "max_space: {}, path: {}".format(max(widths), paths[widths.index(max(widths))],
                                       max(heights), paths[heights.index(max(heights))],
                                       max(depths), paths[depths.index(max(depths))],
                                       max(spaces), paths[spaces.index(max(spaces))]))

print("\nmin_widths: {}, path: {} \n"
      "min_heights: {}, path: {} \n"
      "min_depths: {}, path: {} \n"
      "min_space: {}, path: {}".format(min(widths), paths[widths.index(min(widths))],
                                       min(heights), paths[heights.index(min(heights))],
                                       min(depths), paths[depths.index(min(depths))],
                                       min(spaces), paths[spaces.index(min(spaces))]))

# plot slice
fig, axs = plt.subplots(1, 3, figsize=(6, 3), sharey=True)

axs[0].hist(depths, color='blue', edgecolor='black')
axs[0].set_title('Depth distribution')
axs[1].hist(heights, color='blue', edgecolor='black')
axs[1].set_title('Heights distribution')
axs[2].hist(widths, color='blue', edgecolor='black')
axs[2].set_title('Widths distribution')
# axs[3].hist(spaces, color='blue', edgecolor='black')
# axs[3].set_title('Spaces distribution')

plt.show(fig)
