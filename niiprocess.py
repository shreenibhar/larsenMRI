import os
import sys
import numpy as np
import nibabel as nib


def flatten(img, threshold):
  stack = []
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      for k in range(img.shape[2]):
        if np.mean(img[i, j, k, :]) > threshold:
          stack.append(img[i, j, k, :])
  return np.array(stack).T


def iterate(root, path_list, threshold):
  for file_ in os.listdir(os.path.join(root, *path_list)):
    path_list.append(file_)
    if os.path.isdir(os.path.join(root, *path_list)):
      iterate(root, path_list, threshold) # Go into directories
    else:
      if path_list[-1].endswith(".nii.tar.gz"):
        os.system("tar -xvzf %s" % (os.path.join(root, *path_list)))
        os.system("rm %s" % (os.path.join(root, *path_list))) # Removes extacted file
        path_list[-1] = path_list[-1].strip(".tar.gz") # Updates the name to the new extracted name
      elif path_list[-1].endswith(".nii.gz"):
        os.system("gunzip %s" % (os.path.join(root, *path_list))) # Auto removes extracted file
        path_list[-1] = path_list[-1].strip(".gz") # Updates the name to the new extracted name
      elif not path_list[-1].endswith(".nii"):
        print("Ignoring unsupported file %s" % (os.path.join(root, *path_list)))
      img = nib.load(os.path.join(root, *path_list)).get_fdata()
      flat_img = flatten(img, threshold)
      print(os.path.join(root, *path_list), img.shape, "==>", flat_img.shape)
      path_list[-1] = path_list[-1].strip(".nii") + ".txt" # Updates the name to txt file
      header_txt = "%d %d" % (flat_img.shape[0], flat_img.shape[1]) # Write shape in the first line
      np.savetxt("datatxt/" + "_".join(path_list), flat_img, header=header_txt)
      print("Written flat file")
    path_list.pop()


if __name__ == "__main__":
  threshold = int(sys.argv[1])
  iterate("datanii", [], threshold)
