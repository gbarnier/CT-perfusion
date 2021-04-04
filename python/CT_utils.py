# Module containing useful functions for CT image processing
# Module for DCM reading
import pydicom as dicom
import numpy as np
import os
# Modules for image registration
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
# Modules for mask generation
from skimage import morphology
from scipy import ndimage
# Visualization modules
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from celluloid import Camera
# Spatial desampling
from scipy.ndimage import uniform_filter
# Time interpolation
import scipy.interpolate as interpolate
# Image rotation for given time step
from  skimage.measure import regionprops, label, find_contours
from skimage.transform import rotate
# Time filtering
from scipy.ndimage import convolve1d
from skimage import feature


def read_slices(dir_ct, verbose=True):
    """Function to read all slices for a given patient"""
    # Get the number of CT images in the directory
    list_ct_files = os.listdir(dir_ct)
    number_files_ct = len(list_ct_files)
    if verbose:
        print("\tThere are ",number_files_ct," CT images for this patient")

    # Create empty lists
    ct_image_list = [] # Contains pydicom dataset for each ct_image
    ct_snapshot_number = [] # Contains snapshot number (ranging from 0 to number of snapshot-1) for each image
    ct_slice_position = [] # Contains slice position for each image
    list_snapshot = [] # List of snapshots (has only #snapshot elements)

    # Load the data for all the files
    for i_file in range(len(list_ct_files)):

        # Create the full path for the image
        file_full_path = dir_ct + list_ct_files[i_file]

        # Read the image
        ct_image_temp = dicom.read_file(file_full_path)

        # Append to the object list
        ct_image_list.append(ct_image_temp)

        # Append to the snapshot list (one number per snapshot)
        ct_snapshot_number.append(int(ct_image_list[i_file].AcquisitionNumber))

        # Append to the slice position (for a given snapshot, each slice has a different position (i.e., z-coordinate)
        ct_slice_position.append(float(ct_image_list[i_file].SliceLocation))

        # Generate the list of snapshots
        if ct_image_list[i_file].AcquisitionNumber not in list_snapshot:
            list_snapshot.append(int(ct_image_list[i_file].AcquisitionNumber))

    # Get the slices' position for each snapshot
    ct_slice_position_axis = np.unique(ct_slice_position).tolist()

    # Display information
    if verbose:
        print("\tTotal number of files for CT images = ", len(ct_image_list))
        print("\tTotal number of slices per image = ",len(ct_slice_position_axis))
        print("\tTotal number of snapshots for this patient = ",len(list_snapshot))
        print("\tFor one snapshot, these are the slices z-positions = ", ct_slice_position_axis)

    # Image dimension/shape
    n_rows_ct = int(ct_image_list[0].Rows)
    n_cols_ct = int(ct_image_list[0].Columns)
    n_slices_ct = len(ct_slice_position_axis)
    n_snapshot_ct = len(list_snapshot)

    # Shape of numpy array that will contain all the CT images
    shape_ct_image = (n_snapshot_ct,n_slices_ct,n_rows_ct,n_cols_ct)
    n_ct_image = n_snapshot_ct*n_slices_ct*n_rows_ct*n_cols_ct # Total number of elements

    # Allocat array for CT images
    # ct_image = np.zeros(shape_ct_image, dtype=ct_image_list[0].pixel_array.dtype)
    ct_image = np.zeros(shape_ct_image, dtype=np.float32)
    dx_ct = float(ct_image_list[0].PixelSpacing[0])
    dy_ct = float(ct_image_list[0].PixelSpacing[1])
    dz_ct = float(ct_image_list[0].SliceThickness)

    # Create axes for visualization
    x_axis_ct = np.arange(0.0, (n_rows_ct)*dx_ct, dx_ct)
    y_axis_ct = np.arange(0.0, (n_cols_ct)*dy_ct, dy_ct)
    z_axis_ct = np.array(ct_slice_position_axis)

    # Loop over snapshot
    for i_image in range(len(ct_image_list)):
        i_snap_index = list_snapshot.index(ct_snapshot_number[i_image]) # Get the index for snapshot number
        i_slice_index = ct_slice_position_axis.index(ct_slice_position[i_image]) # Get the slice index
        ct_image[i_snap_index,i_slice_index,:,:] = transform_to_hu(ct_image_list[i_image]) # Copy data into numpy array

    # Getting time sampling
    acquistion_time = np.zeros((n_slices_ct, n_snapshot_ct))

    # Loop over all files
    for i_file in range(number_files_ct):

        # Get the slice depth position
        depth = float(ct_image_list[i_file].SliceLocation)

        # Get the index for that depth position
        i_slice_index = ct_slice_position_axis.index(depth)

        # Get the snapshot index
        snap_number = int(ct_image_list[i_file].AcquisitionNumber)
        i_snap_index = list_snapshot.index(snap_number)

        # Get the acquisition time for this file
        acquistion_time_raw = ct_image_list[i_file].AcquisitionTime
        hour = int(str(acquistion_time_raw)[:2])
        min = int(str(acquistion_time_raw)[2:4])
        sec = float(str(acquistion_time_raw)[4:])
        acquistion_time[i_slice_index, i_snap_index] = hour*3600 +min*60 +sec

    return ct_image, x_axis_ct, y_axis_ct, z_axis_ct, acquistion_time

def read_Tmax(tmax_dir, verbose=True):
    """Function to read Tmax slices"""
    list_tmax_files = os.listdir(tmax_dir)
    number_files_tmax = len(list_tmax_files)
    if verbose:
        print("\tThere are ",number_files_tmax," TMax images for this patient")

    # Create an empty list that is going to contain the "pydicom dataset" objects for each image
    tmax_map_list = []
    tmax_snapshot_number = []
    tmax_slice_position = []

    for i_file in range(len(list_tmax_files)):

        # Create the full path for the image
        file_full_path = tmax_dir + list_tmax_files[i_file]

        # Read the image
        tmax_map_temp = dicom.read_file(file_full_path)

        # Append to the object list
        tmax_map_list.append(tmax_map_temp)

        ########### Debug ###########
        # Print dictionary
        # print("Dictionary: ", ct_image_list[0].dir())
        # Append to the snapshot list (one number per snapshot)
        # tmax_snapshot_number.append(int(tmax_map_list[i_file].AcquisitionNumber))
        #############################

        # Append to the slice position (for a given snapshot, each slice has a different position (i.e., z-coordinate)
        tmax_slice_position.append(float(tmax_map_list[i_file].SliceLocation))

    # Create axis for slice positions
    tmax_slice_position_axis = np.unique(tmax_slice_position).tolist()

    # Display information
    if verbose:
        print("\tTotal number of files for TMax maps = ", len(tmax_map_list))
        print("\tTotal number of slices per TMax map = ",len(tmax_slice_position_axis))
        print("\tFor one snapshot, these are the slices z-positions = ", tmax_slice_position_axis)

    # Image dimension/shape
    n_rows_tmax = int(tmax_map_list[0].Rows)
    n_cols_tmax = int(tmax_map_list[0].Columns)
    n_slices_tmax = len(tmax_slice_position_axis)

    # Shape of numpy array that will contain all the TMax slices
    shape_tmax_map = (n_slices_tmax,n_rows_tmax,n_cols_tmax)
    n_tmax_map = n_slices_tmax*n_rows_tmax*n_cols_tmax
    tmax_map = np.zeros(shape_tmax_map, dtype=np.float32)

    # Loop over snapshot
    for i_image in range(len(tmax_map_list)):
        i_slice_index = tmax_slice_position_axis.index(tmax_slice_position[i_image]) # Get the slice index
        tmax_map[i_slice_index,:,:] = tmax_map_list[i_image].pixel_array
    return tmax_map, tmax_slice_position_axis

def transform_to_hu(medical_image):
    """Function to convert image to Hounsfield units"""
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = medical_image.pixel_array * slope + intercept
    return hu_image

def window_image(image, window_center, window_width):
    """Windowing function to clip image values"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

def skull_mask(image):
    """Function to create Skull mask based on brain image (i.e., windowed with window_image(image, 40, 80))"""
    # morphology.dilation creates a segmentation of the image
    # If one pixel is between the origin and the edge of a square of size
    # 5x5, the pixel belongs to the same class

    # We can instead use a circule using: morphology.disk(2)
    # In this case the pixel belongs to the same class if it's between the origin
    # and the radius

    segmentation = morphology.dilation(image, np.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(np.int))
    # The size of label_count is the number of classes/segmentations found

    # We don't use the first class since it's the background
    label_count[0] = 0

    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()

    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    return mask

def image_registration(ref_image, off_image, img_to_reg=None, upsample_fac=50):
    """ Function to co-register off_image with respect to off_image"""
    shift, error, diffphase = phase_cross_correlation(ref_image, off_image, upsample_factor=upsample_fac)
    if img_to_reg is None:
        img_to_reg = off_image
    reg_image = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(img_to_reg), shift)))
    return reg_image


def create_CT_gif(img_slices, gif_file, vmin=0, vmax=80, x_axis=None, y_axis=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    camera = Camera(fig)
    for ii in range(img_slices.shape[0]):
        if x_axis is not None and y_axis is not None:
            cax = ax.pcolormesh(x_axis, y_axis, np.flipud(img_slices[ii,:, :]), vmin=vmin, vmax=vmax, cmap=plt.get_cmap("Greys").reversed())
        else:
            cax = ax.pcolormesh(np.flipud(img_slices[ii,:, :]), vmin=vmin, vmax=vmax, cmap=plt.get_cmap("Greys").reversed())
        camera.snap()
    animation = camera.animate()
    animation.save(gif_file, writer = 'imagemagick')


def image4D_registration(image):
    """Function to co-register CT images"""
    reg_ct_image = np.zeros(image.shape, dtype=np.float32)
    for depth_idx in range(image.shape[1]):
        # Creating reference image
        brain_image0 = window_image(image[0,depth_idx,:, :], 40, 80)
        mask0 = skull_mask(brain_image0)
        reg_ct_image[0,depth_idx,:,:] = image[0,depth_idx,:, :]*mask0
        # Coregistration of all depth slices
        for idx in range(1,image.shape[0]):
            brain_image = window_image(image[idx,depth_idx,:, :], 40, 80)
            mask = skull_mask(brain_image)
            unreg_ct_image = image[idx,depth_idx,:, :]*mask
            reg_ct_image[idx,depth_idx,:,:] = image_registration(mask0, mask, img_to_reg=unreg_ct_image)
    return reg_ct_image

def estimate_rot_ang(image):
    """Function to estimate image rotation on central CT image for each time snapshot"""
    depth_idx = np.int((image.shape[1]+0.5)/2.0)
    ang = []
    for idx in range(image.shape[0]):
        # Creating reference image
        brain_image = window_image(image[idx,depth_idx,:, :], 40, 80)
        mask = label(skull_mask(brain_image))
        props = regionprops(mask)
        ang.append(props[0].orientation*180.0/np.pi)
    return ang

def image4D_rotation(image):
    """Rotate CT images using estimated orientation angle"""
    ang = estimate_rot_ang(image)
    image_rot = np.zeros_like(image)
    for idx in range(image.shape[0]):
        for depth_idx in range(image.shape[1]):
            image_rot[idx, depth_idx, :, :] = rotate(image[idx, depth_idx, :, :], -ang[idx])
    return image_rot, ang

def tmax_rotation(tmax, ang):
    """Rotate Tmax map using estimated orientation angle"""
    tmax_rot = np.zeros_like(tmax)
    for idx in range(tmax.shape[0]):
        tmax_rot[idx, :, :] = rotate(tmax[idx, :, :], -ang[idx])
    return tmax_rot


def window_image4D(image, window_center, window_width):
    """Windowing function to clip image values"""
    wind_image = np.zeros(image.shape, dtype=np.float32)
    for idx in range(image.shape[0]):
        for depth_idx in range(image.shape[1]):
            wind_image[idx,depth_idx,:, :] = window_image(image[idx,depth_idx,:, :], window_center, window_width)
    return wind_image


def spatial_ave_image4d(image, size=2):
    """Function to spatially average input CT 4D image"""
    output_slice = uniform_filter(image[0, 0, :, :], size=size, mode="constant")
    output_image = np.zeros((image.shape[0], image.shape[1], output_slice.shape[0], output_slice.shape[1]))
    for idx in range(image.shape[0]):
        for depth_idx in range(image.shape[1]):
            output_image[idx, depth_idx, :, :] = uniform_filter(image[idx, depth_idx, :, :], size=size)
    return output_image


def interpolate_time_image4d(image, time_axes, nt=80, dt=1.0):
    """Function to interpolate voxel values in time for each time slice"""
    nx = image.shape[3]
    ny = image.shape[2]
    n_depths = image.shape[1]
    n_snapshots = image.shape[0]
    image_int = np.zeros((nt, n_depths, ny, nx))
    tmin = time_axes.flatten().min()
    for idx in range(n_depths):
        t_ax = time_axes[idx,:]
        time_axis = np.linspace(tmin, tmin+(nt-1)*dt, nt)
        time_series = np.reshape(image[:, idx, :, :], (n_snapshots, ny*nx)).T
        f = interpolate.interp1d(t_ax, time_series, kind='linear', fill_value="extrapolate")
        new_time_series = f(time_axis)
        image_int[:, idx, :, :] = np.reshape(new_time_series.T, (nt, ny, nx))
    time_axis -= time_axis[0]
    return image_int, time_axis

def filter_time_image4d(image, n_rect=6):
    filt = np.ones(n_rect)
    filt /= np.sum(filt)
    image_filt = convolve1d(image, filt, axis=0, mode="nearest")
    return image_filt


# Skull stripping mask
def genCont(image, cont):
    """Function to create image contour from coordinates"""
    cont_imag = np.zeros_like(image)
    for ii in range(len(cont)):
        cont_imag[cont[ii,0],cont[ii,1]] = 1
    return cont_imag

def skull_strip_mask(image, bone_hu=110, ct_inf=-110, ct_sup=120):
    """Function to create Skull mask"""
    img_max = image.ravel().max()
    # Selecting areas with certain
    image_mask = np.zeros_like(image).astype(np.int)
    image_mask[(bone_hu < image) & (image < img_max)] = 1
    # Removing objects with area smaller than a certain values
    image_mask_clean = morphology.remove_small_objects(image_mask.astype(bool), 1500)

    # Improving skull definition
    labels, label_nb = ndimage.label(image_mask_clean)
    se = morphology.disk(10)
    close_small_bin = morphology.closing(labels, se)
    # Finding contours of the various areas
    contours = find_contours(close_small_bin,0)

    # Creating masks of the various rounded areas
    areas = []
    masks = []
    for contour in contours:
        cont = genCont(image_mask_clean,np.array(contour,dtype=np.int)).astype(np.int)
        mask = morphology.dilation(cont, np.ones((2, 2)))
        mask = ndimage.morphology.binary_fill_holes(mask)
        mask = morphology.dilation(mask, np.ones((3, 3)))
        masks.append(mask.copy())
        # Computing areas to find correct inner portion
        areas.append(np.sum(mask.ravel()))

    #use_Tmax = flag to check if an inner portion of the brain is present in the current section
    if len(areas) == 1:
        # If only one contour is found, there is no inner portion
        mask = masks[0].astype(np.float32)
        use_Tmax = 0
    elif len(areas) > 1:
        # If two or more contours have been found, take the second-largest one as inner portion
        sort_idx = np.argsort(areas)
        mask = masks[sort_idx[1]].astype(np.float32)
        use_Tmax = 1

        # Improving skull definition
        maskedImg = image * mask
        image_mask = np.zeros_like(maskedImg).astype(np.int)
        image_mask[(ct_inf < maskedImg) & (maskedImg < ct_sup)] = 1
        # Removing objects with area smaller than a certain values
        image_mask_clean = morphology.remove_small_objects(image_mask.astype(bool), 3000)

        # Improving skull definition again
        labels, label_nb = ndimage.label(image_mask_clean)
        se = morphology.disk(2)
        close_small_bin = morphology.closing(labels, se)
        if close_small_bin.max() == 2:
            image_mask = np.zeros_like(maskedImg).astype(np.int)
            image_mask[close_small_bin == 2] = 1
            image_mask_clean = morphology.remove_small_objects(image_mask.astype(bool), 6000)
            mask = ndimage.morphology.binary_fill_holes(image_mask_clean)
            maskedImg = image * mask
    else:
        # No areas have been found
        mask = np.zeros_like(image_mask_clean, dtype=np.float32)
        use_Tmax = 0
    return mask, use_Tmax

def skull_strip_mask_4d(image, bone_hu=110):
    """Function to create inner-brain portion for each depth slice"""
    nz = image.shape[1]
    ny = image.shape[2]
    nx = image.shape[3]
    inner_masks = np.zeros((nz, ny, nx))
    use_Tmax = np.zeros(nz)
    for idx in range(nz):
        inner_masks[idx, :, :], use_Tmax[idx] = skull_strip_mask(image[0, idx, :, :], bone_hu)
    return inner_masks, use_Tmax
