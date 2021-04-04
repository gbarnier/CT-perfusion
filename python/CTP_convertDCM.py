#! /usr/bin/env python3
import argparse
import numpy as np
import h5py
from CT_utils import *
from matplotlib import pyplot as plt
import copy
import random
from CTP_config import config
import pyVector
import os

if __name__ == "__main__":

    # Parsing command line
    parser = argparse.ArgumentParser(description='Program to create patient formatted file for Tmax estimation')
    parser.add_argument("dcm_dir", help="Directory path containing the DCM files of a patient", type=str)
    parser.add_argument("tmax_dir", help="Directory containing the Tmax value file", type=str)
    parser.add_argument("wind_cent", help="Windowing center value", type=float)
    parser.add_argument("wind_widt", help="Windowing width value", type=float)
    parser.add_argument("out_file", help="Output HDF5 file contaning processed data", type=str)
    parser.add_argument("--verbose","-v", help="Verbosity level of the code", default=True, type=bool)
    parser.add_argument("--size","-s", help="Spatial average filter width", default=2, type=int)
    parser.add_argument("--desample","-d", help="Spatial desampling", default=2, type=int)
    parser.add_argument("--raw", help="Provide raw CT data (1/0)", type=int)
    parser.add_argument("--SEP", help="Output SEP files", default=1, type=int)
    parser.add_argument("--slice", help="Use a subset of data by slicing (set to 1 to activate)", type=int)
    parser.add_argument("--min1", help="Minimum window value on axis #1 [mm]", type=float)
    parser.add_argument("--max1", help="Maximum window value on axis #1 [mm]", type=float)
    parser.add_argument("--min2", help="Minimum window value on axis #2 [mm]", type=float)
    parser.add_argument("--max2", help="Maximum window value on axis #2 [mm]", type=float)
    parser.add_argument("--i_min3", help="Minimum depth index (starts at 0) on axis #3 [mm]", type=int)
    parser.add_argument("--i_max3", help="Maximum depth index (starts at 0) on axis #3 [mm]", type=int)
    args = parser.parse_args()

    ##### Debugging tests
    # x = np.zeros((3,))
    # y = np.reshape(x, (x.shape[0],1,1))
    # print("x shape: ", x.shape)
    # print("y shape: ", y.shape)
    # print("x: ", x)
    # print("y: ", y)
    #
    # z = np.ones((3,2,3))
    # print("z shape: ", z.shape)
    # a1 = z * x
    # a2 = z * y
    # print("a1 shape: ", a1.shape)
    # print("a2 shape: ", a2.shape)

    # exit(0)
    # x = np.arange(0.0, 10.0, 0.75)
    # print("x[11]: ", x[11])
    # print("x shape: ", x.shape)
    # y = x[0:11]
    # print("y: ", y)
    # print("y shape: ", y.shape)
    # exit(0)

    # Processing parsing arguments
    # Display information
    verb = args.verbose

    # Other necessary arguments
    wind_c = args.wind_cent # Hounsfield units window center
    wind_w = args.wind_widt # Hounsfield units window width
    out_file = args.out_file
    size = args.size # Spatial average filter width
    desample = args.desample # Spatial average filter width
    raw = True if args.raw == 1 else False
    SEP = True if args.SEP == 1 else False

    # Paramter to extract a subset of the data
    if args.slice == 1: subset = True
    else: subset = False

    if verb:
        print("Window center [HU]: ", wind_c)
        print("Window width [HU]: ", wind_w)
        print("Output file: ", out_file)
        print("Spatial average filter width: ", args.size)
        if raw: print("User has requested to output raw data")
        if SEP: print("User has requested to output SEP files")

    # Getting file folders
    dmc_dir = args.dcm_dir
    tmax_dir = args.tmax_dir

    # Adding final slash if necessary
    if dmc_dir[-1] != "/":
        dmc_dir += "/"
    if tmax_dir[-1] != "/":
        tmax_dir += "/"

    if verb:
        print("DCM directory: ", dmc_dir)
        print("tmax directory: ", tmax_dir)


    # Reading Tmax file
    # if verb:
    #     print("Reading Tmax values at %s" % tmax_dir)
    # tmax, tmax_z_axis = read_Tmax(tmax_dir)
    # tmax_raw = copy.deepcopy(tmax)
    # tmax_z_axis_raw = copy.deepcopy(tmax_z_axis)
    # if verb:
    #     print("Tmax-map rotation")
    # tmax = tmax_rotation(tmax, ang)
    # exit(0)

    # Output file names
    # out_raw_file = "raw_"+out_file # Output file for raw
    # out_file_ctMasked = "ctMasked_"+out_file # Output file for CT data maked with the mask based on CT values
    # out_file_tmaxMasked = "tmaxMasked_"+out_file # Output file for CT data maked with mase based on TMax values
    # out_ctMask = "ctMask_"+out_file # Output file for the mask based on CT values
    # out_tmaxMask = "ctMask_"+out_file # Output file for the mask based on TMax values
    # out_train = "train_"+out_file # Output file for training format

    ############################################################################
    ######################### Ettore's processing steps ########################
    ############################################################################
    # Processing slices
    if verb:
        print("Reading data at: ",  dmc_dir)
    ct_image, x_axis_ct, y_axis_ct, z_axis_ct, time_axes = read_slices(dmc_dir, verbose=verb)

    # Saving initial "raw" data
    if raw:
        ct_image_raw = copy.deepcopy(ct_image)
        x_axis_ct_raw = copy.deepcopy(x_axis_ct)
        y_axis_ct_raw = copy.deepcopy(y_axis_ct)
        z_axis_ct_raw = copy.deepcopy(z_axis_ct)
        time_axis_raw = np.arange(55,)

    if verb:
        print("Image rotation")
    ct_image, ang = image4D_rotation(ct_image)
    if verb:
        print("Co-registering slices")
    ct_image_reg = image4D_registration(ct_image)

    if verb:
        print("Creating skull-stripping mask")
    inner_mask, use_Tmax = skull_strip_mask_4d(ct_image_reg)
    print("nb of non-zero element inner_mask: ", np.sum(inner_mask))

    if verb:
        print("Windowing slices with window centered at %s and width of %s"%(wind_c, wind_w))
    ct_image_reg_wind = window_image4D(ct_image_reg, wind_c, wind_w)

    # Space desampling (for now simple desampling)
    if verb:
        print("Applying a %s-point moving average on CT images and desampling by a factor of %s" % (size, desample) )
    ct_image_reg_wind = spatial_ave_image4d(ct_image_reg_wind, size=size)[:,:,::desample,::desample]
    x_axis_ct = x_axis_ct[::desample]
    y_axis_ct = y_axis_ct[::desample]
    inner_mask = inner_mask[:,::desample,::desample]

    dx = x_axis_ct[1]-x_axis_ct[0]
    dy = y_axis_ct[1]-y_axis_ct[0]
    dz = z_axis_ct[1]-z_axis_ct[0]

    # Time interpolation
    large_gap = False # Flag to check if a larger time sampling is deteched
    if np.diff(time_axes).max() > 4.0:
        large_gap = True
        print("WARNING! Large time sampling gap detected for %s" % dmc_dir)
    if verb:
        print("Performing time interpolation to regularize time sample")
    ct_image_reg_wind, time_axis = interpolate_time_image4d(ct_image_reg_wind, time_axes)
    n_rect = 6
    if verb:
        print("Performing time filtering with time filter length of %s" % n_rect)
    ct_image_reg_wind = filter_time_image4d(ct_image_reg_wind, n_rect)

    # Reading Tmax file
    if verb:
        print("Reading Tmax values at %s" % tmax_dir)
    tmax, tmax_z_axis = read_Tmax(tmax_dir)
    print("tmax min value: ",np.min(tmax))
    print("tmax max value: ",np.max(tmax))
    tmax_raw = copy.deepcopy(tmax)
    tmax_z_axis_raw = copy.deepcopy(tmax_z_axis)
    if verb:
        print("Tmax-map rotation")
    tmax = tmax_rotation(tmax, ang)

    ############################# Applying mask ################################
    # Creating the mask based on CT values
    print("ct_image_reg_wind shape: ", ct_image_reg_wind.shape)
    ct_avg = np.mean(ct_image_reg_wind,axis=0)
    ct_max = np.max(ct_image_reg_wind,axis=0)
    print("ct_max shape: ", ct_max.shape)
    print("ct_max min: ", np.min(ct_max))
    print("ct_max max: ", np.max(ct_max))
    print("config.ct_inf: ", config.ct_inf)
    print("config.ct_sup: ", config.ct_sup)
    mask_ct = 1.0*(ct_max > config.ct_inf) * (ct_max < config.ct_sup) # Create mask based on CT max/min values
    print("nb of non-zero element mask ct: ", np.sum(mask_ct))
    print("ct_image_reg_wind min: ", np.min(ct_image_reg_wind))
    print("ct_image_reg_wind max: ", np.max(ct_image_reg_wind))
    print("mask_ct shape: ", mask_ct.shape)
    print("ct_image_reg_wind shape: ", ct_image_reg_wind.shape)
    test = mask_ct*ct_image_reg_wind
    print("test shape: ", test.shape)
    print("test min: ", np.min(test))
    print("test max: ", np.max(test))
    print("mask_ct shape: ", mask_ct.shape)

    ########################## Extract a subset of the image ###################
    if subset:

        # Get min/max axes values
        min1=args.min1 if args.min1 != None else x_axis_ct[0]
        max1=args.max1 if args.max1 != None else x_axis_ct[len(x_axis_ct)-1]
        min2=args.min2 if args.min2 != None else y_axis_ct[0]
        max2=args.max2 if args.max2 != None else y_axis_ct[len(x_axis_ct)-1]
        i_min3=args.i_min3 if args.i_min3 != None else 0
        i_max3=args.i_max3 if args.i_max3 != None else len(z_axis_ct)

        print("User has requested to process a subset of the CT image")
        print("x min: ", min1, "[mm]")
        print("x max: ", max1, "[mm]")
        print("y min: ", min2, "[mm]")
        print("y max: ", max2, "[mm]")
        print("iz min: ", i_min3, "[sample]")
        print("iz max: ", i_max3, "[sample]")

        # Compute index for bounds
        i_min1 = np.argmin(np.abs(min1-x_axis_ct)) # Index on the x-axis for min x-value
        i_max1 = np.argmin(np.abs(max1-x_axis_ct)) # Index on the x-axis for max x-value
        i_min2 = np.argmin(np.abs(min2-y_axis_ct)) # Index on the y-axis for min y-value
        i_max2 = np.argmin(np.abs(max2-y_axis_ct)) # Index on the y-axis for max y-value
        if i_min1 >= i_max1 or i_min2 >= i_max2 or i_min3 > i_max3:
            sys.exit('Please provide valid bounds')

        if verb:
            print("i_min1: ", i_min1)
            print("i_max1: ", i_max1)
            print("i_min2: ", i_min2)
            print("i_max2: ", i_max2)
            print("i_min3: ", i_min3)
            print("i_max3: ", i_max3)

        # Update the axes
        print("Shape axes before windowing: ")
        print("x-axis shape: ", x_axis_ct.shape)
        print("y-axis shape: ", y_axis_ct.shape)
        print("z-axis shape: ", z_axis_ct.shape)
        print("z-tmax_z_axis: ", tmax_z_axis)
        print("type z-tmax_z_axis: ", type(tmax_z_axis))
        print("type z_axis_ct: ", type(z_axis_ct))
        x_axis_ct = x_axis_ct[i_min1:i_max1+1]
        y_axis_ct = y_axis_ct[i_min2:i_max2+1]
        z_axis_ct = z_axis_ct[i_min3:i_max3+1]
        tmax_z_axis = tmax_z_axis[i_min3:i_max3+1]
        print("Shape axes after windowing: ")
        print("x-axis shape: ", x_axis_ct.shape)
        print("y-axis shape: ", y_axis_ct.shape)
        print("z-axis shape: ", z_axis_ct.shape)
        print("z-tmax_z_axis: ", tmax_z_axis)
        print("type z-tmax_z_axis: ", type(tmax_z_axis))
        print("type z_axis_ct: ", type(z_axis_ct))

        # Extract the data
        print("Shape arrays before windowing: ")
        print("ct_image_reg_wind shape: ", ct_image_reg_wind.shape)
        print("tmax shape: ", tmax.shape)
        print("inner_mask shape: ", inner_mask.shape)
        print("use_Tmax shape: ", use_Tmax.shape)
        ct_image_reg_wind = ct_image_reg_wind[:,i_min3:i_max3+1,i_min2:i_max2+1,i_min1:i_max1+1]
        tmax = tmax[i_min3:i_max3+1,i_min2:i_max2+1,i_min1:i_max1+1]
        inner_mask = inner_mask[i_min3:i_max3+1,i_min2:i_max2+1,i_min1:i_max1+1]
        mask_ct = mask_ct[i_min3:i_max3+1,i_min2:i_max2+1,i_min1:i_max1+1]
        use_Tmax = use_Tmax[i_min3:i_max3+1]
        print("Shape arrays after windowing: ")
        print("ct_image_reg_wind shape: ", ct_image_reg_wind.shape)
        print("tmax shape: ", tmax.shape)
        print("inner_mask shape: ", inner_mask.shape)
        print("use_Tmax shape: ", use_Tmax.shape)

    print("-"*40)
    print("tmax max: ", np.max(tmax))
    print("tmax min: ", np.min(tmax))
    print("tmax avg: ", np.mean(tmax))
    tmax_qc = 1.0*(tmax > 400)
    print("nb elements greater than 400 in tmax: ", np.sum(tmax_qc))
    print("-"*40)

    ######################### Convert to training format #######################
    print("shape inner_mask: ", inner_mask.shape)
    print("shape mask_ct: ", mask_ct.shape)
    print("nb of non-zero element inner_mask: ", np.sum(inner_mask))
    # print("nb of non-zero element use_Tmax: ", np.sum(use_Tmax))
    print("nb of non-zero element mask_ct: ", np.sum(mask_ct))
    skull_mask = inner_mask*np.reshape(use_Tmax,(use_Tmax.shape[0],1,1))*mask_ct
    # skull_mask = inner_mask*mask_ct
    print("Shape use_Tmax: ", use_Tmax.shape)
    print("nb of non-zero element use_Tmax: ", np.sum(use_Tmax))
    print("nb of non-zero element skull_mask: ", np.sum(skull_mask))
    print("shape skull_mask: ", skull_mask.shape)
    n_train = int(np.sum(skull_mask)) # Compute the number of non-zero elements in the mask
    print("nb of training data point for this patient: ", n_train)

    data_train = np.zeros((n_train, 1, ct_image_reg_wind.shape[0])) # 2D array that contains the CT image for all voxels
    tmax_train = np.zeros((n_train, 1)) # 2D array that contains the TMax values for all voxels
    xyz_index = np.zeros((n_train, 3)) # 2D arrays that contain the (x,y,z) index
    nx = ct_image_reg_wind.shape[3]
    ny = ct_image_reg_wind.shape[2]
    nz = ct_image_reg_wind.shape[1]
    i_train = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                if skull_mask[iz,iy,ix] > 0.0:
                    data_train[i_train, 0, :] = ct_image_reg_wind[:,iz,iy,ix]
                    tmax_train[i_train, 0] = tmax[iz,iy,ix]
                    xyz_index[i_train,2] = iz # Record the index on the z-axis for this example
                    xyz_index[i_train,1] = iy # Record the index on the y-axis for this example
                    xyz_index[i_train,0] = ix # Record the index on the x-axis for this example
                    i_train+=1

    # Apply mask to CT data
    print("Max value CT before mask: ", np.max(ct_image_reg_wind))
    print("Min value CT before mask: ", np.min(ct_image_reg_wind))
    ct_image_reg_wind_mask = ct_image_reg_wind*skull_mask
    tmax_mask = tmax*skull_mask
    print("Max value CT after mask: ", np.max(ct_image_reg_wind_mask))
    print("Min value CT after mask: ", np.min(ct_image_reg_wind_mask))

    # Apply mask one more time
    print("Applying mask twice: ")
    inner_mask_new, use_Tmax_new = skull_strip_mask_4d(ct_image_reg_wind)
    ct_image_reg_wind_new = ct_image_reg_wind*inner_mask_new
    print("Done applying mask twice")

    ############################################################################
    ############################# Saving files #################################
    ############################################################################

    ################################# Qc #######################################
    if verb:
        print("Shape 4D data: ", ct_image_reg_wind.shape)
        print("Shape 3D TMax: ", tmax.shape)
        print("Large gap?: ", large_gap)
        print("Shape of training data: ", data_train.shape)
        print("Shape of labels: ", tmax_train.shape)
        # print("nx: ", x_axis_ct.shape[0], "dx: ", x_axis_ct[1]-x_axis_ct[0], "ox: ", x_axis_ct[0])
        # print("ny: ", y_axis_ct.shape[0], "dy: ", y_axis_ct[1]-y_axis_ct[0], "oy: ", y_axis_ct[0])
        # print("nz: ", z_axis_ct.shape[0], "dz: ", z_axis_ct[1]-z_axis_ct[0], "oz: ", z_axis_ct[0])
        # print("nt: ", time_axis.shape[0], "dt: ", time_axis[1]-time_axis[0], "ot: ", time_axis[0])
        if raw:
            print("Shape 4D raw data: ", ct_image_raw.shape)
            print("Shape 3D raw TMax: ", tmax_raw.shape)
            print("raw nx: ", x_axis_ct_raw.shape[0], "raw dx: ", x_axis_ct_raw[1]-x_axis_ct_raw[0], "raw ox: ", x_axis_ct_raw[0])
            print("raw ny: ", y_axis_ct_raw.shape[0], "raw dy: ", y_axis_ct_raw[1]-y_axis_ct_raw[0], "raw oy: ", y_axis_ct_raw[0])
            print("raw nz: ", z_axis_ct_raw.shape[0], "raw dz: ", z_axis_ct_raw[1]-z_axis_ct_raw[0], "raw oz: ", z_axis_ct_raw[0])
            print("raw nt: ", time_axis_raw.shape[0], "raw dt: ", time_axis_raw[1]-time_axis_raw[0], "raw ot: ", time_axis_raw[0])

    ############################### Processed data #############################
    # HF files
    if verb:
        print("Writing output .h5 file: ", out_file)
    hf = h5py.File(out_file, 'w')

    # Data/Labels
    hf.create_dataset('4d_cube', data=ct_image_reg_wind) # 4D pre-processed data (not masked)
    hf.create_dataset('tmax', data=tmax) # TMax 3D map
    hf.create_dataset('data_train', data=data_train) # Data processed + masked reshaped in 3D for PyTorch
    hf.create_dataset('tmax_train', data=tmax_train) # Labeled processed + masked reshaped in 2D for PyTorch
    hf.create_dataset('xyz_index', data=xyz_index) # 3D Array containing indicied for training points
    hf.create_dataset('skull_mask', data=skull_mask) # Flag to check whether an inner brain portion has been found
    hf.create_dataset('4d_cube_m', data=ct_image_reg_wind_mask) # Flag to check whether an inner brain portion has been found
    hf.create_dataset('tmax_m', data=tmax_mask) # Flag to check whether an inner brain portion has been found
    hf.create_dataset('4d_cube_new', data=ct_image_reg_wind_new) # Flag to check whether an inner brain portion has been found
    # Axes
    hf.create_dataset('x_axis', data=x_axis_ct)
    hf.create_dataset('y_axis', data=y_axis_ct)
    hf.create_dataset('z_axis', data=z_axis_ct)
    hf.create_dataset('large_gap', data=large_gap) # Time-axis QC
    hf.create_dataset('time_axis', data=time_axis)
    hf.create_dataset('tmax_z_axis', data=tmax_z_axis)

    hf.create_dataset('mask_ct', data=mask_ct) # 3D mask based on TMax values
    hf.create_dataset('inner_mask', data=inner_mask) # Skull-stripping mask
    # hf.create_dataset('use_Tmax', data=use_Tmax) # Flag to check whether an inner brain portion has been found

    # Write raw data
    if raw:
        hf.create_dataset('4d_cube_raw', data=ct_image_raw)
        hf.create_dataset('x_axis_raw', data=x_axis_ct_raw)
        hf.create_dataset('y_axis_raw', data=y_axis_ct_raw)
        hf.create_dataset('z_axis_raw', data=z_axis_ct_raw)
        hf.create_dataset('time_axis_raw', data=time_axis_raw)
        hf.create_dataset('tmax_raw', data=tmax_raw)
        hf.create_dataset('tmax_z_axis_raw', data=tmax_z_axis_raw)
    # Close file
    hf.close()

    # Write SEP files
    if SEP:
        if verb:
            print("Writing file for processed data in SEP format")
        hf = h5py.File(out_file, 'r')

        # Get axis information
        ox = x_axis_ct[0]
        oy = y_axis_ct[0]
        oz = z_axis_ct[0]

        # Processed CT image
        imageCT = np.array(hf.get("4d_cube"), dtype=np.float32)
        vec = pyVector.vectorIC(imageCT)
        vec.writeVec(out_file+"_ct.H")
        command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file+"_ct.H"
        os.system(command)

        # TMax map
        tmax = np.array(hf.get("tmax"), dtype=np.float32)
        vec = pyVector.vectorIC(tmax)
        vec.writeVec(out_file+"_tmax.H")
        command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file+"_tmax.H"
        os.system(command)

        # Training data
        data_train = np.array(hf.get("data_train"), dtype=np.float32)
        vec = pyVector.vectorIC(data_train)
        vec.writeVec(out_file+"_data_train.H")

        # Training labels
        tmax_train = np.array(hf.get("tmax_train"), dtype=np.float32)
        vec = pyVector.vectorIC(tmax_train)
        vec.writeVec(out_file+"_tmax_train.H")

        # Position indices
        xyz_index = np.array(hf.get("xyz_index"), dtype=np.float32)
        vec = pyVector.vectorIC(xyz_index)
        vec.writeVec(out_file+"_xyz_index.H")

        # Skull mask
        skull_mask = np.array(hf.get("skull_mask"), dtype=np.float32)
        vec = pyVector.vectorIC(skull_mask)
        vec.writeVec(out_file+"_skull_mask.H")
        command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file+"_skull_mask.H"
        os.system(command)

        # Inner mask
        mask_ct = np.array(hf.get("mask_ct"), dtype=np.float32)
        vec = pyVector.vectorIC(mask_ct)
        vec.writeVec(out_file+"_mask_ct.H")
        command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file+"_mask_ct.H"
        os.system(command)

        # CT data after masking
        imageCT_masked = np.array(hf.get("4d_cube_m"), dtype=np.float32)
        vec = pyVector.vectorIC(imageCT_masked)
        vec.writeVec(out_file+"_ctm.H")
        command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file+"_ctm.H"
        os.system(command)

        # Tmax after masking
        tmax_m = np.array(hf.get("tmax_m"), dtype=np.float32)
        vec = pyVector.vectorIC(tmax_m)
        vec.writeVec(out_file+"_tmax_m.H")
        command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file+"_tmax_m.H"
        os.system(command)

        # Tmax after masking
        inner_mask = np.array(hf.get("inner_mask"), dtype=np.float32)
        vec = pyVector.vectorIC(inner_mask)
        vec.writeVec(out_file+"_inner_mask.H")
        command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file+"_inner_mask.H"
        os.system(command)

        # Tmax after masking
        cube_new = np.array(hf.get("4d_cube_new"), dtype=np.float32)
        vec = pyVector.vectorIC(cube_new)
        vec.writeVec(out_file+"_ctm_new.H")
        command="echo 'o1="+str(ox)+" d1="+str(dx)+" o2="+str(oy)+" d2="+str(dy)+ " o3="+str(oz)+" d3="+str(dz)+"'>> "+out_file+"_ctm_new.H"
        os.system(command)

        # SEP files
        if raw:
            if verb: print("Writing SEP files for raw data")
            hf_raw = h5py.File(out_file, 'r')
            imageCT_raw = np.array(hf_raw.get("4d_cube_raw"), dtype=np.float32)
            vec = pyVector.vectorIC(imageCT_raw)
            vec.writeVec(out_file+"_ct_raw.H")
            tmax_raw = np.array(hf_raw.get("tmax_raw"), dtype=np.float32)
            vec = pyVector.vectorIC(tmax_raw)
            vec.writeVec(out_file+"_tmax_raw.H")
