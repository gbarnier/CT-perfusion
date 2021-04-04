ACO_GPU_PACK_3D = /net/server2/sep/gbarnier/code/gpu/acousticIsoLib_3D/local/
export PYTHONPATH = ${ACO_GPU_PACK_3D}lib/python3.6/:${ACO_GPU_PACK_3D}lib/python/

################################# 1D analysis ##################################
# Full data
include make-data-full.m

# Test on subslice
include make-data-subset1.m
include make-train-subset1.m

# Test on one full slice
include make-data-subset2.m
include make-train-subset2.m

# Test on one full head
include make-data-subset3.m
include make-train-subset3.m

# Test on 4 heads
include make-data-subset4.m
include make-train-subset4.m

# Test on 8 heads
include make-data-subset5.m
include make-train-subset5.m
include make-train5-base-result.m
include make-train5-fc-result.m
include make-train5-kir-result.m
include make-train5-blog-result.m

# Test on 15 heads
include make-train6-inversion.m

# Debugging
include make-debug.m

################################# 2D analysis ##################################
include make-ctPerf2D-data.m
