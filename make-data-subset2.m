################################################################################
################################# Training subset 2 ############################
################################################################################
# Train: /net/server2/sep/gbarnier/projects/CTperf/dat/S00243-train-win1.h5
# Dev: /net/server2/sep/gbarnier/projects/CTperf/dat/S00243-train-win1.h5
# Test: /net/server2/sep/gbarnier/projects/CTperf/dat/S00243-train-win2.h5

################################ One full slice ################################
# Patient S00243
dat/S00243-train2-full.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1

dat/S00243-train2-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --slice 1 --i_min3 3 --i_max3 3

dat/S00243-train2-win2.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --slice 1 --i_min3 2 --i_max3 2

dat/S00243-train2-win3.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --slice 1 --i_min3 4 --i_max3 4

dat/S00243-train2-win4.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --slice 1 --i_min3 2 --i_max3 2

dat/S00243-train2-win5.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --slice 1 --i_min3 1 --i_max3 1

dat/S00243-train2-win6.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --slice 1 --i_min3 5 --i_max3 5

dat/S00295-train2-win7.h5:
	./python/CTP_convertDCM.py raw_data/SS00295/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00295/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0 --slice 1 --i_min3 3 --i_max3 3

# Generate data
makeDat2:
	make dat/S00243-train2-win1.h5 -B
	make dat/S00243-train2-win2.h5 -B
	make dat/S00243-train2-win3.h5 -B
	amke dat/S00243-train2-win4.h5 -B

# QC data
dataQcTrain2-win%:
	# Data wihtout window
	# Window3d n4=1 f4=30 < dat/S00243-train-full.h5_ct.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="CT" | Xtpen pixmaps=y &
	# Window3d n4=1 f4=30 < dat/S00243-train-full.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="CT+mask" | Xtpen pixmaps=y &
	# Window3d f3=0 < dat/S00243-train-full.h5_skull_mask.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=1 title="Total mask" | Xtpen pixmaps=y &
	# Window3d f3=0 < dat/S00243-train-full.h5_inner_mask.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=1 title="Inner mask" | Xtpen pixmaps=y &
	# Data subset used for training
	# Window3d f4=0 n3=1 f3=3 < dat/S00243-train-full.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Training CT+mask" | Xtpen pixmaps=y &
	# Window3d f3=0 < dat/S00243-train2-win$*.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Training CT+mask" | Xtpen pixmaps=y &
	# Window3d f3=0 < dat/S00243-train2-win$*.h5_skull_mask.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=1 title="Training mask" | Xtpen pixmaps=y &
	Window3d f4=0 n3=1 f3=0 < dat/S00295-train2-win7.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Training CT+mask" | Xtpen pixmaps=y &

# QC Labels
labels2QcDisp:
	# Data wihtout window
	# Window3d f3=0 < dat/S00243-train-full.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Tmax+mask" | Xtpen pixmaps=y &
	# Window3d f3=0 < dat/S00243-train-full.h5_skull_mask.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=1 title="Total mask" | Xtpen pixmaps=y &
	# # Data subset used for training
	# Window3d f3=0 < dat/S00243-train2-win$*.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Tmax+mask" | Xtpen pixmaps=y &
	Window3d f4=0 n3=1 f3=0 < dat/S00295-train2-win7.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Tmax" | Xtpen pixmaps=y &

trainQC1-f%:
	Window3d n3=1000 f3=$* < dat/S00243-train2-win2.h5_data_train.H | Graph grid=y min2=-10.0 max2=160 | Xtpen &
