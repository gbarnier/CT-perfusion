################################################################################
################################ Data processing ###############################
################################################################################

################################## One slice ###################################
# Patient S00243
dat/S00243-train-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --min1 60 --max1 120 --min2 80 --max2 140 --i_min3 3 --i_max3 3

dat/S00243-train-win2.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --min1 60 --max1 120 --min2 80 --max2 140 --i_min3 2 --i_max3 2

dat/S00243-train-win3.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --min1 60 --max1 120 --min2 80 --max2 140 --i_min3 4 --i_max3 4

dat/S00243-train-win4.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --min1 60 --max1 120 --min2 80 --max2 140 --i_min3 1 --i_max3 1

dat/S00243-train-full.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1

# Generate data
makeDat:
	make dat/S00243-train-win1.h5 -B
	make dat/S00243-train-win2.h5 -B
	make dat/S00243-train-win3.h5 -B
	make dat/S00243-train-win4.h5 -B
	make dat/S00243-train-full.h5 -B

# QC data
dataQc%:
	# Data wihtout window
	Window3d n4=1 f4=30 < dat/S00243-train-full.h5_ct.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="CT" | Xtpen pixmaps=y &
	Window3d n4=1 f4=30 < dat/S00243-train-full.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="CT+mask" | Xtpen pixmaps=y &
	Window3d f3=0 < dat/S00243-train-full.h5_skull_mask.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=1 title="Total mask" | Xtpen pixmaps=y &
	Window3d f3=0 < dat/S00243-train-full.h5_inner_mask.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=1 title="Inner mask" | Xtpen pixmaps=y &
	# Data subset used for training
	Window3d f4=0 n3=1 f3=3 < dat/S00243-train-full.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Training CT+mask" | Xtpen pixmaps=y &
	Window3d f3=0 < dat/S00243-train-win$*.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Training CT+mask" | Xtpen pixmaps=y &
	Window3d f3=0 < dat/S00243-train-win$*.h5_skull_mask.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=1 title="Training mask" | Xtpen pixmaps=y &
	# Density curves
	Window3d n3=2000 f3=0 < dat/S00243-train-win$*.h5_data_train.H | Graph min2=0.0 max2=120 grid=y title="Training data #1" | Xtpen pixmaps=y &
	Window3d n3=2000 f3=2000 < dat/S00243-train-win$*.h5_data_train.H | Graph min2=0.0 max2=120 grid=y title="Training data #2" | Xtpen pixmaps=y &
	Window3d n3=1000 f3=4000 < dat/S00243-train-win$*.h5_data_train.H | Graph min2=0.0 max2=120 grid=y title="Training data #3" | Xtpen pixmaps=y &

# QC Labels
labelsQc%:
	# Data wihtout window
	Window3d f3=0 < dat/S00243-train-full.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Tmax+mask" | Xtpen pixmaps=y &
	Window3d f3=0 < dat/S00243-train-full.h5_skull_mask.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=1 title="Total mask" | Xtpen pixmaps=y &
	# Data subset used for training
	Window3d f3=0 < dat/S00243-train-win$*.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Tmax+mask" | Xtpen pixmaps=y &
