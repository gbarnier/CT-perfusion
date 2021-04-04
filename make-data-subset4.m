########################### Training on mutliple patients ######################
wind_c4 = 80
wind_w4 = 160

# Train
dat/S00243-train4-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 1

dat/S000267-train4-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00267/Ct\ Head\ Perfusion\ Wcontrast/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00267/Ct\ Head\ Perfusion\ Wcontrast/RAPID\ TMax\ \[s\]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1

dat/S00295-train4-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00295/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00295/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

dat/S00275-train4-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00275/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00275/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

# Dev
dat/S00286-train4-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00286/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 3/ raw_data/SS00286/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 333/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

# Test
dat/S00287-train4-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00287/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00287/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

################################################################################
############################# Launch data computation ##########################
################################################################################
makeData4:
	# make dat/S00243-train4-win1.h5 -B
	# make dat/S000267-train4-win1.h5 -B
	make dat/S00295-train4-win1.h5 -B
	# make dat/S00275-train4-win1.h5 -B
	# make dat/S00286-train4-win1.h5 -B
	# make dat/S00287-train4-win1.h5 -B

################################################################################
################################# QC data ######################################
################################################################################
# QC data
data4Qc-t%:
	# Data wihtout window
	Window3d n4=1 f4=$* < dat/S00243-train4-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train1" | Xtpen pixmaps=y &
	Window3d n4=1 f4=$* < dat/S000267-train4-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train2" | Xtpen pixmaps=y &
	Window3d n4=1 f4=$* < dat/S00295-train4-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train3" | Xtpen pixmaps=y &
	Window3d n4=1 f4=$* < dat/S00275-train4-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train4" | Xtpen pixmaps=y &
	Window3d n4=1 f4=$* < dat/S00286-train4-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Dev1" | Xtpen pixmaps=y &
	# Window3d n4=1 f4=$* < dat/S00287-train4-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Test1" | Xtpen pixmaps=y &

tmax4Qc:
	# Data wihtout window
	Window3d < dat/S00243-train4-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train1" | Xtpen pixmaps=y &
	Window3d < dat/S000267-train4-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train2" | Xtpen pixmaps=y &
	Window3d < dat/S00295-train4-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train3" | Xtpen pixmaps=y &
	Window3d < dat/S00275-train4-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train4" | Xtpen pixmaps=y &
	Window3d < dat/S00286-train4-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Dev1" | Xtpen pixmaps=y &
	Window3d < dat/S00287-train4-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Test1" | Xtpen pixmaps=y &

train4Qc:
	Window3d n2=1 f2=100 n3=1 f3=4 n3=2 < dat/S00243-train4-win1.h5_ctm.H | Transp plane=12 > t1.H
	Graph < t1.H | Xtpen &
