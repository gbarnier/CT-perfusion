########################### Training on mutliple patients ######################
wind_c5 = 80
wind_w5 = 160

# Train #1
dat/S00243-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c4} ${wind_w4} $@ -v 1

# Train #2
dat/S000267-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00267/Ct\ Head\ Perfusion\ Wcontrast/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00267/Ct\ Head\ Perfusion\ Wcontrast/RAPID\ TMax\ \[s\]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1

# Train #3
dat/S000271-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/RAPID\ TMax\ \[s\]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1

# Train #4
dat/S00275-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00275/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00275/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

# Train #5
dat/S00286-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00286/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 3/ raw_data/SS00286/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 333/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

# Train #6
dat/S00287-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00287/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00287/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

# Train #7
dat/S00295-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00295/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00295/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

# Train #8
dat/S00292-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00292/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00292/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

# Dev #1
dat/S00293-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00293/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00293/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

# Test #1
dat/S00291-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00291/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00291/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0

# Patient S00289
dat/S00289-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00289/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00289/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00288
dat/S00288-train5-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00288/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00288/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

################################################################################
############################# Launch data computation ##########################
################################################################################
makeData5:
	# make dat/S000271-train5-win1.h5 -B
	# make dat/S00293-train5-win1.h5 -B
	# make dat/S00292-train5-win1.h5 -B
	# make dat/S00243-train5-win1.h5 -B
	# make dat/S000267-train5-win1.h5 -B
	# make dat/S00295-train5-win1.h5 -B
	# make dat/S00275-train5-win1.h5 -B
	# make dat/S00286-train5-win1.h5 -B
	# make dat/S00287-train5-win1.h5 -B
	make dat/S00291-train5-win1.h5 -B
	make dat/S00289-train5-win1.h5 -B
	make dat/S00288-train5-win1.h5 -B

################################################################################
################################# QC data ######################################
################################################################################
# QC data
data5Qc-t%:
	# Data wihtout window
	Window3d n4=1 f4=$* < dat/S00243-train5-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train1" | Xtpen pixmaps=y &
	# Window3d n4=1 f4=$* < dat/S000267-train5-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train2" | Xtpen pixmaps=y &
	# Window3d n4=1 f4=$* < dat/S00295-train5-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train3" | Xtpen pixmaps=y &
	# Window3d n4=1 f4=$* < dat/S00275-train5-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train4" | Xtpen pixmaps=y &
	# Window3d n4=1 f4=$* < dat/S00286-train5-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train5" | Xtpen pixmaps=y &
	# Window3d n4=1 f4=$* < dat/S00287-train5-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train6" | Xtpen pixmaps=y &
	# Window3d n4=1 f4=$* < dat/S00318-train5-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train7" | Xtpen pixmaps=y &
	# Window3d n4=1 f4=$* < dat/S00323-train5-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train8" | Xtpen pixmaps=y &
	# Window3d n4=1 f4=$* < dat/S00306-train5-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Train9" | Xtpen pixmaps=y &

tmax5Qc:
	# Data wihtout window
	Window3d < dat/S00243-train5-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train1" | Xtpen pixmaps=y &
	Window3d < dat/S000267-train5-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train2" | Xtpen pixmaps=y &
	Window3d < dat/S000271-train5-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train3" | Xtpen pixmaps=y &
	Window3d < dat/S00275-train5-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train4" | Xtpen pixmaps=y &
	Window3d < dat/S00286-train5-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train5" | Xtpen pixmaps=y &
	Window3d < dat/S00287-train5-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train6" | Xtpen pixmaps=y &
	Window3d < dat/S00295-train5-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train7" | Xtpen pixmaps=y &
	Window3d < dat/S00293-train5-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Train8" | Xtpen pixmaps=y &
	Window3d < dat/S00292-train5-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Dev1" | Xtpen pixmaps=y &



train5Qc:
	Window3d n2=1 f2=100 n3=1 f3=4 n3=2 < dat/S00243-train5-win1.h5_ctm.H | Transp plane=12 > t1.H
	Graph < t1.H | Xtpen &

testFig-t%:
	Window3d n3=1 f3=3 < dat/S00243-train4-win1.h5_ct_raw.H | Grey grid=y color=g newclip=1 bclip=0 eclip=160 title="Raw" | Xtpen pixmaps=y &
	Window3d n3=1 f3=3 < dat/S00243-train4-win1.h5_ct.H | Grey grid=y color=g newclip=1 bclip=0 eclip=160 title="CT" | Xtpen pixmaps=y &
	Window3d n3=1 f3=3 < dat/S00243-train4-win1.h5_ctm.H | Grey grid=y color=g newclip=1 bclip=0 eclip=160 title="CT+Mask" | Xtpen pixmaps=y &
