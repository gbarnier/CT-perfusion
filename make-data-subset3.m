################################ One full head #################################
wind_c3 = 80
wind_w3 = 160

# Patient S00243
dat/S00243-train3-full.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c3} ${wind_w3} $@ -v 1

dat/S00243-train3-win1.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c3} ${wind_w3} $@ -v 1 --slice 1 --i_min3 0 --i_max3 2

dat/S00243-train3-win2.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c3} ${wind_w3} $@ -v 1 --slice 1 --i_min3 4 --i_max3 10

dat/S00243-train3-win3.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c3} ${wind_w3} $@ -v 1 --slice 1 --i_min3 3 --i_max3 3

dat/S00295-train3-win4.h5:
	./python/CTP_convertDCM.py raw_data/SS00295/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00295/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c4} ${wind_w4} $@ -v 1 --raw 0 --slice 1

# Generate data
makeDat3:
	make dat/S00243-train3-win1.h5 -B
	make dat/S00243-train3-win2.h5 -B
	make dat/S00243-train3-win3.h5 -B
	make dat/S00295-train3-win4.h5 -B

# QC data
dataQcTrain3-win:
	# Data without window
	Window3d n4=1 f4=30 < dat/S00243-train3-win1.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Win1" | Xtpen pixmaps=y &
	Window3d n4=1 f4=30 < dat/S00243-train3-win2.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Win2" | Xtpen pixmaps=y &
	Window3d n4=1 f4=30 < dat/S00243-train3-win3.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Win3" | Xtpen pixmaps=y &
	Window3d n4=1 f4=30 < dat/S00295-train3-win4.h5_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Win4" | Xtpen pixmaps=y &

# QC Labels
labels3Qc:
	Window3d f3=0 < dat/S00243-train3-win1.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Win1" | Xtpen pixmaps=y &
	Window3d f3=0 < dat/S00243-train3-win2.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Win2" | Xtpen pixmaps=y &
	Window3d f3=0 < dat/S00243-train3-win3.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Win3" | Xtpen pixmaps=y &
	Window3d f3=0 < dat/S00243-train3-win4.h5_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Win4" | Xtpen pixmaps=y &
