############################ Crop 2d axial slice ###############################
dat/S00243-test1.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1 --min1 40 --max1 180 --min2 30 --max2 190 --slice 1

slice-disp:
	Window3d n3=10 f3=0 n4=1 f4=40 < dat/S00243-test1.h5_ctm.H | Grey gainpanel=a grid=y | Xtpen pixmaps=y &
