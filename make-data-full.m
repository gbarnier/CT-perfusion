################################################################################
############################### Data processing ################################
################################################################################
# Windowing/Clipping values
wind_c = 80
wind_w = 160

################################################################################
################################# Training #####################################
################################################################################
# Patient S00233
dat/S00233.h5:
	./python/CTP_convertDCM.py raw_data/SS00233/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00233/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00239
dat/S00239.h5:
	./python/CTP_convertDCM.py raw_data/SS00239/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00239/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00242
dat/S00242.h5:
	./python/CTP_convertDCM.py raw_data/SS00242/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 8/ raw_data/SS00242/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 833/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00243
dat/S00243.h5:
	./python/CTP_convertDCM.py raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 9/ raw_data/SS00243/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ TMax\ \[s\]\ -\ 933/ ${wind_c} ${wind_w} $@ -v 1

##################################### DEV ######################################
dat/S00250.h5:
	./python/CTP_convertDCM.py raw_data/SS00250/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00250/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0
################################################################################

# Patient S00254
dat/S00254.h5:
	./python/CTP_convertDCM.py raw_data/SS00254/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00254/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S000271
dat/S000271.h5:
	./python/CTP_convertDCM.py raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00271/Ct\ Head\ Perfusion\ Wcontrast/RAPID\ TMax\ \[s\]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1

# Patient S00275
dat/S00275.h5:
	./python/CTP_convertDCM.py raw_data/SS00275/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00275/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00286
dat/S00286.h5:
	./python/CTP_convertDCM.py raw_data/SS00286/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 3/ raw_data/SS00286/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 333/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00287
dat/S00287.h5:
	./python/CTP_convertDCM.py raw_data/SS00287/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00287/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00288
dat/S00288.h5:
	./python/CTP_convertDCM.py raw_data/SS00288/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00288/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00289
dat/S00289.h5:
	./python/CTP_convertDCM.py raw_data/SS00289/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00289/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00291
dat/S00291.h5:
	./python/CTP_convertDCM.py raw_data/SS00291/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00291/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00292
dat/S00292.h5:
	./python/CTP_convertDCM.py raw_data/SS00292/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00292/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

# Patient S00293
dat/S00293.h5:
	./python/CTP_convertDCM.py raw_data/SS00293/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00293/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

##################################### TEST #####################################
# Patient S00295
dat/S00295.h5:
	./python/CTP_convertDCM.py raw_data/SS00295/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00295/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0
################################################################################
# Patient S00297
dat/S00297.h5:
	./python/CTP_convertDCM.py raw_data/SS00297/CT_Head_Perf/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00297/CT_Head_Perf/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0

############################## Anomalous "high" TMax values? ###################
# Patient S00241
# dat/S00241.h5:
# 	./python/CTP_convertDCM.py raw_data/SS00241/CT_Head_Perfusion/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00241/CT_Head_Perfusion/RAPID\ TMax\ [s]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1 --raw 0
#
# # Patient S000267
# dat/S000267.h5:
# 	./python/CTP_convertDCM.py raw_data/SS00267/Ct\ Head\ Perfusion\ Wcontrast/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00267/Ct\ Head\ Perfusion\ Wcontrast/RAPID\ TMax\ \[s\]\ -\ 733/ ${wind_c} ${wind_w} $@ -v 1
#
# # Patient S00306
# dat/S00306.h5:
# 	./python/CTP_convertDCM.py raw_data/SS00306/Ct\ Head\ Perfusion\ Wcontrast/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00306/Ct\ Head\ Perfusion\ Wcontrast/RAPID\ Tmax\ -\ 724/ ${wind_c} ${wind_w} $@ -v 1 --raw 0
#
# # Patient S00318
# dat/S00318.h5:
# 	./python/CTP_convertDCM.py raw_data/SS00318/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00318/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ Tmax\ -\ 724/ ${wind_c} ${wind_w} $@ -v 1 --raw 0
#
# # Patient S00323
# dat/S00323.h5:
# 	./python/CTP_convertDCM.py raw_data/SS00323/Head\ Our_Perfusion_Protocol\ \(Adult\)/Perfusion\ \(v2\)\ 4D\ 10.0\ H20f\ -\ 7/ raw_data/SS00323/Head\ Our_Perfusion_Protocol\ \(Adult\)/RAPID\ Tmax\ -\ 724/ ${wind_c} ${wind_w} $@ -v 1 --raw 1

################################################################################
############################# Launch data computation ##########################
################################################################################
makeData:
	# make dat/S00233.h5 -B
	# make dat/S00239.h5 -B
	# make dat/S00241.h5 -B
	# make dat/S00242.h5 -B
	# make dat/S00243.h5 -B
	# make dat/S00250.h5 -B
	# make dat/S00254.h5 -B
	# make dat/S000267.h5 -B
	# make dat/S000271.h5 -B
	# make dat/S00275.h5 -B
	# make dat/S00286.h5 -B
	# make dat/S00287.h5 -B
	# make dat/S00288.h5 -B
	# make dat/S00289.h5 -B
	make dat/S00291.h5 -B
	make dat/S00292.h5 -B
	make dat/S00293.h5 -B
	make dat/S00295.h5 -B
	make dat/S00297.h5 -B

################################################################################
################################# QC data ######################################
################################################################################
dataFileQc=dat/S00293.h5

# QC data
dataQc-t%:
	Window3d n4=1 f4=$* < ${dataFileQc}_ctm.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Data" | Xtpen pixmaps=y &
	Window3d n4=1 f4=$* < ${dataFileQc}_ct.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=-10 eclip=120 title="Data No mask" | Xtpen pixmaps=y &
	Window3d n4=1 f4=$* < ${dataFileQc}_tmax_m.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Label" | Xtpen pixmaps=y &
	Window3d n4=1 f4=$* < ${dataFileQc}_tmax.H | Grey gainpanel=a grid=y color=j newclip=1 bclip=0 eclip=400 title="Label no mask" | Xtpen pixmaps=y &

# Histogram
ctHisto:
	Window3d n3=1 f3=$* < ${dataFileQc}_data_train.H | Histogram | Graph | Xtpen &
	Attr < ${dataFileQc}_data_train.H

tmaxHisto:
	# Transp plane=12 < dat/S00233.h5_tmax_train.H | Histogram > t1.H
	# Transp plane=12 < dat/S00239.h5_tmax_train.H | Histogram > t2.H
	# Transp plane=12 < dat/S00241.h5_tmax_train.H | Histogram > t3.H
	# Transp plane=12 < dat/S00242.h5_tmax_train.H | Histogram > t4.H
	# Transp plane=12 < dat/S00243.h5_tmax_train.H | Histogram > t5.H
	# Transp plane=12 < dat/S00250.h5_tmax_train.H | Histogram > t6.H
	# Transp plane=12 < dat/S00254.h5_tmax_train.H | Histogram > t7.H
	# Transp plane=12 < dat/S000267.h5_tmax_train.H | Histogram > t8.H
	# Transp plane=12 < dat/S000271.h5_tmax_train.H | Histogram > t9.H
	# Transp plane=12 < dat/S00275.h5_tmax_train.H | Histogram > t10.H
	# Transp plane=12 < dat/S00286.h5_tmax_train.H | Histogram > t11.H
	# Transp plane=12 < dat/S00287.h5_tmax_train.H | Histogram > t12.H
	# Transp plane=12 < dat/S00288.h5_tmax_train.H | Histogram > t13.H
	# Transp plane=12 < dat/S00289.h5_tmax_train.H | Histogram > t14.H
	# Transp plane=12 < dat/S00291.h5_tmax_train.H | Histogram > t15.H
	# Transp plane=12 < dat/S00292.h5_tmax_train.H | Histogram > t16.H
	# Transp plane=12 < dat/S00293.h5_tmax_train.H | Histogram > t17.H
	# Transp plane=12 < dat/S00295.h5_tmax_train.H | Histogram > t18.H
	# Transp plane=12 < dat/S00297.h5_tmax_train.H | Histogram > t19.H
	Cat axis=2 t6.H t7.H t8.H t9.H t10.H | Graph legend=y curvelabel="250:254:267:271:275" legendloc=tr | Xtpen &
	Cat axis=2 t1.H t2.H t3.H t4.H t5.H | Graph legend=y curvelabel="233:239:241:242:243" legendloc=tr | Xtpen &
	Cat axis=2 t11.H t12.H t13.H t14.H t15.H | Graph legend=y curvelabel="286:287:288:289:291" legendloc=tr | Xtpen &
	Cat axis=2 t16.H t17.H t18.H t19.H | Graph legend=y curvelabel="292:293:295:297" legendloc=tr | Xtpen &
