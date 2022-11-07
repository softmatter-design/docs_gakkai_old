set term pngcairo font "Arial,14" 
set colorsequence classic 
# 
data = "hist.dat" 
data4 = "hist_4.dat"
data5 = "hist_5.dat"
set output "Histgram2.png"
# set label 1 sprintf("Tg = %.3f", tg) left at tg, y_tg-20 
#
set size square
set xrange [0.:1.2]
#set yrange [0:100]
set xlabel "Arg. Con."
set ylabel "Freq."
set style fill solid 0.5
set boxwidth 0.05
#
plot	data w boxes ti "3-Chains", \
data4 w boxes ti "4-Chains", \
data5 w boxes ti "5-Chains"