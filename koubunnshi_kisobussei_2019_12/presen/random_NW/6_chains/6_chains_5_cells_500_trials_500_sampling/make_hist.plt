set term pngcairo font "Arial,14" 
set colorsequence classic 
# 
data = "/home/hiroshi/network/random_NW/6_chains/6_chains_5_cells_500_trials_500_sampling/hist.dat" 
set output "/home/hiroshi/network/random_NW/6_chains/6_chains_5_cells_500_trials_500_sampling/Histgram.png"
# set label 1 sprintf("Tg = %.3f", tg) left at tg, y_tg-20 
#
set size square
# set xrange [0:1.0]
#set yrange [0:100]
set xlabel "Arg. Con."
set ylabel "Freq."
set style fill solid 0.5
set boxwidth 0.00020699999999995722
#
plot	data w boxes noti