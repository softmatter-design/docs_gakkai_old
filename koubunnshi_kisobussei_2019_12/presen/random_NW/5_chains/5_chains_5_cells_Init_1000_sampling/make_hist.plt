set term pngcairo font "Arial,14" 
set colorsequence classic 
# 
data = "/home/hiroshi/network/random_NW/5_chains/5_chains_5_cells_500_trials_1000_sampling/hist.dat" 
set output "/home/hiroshi/network/random_NW/5_chains/5_chains_5_cells_500_trials_1000_sampling/Histgram.png"
# set label 1 sprintf("Tg = %.3f", tg) left at tg, y_tg-20 
#
set size square
# set xrange [0:1.0]
#set yrange [0:100]
set xlabel "Arg. Con."
set ylabel "Freq."
set style fill solid 0.5
set boxwidth 0.0003250000000000197
#
plot	data w boxes noti