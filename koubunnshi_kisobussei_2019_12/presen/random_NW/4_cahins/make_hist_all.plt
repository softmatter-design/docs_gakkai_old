set term pngcairo font "Arial,14" 
set colorsequence classic 
# 
set output "Histgram_all.png"
# set label 1 sprintf("Tg = %.3f", tg) left at tg, y_tg-20 
#
set size square
# set xrange [0:1.0]
#set yrange [0:100]
set xlabel "Arg. Con."
set ylabel "Freq."
set style fill solid 0.5
set boxwidth 0.0021899999999999975
#
plot	"4_chains_5_cells_200_trials_5000_sampling/hist.dat" u 1:($2/5) w l ti "200-5000", \
"4_chains_5_cells_200_trials_1000_sampling/hist.dat" w l ti "200-1000", \
"100_500/4_chains_5_cells_200_trials_1000_sampling/hist.dat" w l ti "50000 200-1000", \
"100_500/4_chains_5_cells_500_trials_1000_sampling/hist.dat" u 1:($2/2.5) w l ti "50000 500-1000", \
"100_500/4_chains_5_cells_500_trials_1000_sampling/hist.dat" u 1:($2/5) w l ti "50000 1000-1000"
