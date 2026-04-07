for noise_rate in 0.0 0.2 0.4 0.6
do
	for group_size in 2 4 8
	do
		for batch_size in 1 4 16
		do
			sbatch scripts/arc/run.sh $noise_rate $group_size $batch_size
		done
	done
done
