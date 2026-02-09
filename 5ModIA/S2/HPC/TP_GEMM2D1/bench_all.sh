source utils.sh

echo BENCHMARKING THE METHODS

# Define ranges for parameters
p_values=(1 2 4)
q_values=(1 2 4)
b_values=(128 256 512)
iter=5
traces="bench_traces"
out="bench_outputs"
csv="bench.csv"

echo m,n,k,b,p,q,algo,lookahead,gflops > $csv

# Loop over different values of p and q
for p in "${p_values[@]}"; do
  for q in "${q_values[@]}"; do
    P=$((p*q))

    # Loop over different values of b
    for b in "${b_values[@]}"; do
      mpi_options="-platform platforms/cluster_crossbar.xml -hostfile hostfiles/cluster_hostfile.txt -np $P"

      # Loop over different values of m, n, k
      for i in 4 8 12; do
        n=$((i*b))
        m=$n
        k=$n

        # Loop over different algorithms
        for algo in p2p bcast; do
          options="-c"
          run
        done

        # Loop over different lookahead values
        for la in $(seq 1 $((k/b))); do
          algo="p2p-i-la"
          options="-c -l $la"
          run
        done
      done
    done
  done
done
