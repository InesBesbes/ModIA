source utils.sh
echo BENCHMARKING THE METHODS
# you can modify these values
p=2
q=2
P=$((p*q))
generate_hostfile $P

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# proper benchmark <--- this could be a TODO for students ? (as in, show weak scaling and/or strong scaling)
mpi_options="-platform platforms/cluster_backbone_256.xml -hostfile hostfiles/hostfile.$P.txt -np $P"
# mpi_options="-platform platforms/cluster_crossbar.xml -hostfile hostfiles/cluster_hostfile.txt -np $P"
b=256
iter=5
traces="bench_traces"
out="bench_outputs"
csv="bench_simple.csv"
echo m,n,k,b,p,q,algo,lookahead,gflops > $csv
for i in 4 8 12
do
  n=$((i*b))
  m=$n
  k=$n
  la=0
  options="-c"
  for algo in p2p bcast
  do
    	run
  done
  for la in $(seq 1 $((k/b)))
  do 
  	algo="p2p-i-la"
  	options="-c -l $la"
    	run
  done
done
