int seed=-1;
  #pragma omp threadprivate(seed)

  int rnd_int() {
    // & 0x7fffffff is equivalent to modulo with RNG_MOD = 2^31
#if defined(_OPENMP)
    if(seed==-1) seed = omp_get_thread_num()+1;
#else
    if(seed==-1) seed = 1;
#endif
    return (seed = (seed * 1103515245 + 12345) & 0x7fffffff);
  }

  void rand_fill(int *x, int n){
    int i;
    for(i=0; i<n; i++){
      x[i]=rnd_int()%n-n/2;
    }
  }

  long usecs (){
    struct timeval t;

    gettimeofday(&t,NULL);
    return t.tv_sec*1000000+t.tv_usec;
  }

  double secs (){
    struct timeval t;

    gettimeofday(&t,NULL);
    return ((double)(t.tv_sec*1000000+t.tv_usec))/1000000.0;
  }

  void do_stuff(double sec){

    long s, e;
    s=0; e=0;
    s = usecs();
    while(((double) e-s)/1000000 < sec)
      {
        e = usecs();
      }
    return;
  }

  int compute_one(double sec){
    do_stuff(sec);
    return 1;
  }

  int f_a(){
    do_stuff(1.0);
    return 1;
  }

  int f_b(){
    do_stuff(1.0);
    return 2;
  }

  int f_c(){
    do_stuff(1.0);
    return 3;
  }

  int f_x(int b, int c){
    do_stuff(1.0);
    return b+c+1;
  }

  int f_y(int a, int x){
    do_stuff(1.0);
    return a+x+2;
  }

  void use_resource(){
    do_stuff(1.0);
    return;
  }
