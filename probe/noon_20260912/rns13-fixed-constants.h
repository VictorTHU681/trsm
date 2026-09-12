#define RNS_R 13
static const int primes[RNS_R]={251,241,239,233,229,227,223,211,199,197,193,191,181};
static const double crt_hi[RNS_R]={0x1.734041465fdf6p-2,0x1.296d69296d693p-2,0x1.e86f65c1dfddcp-3,0x1.9d1daa4ce8ed5p-1,0x1.7324e40d6a337p-2,0x1.e06c43aa4a6e8p-1,0x1.8ae7e472cc574p-1,0x1.ef03a3ca9db9ap-3,0x1.0b93f09014954p-3,0x1.18b0bb207cc05p-1,0x1.c84a47a07f563p-2,0x1.61d7ca632ee93p-3,0x1.61975646b7de1p-1};
static const double crt_lo[RNS_R]={-0x1.97f7d73404146p-57,-0x1.a4a5b5a4a5b5ap-56,-0x1.ac73ae9819b50p-57,0x1.33a3b5499d1dbp-56,-0x1.c81ad466d8df9p-57,0x1.44cafedf4b8f4p-55,-0x1.b8d33a8c0dc6ap-58,0x1.5d6bfb259c831p-59,-0x1.87134be64577ap-57,0x1.95710e4b5edcfp-56,0x1.ed6e17e02a721p-56,0x1.bcfbfaa384b0fp-57,-0x1.db39fd2bd865dp-57};
static const double crt_scale=0x1.b33bbed413ba7p+10;
// M=2155171312918486894641801011047; inputs |x|<=0.5, Q=2^45; exact centered range admits K<=3481.
static const uint64_t crt_fixed[RNS_R]={UINT64_C(0x5cd0105197f7d734),UINT64_C(0x4a5b5a4a5b5a4a5b),UINT64_C(0x3d0decb83bfbb72a),UINT64_C(0xce8ed5267476a934),UINT64_C(0x5cc939035a8cdb1c),UINT64_C(0xf03621d52537428a),UINT64_C(0xc573f239662b9f92),UINT64_C(0x3de0747953b7342c),UINT64_C(0x21727e120292a73c),UINT64_C(0x8c585d903e602995),UINT64_C(0x721291e81fd58ded),UINT64_C(0x2c3af94c65dd26de),UINT64_C(0xb0cbab235bef0712)};
