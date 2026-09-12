#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <errno.h>
static const char* pmu_names[6]={"cpu_cycles","inst_retired","l2d_cache_refill","l1d_cache_refill","stall_frontend","stall_backend"};
static uint64_t pmu_config[6];
static int pmu_fds[1024][6],pmu_ready[1024],pmu_on,pmu_initialized;
static uint64_t pmu_vals[1024][9];
static void pmu_init(void) {
    if(pmu_initialized)return; pmu_initialized=1;
    for(int i=0;i<6;i++) { char path[512],line[128];snprintf(path,sizeof(path),"/sys/bus/event_source/devices/armv8_pmuv3_0/events/%s",pmu_names[i]);FILE*f=fopen(path,"r");
        if(!f||!fgets(line,sizeof(line),f)){perror(path);exit(3);}fclose(f);char* eq=strchr(line,'=');if(!eq)exit(4);pmu_config[i]=strtoull(eq+1,0,0);
    }
}
static void pmu_begin(int tid) {
    if(!pmu_on)return;
    if(!pmu_ready[tid]) {
        for(int i=0;i<6;i++) {
            struct perf_event_attr a={0};a.size=sizeof(a);a.type=PERF_TYPE_RAW;a.config=pmu_config[i];a.disabled=1;a.exclude_kernel=1;a.exclude_hv=1;
            a.read_format=PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
            int fd=syscall(SYS_perf_event_open,&a,0,-1,i?pmu_fds[tid][0]:-1,0);
            if(fd<0){perror("perf_event_open");exit(5);}pmu_fds[tid][i]=fd;
        }pmu_ready[tid]=1;
    }
    ioctl(pmu_fds[tid][0],PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP);
    ioctl(pmu_fds[tid][0],PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP);
}
static void pmu_end(int tid) {
    if(!pmu_on)return;
    ioctl(pmu_fds[tid][0],PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP);
    if(read(pmu_fds[tid][0],pmu_vals[tid],sizeof(pmu_vals[tid]))!=sizeof(pmu_vals[tid])){perror("pmu read");exit(6);}
}
