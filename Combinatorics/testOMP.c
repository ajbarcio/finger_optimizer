#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv) {
    // int i;
    int thread_id;
    // int variable, v;
    // int numThreads = omp_get_max_threads();
    // int count = 0;
    // // #pragma omp parallel private(thread_id)
    // // thread_id = omp_get_thread_num();
    // #pragma omp parallel for schedule(static,2) private(variable, v) shared(count)	
    // for (int i = 0; i < 20; i++)
	// {
    //     // variable = i*2;
    //     v = i*10;
	// 	printf("Thread %d of %d is running number %d\n", omp_get_thread_num(), numThreads, i);
    //     // #pragma omp barrier
    //     printf("%d %d\n", variable, v);

	// }
    // #pragma omp critical
    // {
    //     count+=v;
    //     printf("%d \n", count);
    // }
	// return 0;

    int partial_Sum, total_Sum;
    int otherNum, otherSum;

    // #pragma omp parallel for schedule(static)

    int numThreads = omp_get_max_threads();
    printf("%d\n", numThreads);

    #pragma omp parallel private(partial_Sum, otherNum) shared(total_Sum, otherSum)
    {
        partial_Sum = 0;
        total_Sum = 0;
        #pragma omp for schedule(static)
        for (int i = 1; i <= 10; i++) {
            
            partial_Sum +=i;
            otherNum = i;
            // thread_id = 
            printf("thread id %d \n", omp_get_thread_num());
        }
        #pragma omp critical
        {
            total_Sum += partial_Sum;
            otherSum += otherNum;
        }
    }

    // for (i = )

    printf("Total Sum: %d\n", total_Sum);
    printf("Other Sum: %d\n", otherSum);
    return 0;
}