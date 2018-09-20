#include <omp.h>
#include <stdio.h>

long num_steps = 1000000;

double serialExecTime()
{
    double step;

    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;

    double start = omp_get_wtime();

    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;

    printf("Pi = %lf\n", pi);

    return omp_get_wtime() - start;
}

float partial_sum_calculate(int i, double step)
{
    double x, sum;
    x = (i + 0.5) * step;
    sum = 4.0 / (1.0 + x * x);

    return sum;
}

double parallelExecTime()
{
    float Sum;
    double step = 1.0 / (double)num_steps;

    double start = omp_get_wtime();
    int nthrds;
    omp_set_num_threads(2);

#pragma omp parallel
    {
        float partial_sum;
        int i, id;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        for (i = id; i < num_steps; i += nthrds)
        {
            partial_sum = partial_sum_calculate(i, step);

/* Critical section - Only one thread modifies Sum at a time.*/
#pragma omp atomic
            Sum = Sum + partial_sum;
        }
    }

    printf("Number of Threads: %d\nPi = %lf\n", nthrds, Sum * step);

    return omp_get_wtime() - start;
}

int main()
{
    printf("Time taken for Uni-Processor execution: %lf seconds\n", serialExecTime());

    printf("Time taken for Multi-Processor execution: %lf seconds\n", parallelExecTime());

    return 0;
}
