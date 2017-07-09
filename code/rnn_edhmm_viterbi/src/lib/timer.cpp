#include "timer.h"
#include <iostream>


void Timer::Tic()
{
	t_start = time(0);
}

void Timer::Toc()
{
	t_end = time(0);
	//std::cerr << "duration: " << difftime(t_end, t_start) << " seconds" << std::endl;
}

time_t Timer::t_start;
time_t Timer::t_end;