#ifndef TIMER_H
#define TIMER_H

#include <ctime>

class Timer
{
public:
		static void Tic();
		static void Toc();
private:
		static time_t t_start, t_end;
};

#endif