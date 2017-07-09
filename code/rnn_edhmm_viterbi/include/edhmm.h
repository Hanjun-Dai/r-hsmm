#ifndef EDHMM_H
#define EDHMM_H

#include "config.h"
#include <Eigen/Dense>

using namespace Eigen;

class Sequence;

class EDHMM
{
public:

	static void Init();
	static void Mstep(std::vector<Sequence*>& dataset);
	static MatrixXd p_trans, p_init, p_dur;

private:
	static void Normalize();	
};

#endif