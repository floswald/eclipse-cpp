/*
 * KronProdTest()
 * KronDenseTest()
 * MaxFindTest()
 *
 *  Created on: Jul 11, 2012
 *      Author: florianoswald
 */

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "testheader.h"

using namespace std;
using namespace Eigen;

int main(){
	int i = 4;
	int which_KronProdSP = 0;	//0 is print method, 1 is actual function
	switch (i) {
	case 1:
		KronProdTestLoop();
		break;
	case 2:
		KronProdSPMat2Test( which_KronProdSP );
		break;
	case 3:
		KronProdSPMat3Test( which_KronProdSP );
		break;
	case 4:
		KronProdSPMat4Test( which_KronProdSP );
		break;
	case 5:
		KronDenseTest();
		break;
	case 6:
		MaxFindTest();
		break;
	}
	return 0;
}




