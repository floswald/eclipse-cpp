/*
 * testheader.h
 *
 *  Created on: Jul 11, 2012
 *      Author: florianoswald
 */

#ifndef TESTHEADER_H_
#define TESTHEADER_H_

#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef MatrixXf Dynamic2D;
typedef VectorXf Dynamic1D;
typedef Matrix<int, Dynamic, 1> VectorXi;


/////////////////////////////////////////
// Function definitions
/////////////////////////////////////////

// kronecker product for 2 dense matrices
Eigen::VectorXd kronproddense(
		Eigen::MatrixXd a0,
		Eigen::MatrixXd a1,
		Eigen::VectorXd y )
{
	Eigen::VectorXd retvec = Eigen::VectorXd::Zero( a0.rows() * a1.rows() );

	//iterate over rows
	for (int row_idx0=0; row_idx0<a0.rows(); ++row_idx0) {

		int row_offset1 = row_idx0;		// how much to offset index for second matrix a1?
		row_offset1    *= a1.rows();	// tells you size of jump you have to make

		for (int row_idx1=0; row_idx1<a1.rows(); ++row_idx1) {

			//start looping over columns now
			//columns of a0:
			for (int col_idx0=0; col_idx0<a0.cols(); ++col_idx0 ) {

				int col_offset1 = col_idx0;		// same offsetting story for columns
				col_offset1    *= a1.cols();
				double factor1  = a0(row_idx0,col_idx0);	// precompute parts of the multiplication. if many dimensions, that saves a lot of access operations!

				// columns of a1:
				for (int col_idx1=0; col_idx1<a1.cols(); ++col_idx1 ) {

					// compute product at the corresponding index
					retvec( row_offset1 + row_idx1 ) += factor1 * a1(row_idx1,col_idx1) * y( col_offset1 + col_idx1 );
				}
			}
		}
	}
	return retvec;
}


// KronProdMat2looptest: illustrates looping over 2 sparse matrices.
//computes kronecker(a0,a1) * y
// dim(a0) = (n0,m0)
// dim(a1) = (n1,m1)
// length(y) = m0 * m1
// for a0 and a1 SPARSE matrices!
Eigen::VectorXd KronProdSPMat2looptest(
		Eigen::SparseMatrix<double, Eigen::RowMajor> a0,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a1,
		Eigen::VectorXd y) {

	cout << "This is KronProdSPMat2looptest." << endl;
	cout << "Illustrates looping over sparse matrix." << endl;

	Eigen::VectorXd retvec( a0.rows() * a1.rows() );
	if ( y.rows() != a0.cols() * a1.cols() ) {
		cout << "KronProdMat2 error: y and matrices not conformable" << endl;
	}
	cout << "a0.outerSize() " << a0.outerSize() << endl;
	cout << "a0 " << endl;
	cout << a0 << endl;

	for (int row_idx0=0; row_idx0<a0.outerSize(); ++row_idx0) {
		int row_offset1 = row_idx0;
		row_offset1    *= a1.rows();
		cout << "row loop. row number: " << endl;
		cout << row_idx0 << endl;

		// go over columns (inner indices in sparse matrix)
				for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it0(a0,row_idx0); it0; ++it0) {
					double val = it0.value();
					cout << "col iterator number " << it0.col() << endl;
					cout << "it0.value() " << val << endl;
				}


	}
	return retvec;
}

// KronProdSPMat2Print
// computes kronecker(a0,a1) * y, a0 and a1 SPARSE matrices!
// and PRINTS intermediate results and index values.
// dim(a0) = (n0,m0)
// dim(a1) = (n1,m1)
// length(y) = m0 * m1
Eigen::VectorXd KronProdSPMat2Print(
		Eigen::SparseMatrix<double, Eigen::RowMajor> a0,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a1,
		Eigen::VectorXd y) {

	Eigen::VectorXd retvec;
	retvec.setZero( a0.rows() * a1.rows() );
	if ( y.rows() != a0.cols() * a1.cols() ) {
		cout << "KronProdMat2 error: y and matrices not conformable" << endl;
	}
	cout << "a0.outerSize() " << a0.outerSize() << endl;
	cout << "a0 " << endl;
	cout << a0 << endl;

	for (int row_idx0=0; row_idx0<a0.outerSize(); ++row_idx0) {
		int row_offset1 = row_idx0;
		row_offset1    *= a1.rows();
		cout << "row loop. row number: " << endl;
		cout << row_idx0 << endl;

		for (int row_idx1=0; row_idx1<a1.outerSize(); ++row_idx1) {

			// go over columns (inner indices in sparse matrix)
			for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it0(a0,row_idx0); it0; ++it0) {
				cout << "it0.index() = " << it0.index() << endl;
				int col_offset1 = it0.index();
				col_offset1    *= a1.innerSize();
				cout << "col_offset1 = " << col_offset1 << endl;
				double factor1 = it0.value();

				cout << "innermost loop" << endl;
				for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it1(a1,row_idx1); it1; ++it1){
					cout << endl;
					cout << "at a1 index: " << it1.index() << endl;
					cout << "value of product at position is: " << endl;
					cout << factor1 * it1.value() << endl;
					cout << "this written to retvec index:" << endl;
					cout << row_offset1 + row_idx1 << endl;
					cout << "multiplied with y index of" << endl;
					cout << col_offset1 + it1.index() << endl;
					// retvec( row_offset1 + row_idx1 ) += factor1 * it1.value() * y( col_offset1 + it1.index() )
				}
			}

		}
	}
	return retvec;
}


// KronProdSPMat2
// computes kronecker(a0,a1) * y, a0 and a1 SPARSE matrices!
// dim(a0) = (n0,m0)
// dim(a1) = (n1,m1)
// length(y) = m0 * m1
Eigen::VectorXd KronProdSPMat2(
		Eigen::SparseMatrix<double, Eigen::RowMajor> a0,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a1,
		Eigen::VectorXd y) {

	Eigen::VectorXd retvec;
	retvec.setZero( a0.rows() * a1.rows() );
	if ( y.rows() != a0.cols() * a1.cols() ) {
		cout << "KronProdMat2 error: y and matrices not conformable" << endl;
	}
	cout << "a0.outerSize() " << a0.outerSize() << endl;
	cout << "a0 " << endl;
	cout << a0 << endl;

	for (int row_idx0=0; row_idx0<a0.outerSize(); ++row_idx0) {
		int row_offset1 = row_idx0;
		row_offset1    *= a1.rows();
		cout << "row loop. row number: " << endl;
		cout << row_idx0 << endl;

		for (int row_idx1=0; row_idx1<a1.outerSize(); ++row_idx1) {

			// go over columns (inner indices in sparse matrix)
			for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it0(a0,row_idx0); it0; ++it0) {
				cout << "it0.index() = " << it0.index() << endl;
				int col_offset1 = it0.index();
				col_offset1    *= a1.innerSize();
				cout << "col_offset1 = " << col_offset1 << endl;
				double factor1 = it0.value();

				cout << "innermost loop" << endl;
				for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it1(a1,row_idx1); it1; ++it1){
					cout << endl;
					cout << "this written to retvec index:" << endl;
					cout << row_offset1 + row_idx1 << endl;
					cout << "multiplied with y index of" << endl;
					cout << col_offset1 + it1.index() << endl;
					retvec( row_offset1 + row_idx1 ) += factor1 * it1.value() * y( col_offset1 + it1.index() );
				}
			}

		}
	}
	cout << "retvec = " << endl;
	cout << retvec << endl;
	return retvec;
}



// KronProdSPMat3Print
// computes kronecker(a0,kronecker(a1,a2)) * y, a0, a1, a2 SPARSE matrices!
// dim(a0) = (n0,m0)
// dim(a1) = (n1,m1)
// dim(a2) = (n2,m2)
// length(y) = m0 * m1 * m2
Eigen::VectorXd KronProdSPMat3Print(
		Eigen::SparseMatrix<double, Eigen::RowMajor> a0,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a1,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a2,
		Eigen::VectorXd y) {

	Eigen::VectorXd retvec;
	retvec.setZero( a0.rows() * a1.rows() * a2.rows() );
	if ( y.rows() != a0.cols() * a1.cols() * a2.cols() ) {
		cout << "KronProdMat3 error: y and matrices not conformable" << endl;
	}

	//loop rows a0
	for (int row_idx0=0; row_idx0<a0.outerSize(); ++row_idx0) {
		int row_offset1 = row_idx0;
		row_offset1    *= a1.rows();
		cout << "row1 loop. row number: " << endl;
		cout << row_idx0 << endl;

		// loop rows a1
		for (int row_idx1=0; row_idx1<a1.outerSize(); ++row_idx1) {
			int row_offset2 = row_offset1 + row_idx1;
			row_offset2    *= a2.rows();
			cout << "row2 loop. row number: " << endl;
			cout << row_idx1 << endl;

			// loop rows a2
			for (int row_idx2=0; row_idx2<a2.outerSize(); ++row_idx2) {

				// loop cols a0 (non-zero elements only)
				for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it0(a0,row_idx0); it0; ++it0) {
					cout << endl;
					cout << "loop over cols a0" << endl;
					cout << "it0.index() = " << it0.index() << endl;
					int col_offset1 = it0.index();
					col_offset1    *= a1.innerSize();
					cout << "col_offset1 = " << col_offset1 << endl;
					double factor1 = it0.value();

					for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it1(a1,row_idx1); it1; ++it1) {
						cout << endl;
						cout << "loop over cols a1" << endl;
						int col_offset2 = col_offset1 + it1.index();
						col_offset2    *= a2.innerSize();
						double factor2  = factor1 * it1.value();

						cout << endl;
						cout << "loop over cols a2" << endl;
						for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it2(a2,row_idx2); it2; ++it2){
							cout << endl;
							cout << "summing over indices of y: " << endl;
							cout << col_offset2 + it2.index() << endl;
							retvec( row_offset2 + row_idx2 ) += factor2 * it2.value() * y( col_offset2 + it2.index() );
						}
					}

				}
			}
		}
	}
	cout << "retvec = " << endl;
	cout << retvec << endl;
	return retvec;
}

// KronProdSPMat3
// computes kronecker(a0,kronecker(a1,a2)) * y, a0, a1, a2 SPARSE matrices!
// dim(a0) = (n0,m0)
// dim(a1) = (n1,m1)
// dim(a2) = (n2,m2)
// length(y) = m0 * m1 * m2
Eigen::VectorXd KronProdSPMat3(
		Eigen::SparseMatrix<double, Eigen::RowMajor> a0,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a1,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a2,
		Eigen::VectorXd y) {

	Eigen::VectorXd retvec;
	retvec.setZero( a0.rows() * a1.rows() * a2.rows() );
	if ( y.rows() != a0.cols() * a1.cols() * a2.cols() ) {
		cout << "KronProdMat3 error: y and matrices not conformable" << endl;
	}

	//loop rows a0
	for (int row_idx0=0; row_idx0<a0.outerSize(); ++row_idx0) {
		int row_offset1 = row_idx0;
		row_offset1    *= a1.rows();

		// loop rows a1
		for (int row_idx1=0; row_idx1<a1.outerSize(); ++row_idx1) {
			int row_offset2 = row_offset1 + row_idx1;
			row_offset2    *= a2.rows();

			// loop rows a2
			for (int row_idx2=0; row_idx2<a2.outerSize(); ++row_idx2) {

				// loop cols a0 (non-zero elements only)
				for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it0(a0,row_idx0); it0; ++it0) {
					int col_offset1 = it0.index();
					col_offset1    *= a1.innerSize();
					double factor1 = it0.value();

					for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it1(a1,row_idx1); it1; ++it1) {
						int col_offset2 = col_offset1 + it1.index();
						col_offset2    *= a2.innerSize();
						double factor2  = factor1 * it1.value();

						for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it2(a2,row_idx2); it2; ++it2){
							retvec( row_offset2 + row_idx2 ) += factor2 * it2.value() * y( col_offset2 + it2.index() );
						}
					}

				}
			}
		}
	}
	return retvec;
}


// KronProdSPMat4Print
// computes kronecker(a0,kronecker(a1,kronecker(a2,a3))) * y, a0, a1, a2, a3 SPARSE matrices!
// dim(a0) = (n0,m0)
// dim(a1) = (n1,m1)
// dim(a2) = (n2,m2)
// dim(a3) = (n3,m3)
// length(y) = m0 * m1 * m2 * m3
Eigen::VectorXd KronProdSPMat4Print(
		Eigen::SparseMatrix<double, Eigen::RowMajor> a0,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a1,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a2,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a3,
		Eigen::VectorXd y) {

	Eigen::VectorXd retvec;
	retvec.setZero( a0.rows() * a1.rows() * a2.rows() * a3.rows() );
	if ( y.rows() != a0.cols() * a1.cols() * a2.cols()  * a3.cols()) {
		cout << "KronProdMat4 error: y and matrices not conformable" << endl;
	}

	//loop rows a0
	for (int row_idx0=0; row_idx0<a0.outerSize(); ++row_idx0) {
		int row_offset1 = row_idx0;
		row_offset1    *= a1.rows();
		cout << "row1 loop. row number: " << endl;
		cout << row_idx0 << endl;

		// loop rows a1
		for (int row_idx1=0; row_idx1<a1.outerSize(); ++row_idx1) {
			int row_offset2 = row_offset1 + row_idx1;
			row_offset2    *= a2.rows();
			cout << "row2 loop. row number: " << endl;
			cout << row_idx1 << endl;

			// loop rows a2
			for (int row_idx2=0; row_idx2<a2.outerSize(); ++row_idx2) {
				int row_offset3 = row_offset2 + row_idx2;
				row_offset3    *= a3.rows();
				cout << "row3 loop. row number: " << endl;
				cout << row_idx2 << endl;

				// loop rows a3
				for (int row_idx3=0; row_idx3<a3.outerSize(); ++row_idx3) {


					// loop cols a0 (non-zero elements only)
					for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it0(a0,row_idx0); it0; ++it0) {
						cout << endl;
						cout << "loop over cols a0" << endl;
						cout << "it0.index() = " << it0.index() << endl;
						int col_offset1 = it0.index();
						col_offset1    *= a1.innerSize();
						cout << "col_offset1 = " << col_offset1 << endl;
						double factor1 = it0.value();

						// loop cols a1
						for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it1(a1,row_idx1); it1; ++it1) {
							cout << endl;
							cout << "loop over cols a1" << endl;
							int col_offset2 = col_offset1 + it1.index();
							col_offset2    *= a2.innerSize();
							double factor2  = factor1 * it1.value();

							//loop cols a2
							for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it2(a2,row_idx2); it2; ++it2) {
								cout << endl;
								cout << "loop over cols a2" << endl;
								int col_offset3 = col_offset2 + it2.index();
								col_offset3    *= a3.innerSize();
								double factor3  = factor2 * it2.value();

								for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it3(a3,row_idx3); it3; ++it3){
									cout << endl;
									cout << "summing over indices of y: " << endl;
									cout << col_offset3 + it3.index() << endl;
									retvec( row_offset3 + row_idx3 ) += factor3 * it3.value() * y( col_offset3 + it3.index() );
								}
							}
						}

					}
				}
			}
		}
	}
	cout << "retvec = " << endl;
	cout << retvec << endl;
	return retvec;
}

// KronProdSPMat4Print
// computes kronecker(a0,kronecker(a1,kronecker(a2,a3))) * y, a0, a1, a2, a3 SPARSE matrices!
// dim(a0) = (n0,m0)
// dim(a1) = (n1,m1)
// dim(a2) = (n2,m2)
// dim(a3) = (n3,m3)
// length(y) = m0 * m1 * m2 * m3
Eigen::VectorXd KronProdSPMat4(
		Eigen::SparseMatrix<double, Eigen::RowMajor> a0,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a1,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a2,
		Eigen::SparseMatrix<double, Eigen::RowMajor> a3,
		Eigen::VectorXd y) {

	Eigen::VectorXd retvec;
	retvec.setZero( a0.rows() * a1.rows() * a2.rows() * a3.rows() );
	if ( y.rows() != a0.cols() * a1.cols() * a2.cols()  * a3.cols()) {
		cout << "KronProdMat4 error: y and matrices not conformable" << endl;
	}

	//loop rows a0
	for (int row_idx0=0; row_idx0<a0.outerSize(); ++row_idx0) {
		int row_offset1 = row_idx0;
		row_offset1    *= a1.rows();

		// loop rows a1
		for (int row_idx1=0; row_idx1<a1.outerSize(); ++row_idx1) {
			int row_offset2 = row_offset1 + row_idx1;
			row_offset2    *= a2.rows();

			// loop rows a2
			for (int row_idx2=0; row_idx2<a2.outerSize(); ++row_idx2) {
				int row_offset3 = row_offset2 + row_idx2;
				row_offset3    *= a3.rows();

				// loop rows a3
				for (int row_idx3=0; row_idx3<a3.outerSize(); ++row_idx3) {


					// loop cols a0 (non-zero elements only)
					for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it0(a0,row_idx0); it0; ++it0) {
						int col_offset1 = it0.index();
						col_offset1    *= a1.innerSize();
						double factor1 = it0.value();

						// loop cols a1
						for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it1(a1,row_idx1); it1; ++it1) {
							int col_offset2 = col_offset1 + it1.index();
							col_offset2    *= a2.innerSize();
							double factor2  = factor1 * it1.value();

							//loop cols a2
							for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it2(a2,row_idx2); it2; ++it2) {
								int col_offset3 = col_offset2 + it2.index();
								col_offset3    *= a3.innerSize();
								double factor3  = factor2 * it2.value();

								for (Eigen::SparseMatrix<double,RowMajor>::InnerIterator it3(a3,row_idx3); it3; ++it3){
									retvec( row_offset3 + row_idx3 ) += factor3 * it3.value() * y( col_offset3 + it3.index() );
								}
							}
						}

					}
				}
			}
		}
	}
	return retvec;
}


/////////////////////////////////////////
// Helper functions: convert dense to sparse
/////////////////////////////////////////

// converter functions: make sparse vector from dense
Eigen::SparseVector<double> Dense2Sparse(
    Eigen::VectorXd vIn )
{
    Eigen::SparseVector<double> vOut( vIn.rows() );
    for (int i=0;i<vIn.rows();i++) {
        if (vIn( i ) != 0.0 ) {
            vOut.insert( i ) = vIn( i );
        }
    }
    return vOut;
}

// converter functions: make sparse matrix from dense
Eigen::SparseMatrix<double> FillMatRow( Eigen::MatrixXd Rmat )
{
	// fills SparseMatrix retmat with non zero elts of Rmat
	// ROW MAJOR
    // define the return matrix
    Eigen::SparseMatrix<double> retmat( Rmat.rows(), Rmat.cols() );

	// loop over each row of Rmat filling in those elements != 0
    for (int irow=0; irow<Rmat.rows(); irow++) {
	// call dense2sparse!!!
	Eigen::SparseVector<double> rowvec = Dense2Sparse( Rmat.row( irow ) );

        // loop over non-zero elements of rowvec and copy these to retmat
        for (Eigen::SparseVector<double>::InnerIterator col_it(rowvec); col_it; ++col_it) {
            retmat.insert( irow, col_it.index() ) = col_it.value();
        }
    }
	// finish filling: finalize
    retmat.finalize();
    return retmat;
}

/////////////////////////////////////////
// Test functions calling above implementations
/////////////////////////////////////////

int KronProdTestLoop() {
	cout << "KronProdTestLoop. setup some matrices for KronProdSPMat2LoopTest" << endl;

	MatrixXd aa(5,4);
	aa << 1,2,3,4,
		  5,0,0,6,
		  0,0,0,7,
		  8,0,9,0,
		  0,0,10,0;
	SparseMatrix<double,RowMajor> spaa( aa.rows(), aa.cols() );
	MatrixXd bb(2,3);
	bb << 1,2,0,
		  0,0,3;
	SparseMatrix<double,RowMajor> spbb( bb.rows(), bb.cols() );

	spaa = FillMatRow( aa );	// fill spaa with non-zeros of aa in row major fashion.
	spbb = FillMatRow( bb );	// fill spaa with non-zeros of aa in row major fashion.

	cout << "aa = " << endl;
	cout << aa << endl;
	cout << "bb = " << endl;
	cout << bb << endl;

	cout << "spaa = " << endl;
	cout << spaa << endl;
	cout << "spbb = " << endl;
	cout << spbb << endl;


	VectorXd y;
	y.setLinSpaced( aa.cols() * bb.cols(), 1, aa.cols() * bb.cols() );
	cout << "y = " << endl;
	cout << y << endl;

	VectorXd result( KronProdSPMat2looptest( spaa, spbb, y ) );
	return 0;
}

int KronProdSPMat2Test(
		int which) {
	cout << "KronProdSPMat2Test. setup some matrices" << endl;

	MatrixXd aa(5,4);
	aa << 1,2,3,4,
		  5,0,0,6,
		  0,0,0,7,
		  8,0,9,0,
		  0,0,10,0;
	SparseMatrix<double,RowMajor> spaa( aa.rows(), aa.cols() );
	MatrixXd bb(2,3);
	bb << 1,2,0,
		  0,0,3;
	SparseMatrix<double,RowMajor> spbb( bb.rows(), bb.cols() );

	spaa = FillMatRow( aa );	// fill spaa with non-zeros of aa in row major fashion.
	spbb = FillMatRow( bb );	// fill spaa with non-zeros of aa in row major fashion.

	cout << "aa = " << endl;
	cout << aa << endl;
	cout << "bb = " << endl;
	cout << bb << endl;

	cout << "spaa = " << endl;
	cout << spaa << endl;
	cout << "spbb = " << endl;
	cout << spbb << endl;


	VectorXd y;
	y.setLinSpaced( aa.cols() * bb.cols(), 1, aa.cols() * bb.cols() );

	cout << endl;
	cout << "y = " << endl;
	cout << y << endl;
	cout << endl;

	if (which == 0){
		VectorXd result( KronProdSPMat2Print( spaa, spbb, y ) );
	} else if (which == 1){
		VectorXd result(KronProdSPMat2( spaa, spbb, y ) );
	}
	return 0;
}

int KronProdSPMat3Test(
		int which) {
	cout << "KronProdSPMat3Test. setup some matrices" << endl;

	MatrixXd aa(5,4);
	aa << 1,2,3,4,
		  5,0,0,6,
		  0,0,0,7,
		  8,0,9,0,
		  0,0,10,0;
	SparseMatrix<double,RowMajor> spaa( aa.rows(), aa.cols() );
	MatrixXd bb(2,3);
	bb << 1,2,0,
		  0,0,3;
	SparseMatrix<double,RowMajor> spbb( bb.rows(), bb.cols() );
	MatrixXd cc(3,2);
		cc << 0,1,
			  0,3,
			  4,0;
		SparseMatrix<double,RowMajor> spcc( cc.rows(), cc.cols() );


	spaa = FillMatRow( aa );	// fill spaa with non-zeros of aa in row major fashion.
	spbb = FillMatRow( bb );	// fill spaa with non-zeros of aa in row major fashion.
	spcc = FillMatRow( cc );	// fill spaa with non-zeros of aa in row major fashion.

	cout << "aa = " << endl;
	cout << aa << endl;
	cout << "bb = " << endl;
	cout << bb << endl;
	cout << "cc = " << endl;
	cout << cc << endl;

	cout << endl;
	cout << "spaa = " << endl;
	cout << spaa << endl;
	cout << "spbb = " << endl;
	cout << spbb << endl;
	cout << "spcc = " << endl;
	cout << spcc << endl;


	VectorXd y;
	y.setLinSpaced( aa.cols() * bb.cols() * cc.cols(), 1, aa.cols() * bb.cols() * cc.cols() );

	cout << endl;
	cout << "y = " << endl;
	cout << y << endl;
	cout << endl;

	if (which == 0){
		VectorXd result( KronProdSPMat3Print( spaa, spbb, spcc, y ) );
	} else if (which == 1) {
		VectorXd result( KronProdSPMat3( spaa, spbb, spcc, y ) );
	}
	return 0;
}

int KronProdSPMat4Test(
		int which) {
	cout << "KronProdSPMat4Test. setup some matrices" << endl;

	MatrixXd aa(5,4);
	aa << 1,2,3,4,
		  5,0,0,6,
		  0,0,0,7,
		  8,0,9,0,
		  0,0,10,0;
	SparseMatrix<double,RowMajor> spaa( aa.rows(), aa.cols() );
	MatrixXd bb(2,3);
	bb << 1,2,0,
		  0,0,3;
	SparseMatrix<double,RowMajor> spbb( bb.rows(), bb.cols() );
	MatrixXd cc(3,2);
		cc << 0,1,
			  0,3,
			  4,0;
	SparseMatrix<double,RowMajor> spcc( cc.rows(), cc.cols() );
	MatrixXd dd(2,4);
		dd << 0,1,2,0,
			  2,0,0,1;
	SparseMatrix<double,RowMajor> spdd( dd.rows(), dd.cols() );



	spaa = FillMatRow( aa );	// fill spaa with non-zeros of aa in row major fashion.
	spbb = FillMatRow( bb );
	spcc = FillMatRow( cc );
	spdd = FillMatRow( dd );

	cout << "aa = " << endl;
	cout << aa << endl;
	cout << "bb = " << endl;
	cout << bb << endl;
	cout << "cc = " << endl;
	cout << cc << endl;
	cout << "dd = " << endl;
	cout << dd << endl;

	cout << endl;
	cout << "spaa = " << endl;
	cout << spaa << endl;
	cout << "spbb = " << endl;
	cout << spbb << endl;
	cout << "spcc = " << endl;
	cout << spcc << endl;
	cout << "spdd = " << endl;
	cout << spdd << endl;


	VectorXd y;
	y.setLinSpaced( aa.cols() * bb.cols() * cc.cols() * dd.cols(), 1, aa.cols() * bb.cols() * cc.cols() * dd.cols() );

	cout << endl;
	cout << "y = " << endl;
	cout << y << endl;
	cout << endl;

	if (which == 0){
		VectorXd result( KronProdSPMat4Print( spaa, spbb, spcc, spdd, y ) );
	}
	//} else if (which == 1){
	//	VectorXd result(KronProdSPMat3( spaa, spbb, y ) );
	//}
	return 0;
}




void MaxFindTest()
{
  MatrixXf mat(2,4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;

  MatrixXf::Index   maxIndex;

  float maxVal = mat.rowwise().sum().maxCoeff(&maxIndex);

    std::cout << "Maxima at positions " << endl;
    std::cout << maxIndex << std::endl;

    std::cout << "maxVal " << maxVal << endl;
  }


void KronDenseTest()
{
  Eigen::MatrixXd a0(3,3);
  a0 << 1,2,3,
		4,5,6,
		7,8,9;

  Eigen::MatrixXd a1(2,2);
  a1 << 5,4,
		2,1;

  Eigen::VectorXd y(6);
  y << 10,11,12,13,14,15;

  cout << "a0 = " << a0 << endl;
  cout << "a1 = " << a1 << endl;
  cout << "y = " << y << endl;

  Eigen::VectorXd result = kronproddense(a1,a0,y);

  cout << "result = " << result << endl;
}





#endif /* TESTHEADER_H_ */
