#include <Eigen/Dense>
#include <vector>
#include "HmcSampler.h"

#include "preparation_Eig_Vect.h"


#include <fstream>

using namespace std;
using namespace Eigen;

void samples(
                int n,
                int dim,
                int seed,
                double *initial, 
                int numlin,
                int numquad,
                double *lin,
                double *quad, 
                double *quad_lin,
                double *offset_lin,
                double *offset_quad,
                double *samples_Carray
		 ){

  
  const Map<VectorXd> initial_value(initial, dim);



  ofstream logfile;
  logfile.open ("logfile.txt");
  

  HmcSampler hmc1(dim, seed);
  if (numlin >0){		
    const Map<MatrixXd> F(lin, numlin, dim);
    const Map<VectorXd> g(offset_lin, numlin);

    for(int i=0; i<numlin; i++){
      hmc1.addLinearConstraint(F.row(i),g(i));
    }
  }

  if (numquad >0){

    for(int i=0; i<numquad; i++){
      double *indice = &quad[i*dim*dim];
      const Map<MatrixXd> A_Map(indice, dim, dim);


for(int k=0; k<dim; k++){
for(int l=0; l<dim; l++){
logfile << A_Map(k, l);
}
logfile << endl;
}
logfile << endl;

      MatrixXd A(A_Map);
      const Map<VectorXd> B_Map(&quad_lin[i*dim], dim);
      VectorXd B(B_Map);
      double C = offset_quad[i];  
      hmc1.addQuadraticConstraint(A,B,C);
    }

  }

  hmc1.setInitialValue(initial_value);
  
  MatrixXd samples(n,dim);
  
  for (int i=0; i<n; i++){     
      samples.row(i) = hmc1.sampleNext();  
  }

//static double samples_Carray [n][dim];

  double* result = samples.data();

  for(int k=0; k<n; k++){
    for(int l=0; l<dim;l++){
      samples_Carray[k*dim + l] = result[k*dim + l];
    }
  }


for(int k=0; k< n; k++){
for(int l=0; l<dim; l++){
logfile << result[k*dim + l];
}
logfile << endl;
}

logfile << endl;

for(int k=0; k< n; k++){
for(int l=0; l<dim; l++){
logfile << samples_Carray[k*dim + l];
}
logfile << endl;
}



  logfile.close();

//return samples_Carray;

}
