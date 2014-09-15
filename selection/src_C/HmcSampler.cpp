/* 
 * File:   HmcSampler.cpp
 * Author: aripakman
 * 
 * Created on July 4, 2012, 10:44 AM
 */

#define _USE_MATH_DEFINES   // for the constant M_PI
//#include <cstdlib>
#include <cmath>
#include <tr1/random>
#include <Eigen/Dense>
#include <magnet/math/quartic.hpp>

#include "HmcSampler.h"

using namespace std;
using namespace std::tr1;
using namespace magnet::math;


const double HmcSampler::min_t = 0.00001;

HmcSampler::HmcSampler(const int & d, const int & seed ) {
dim=d;    
//eng1.seed(static_cast<unsigned int >(time(NULL)));
eng1.seed(seed);
ud= uniform_real<>(0,M_PI);
}   



void HmcSampler::setInitialValue(const VectorXd & initial_value){                 
   // double check =_verifyConstraints( initial_value );
  /*  if (check <0) {
        cout << "Initial condition out of constraint!" << endl;
        exit(1);
    } else */
            lastSample = initial_value;
    }



MatrixXd HmcSampler::sampleNext(bool returnTrace ) {  

    MatrixXd tracePoints = MatrixXd(dim,0);   //this matrix will only be filled if(returnTrace)

    double T = ud(eng1);        // sample how much time T to move
    VectorXd b = lastSample;
    VectorXd a = VectorXd(dim);   // initial velocity 

    while(2){
        
        double velsign =0; 
        for (int i =0; i<dim; i++)     // Sample new initial velocity  
        {  a(i)=  nd(eng1);            }

        double tt=T;   // tt is the time left to move 
        double t1;
        double t2;
        int cn1,cn2; //constraint number


        bool first_bounce = true;    // for the first move, we do not fear that a small t1 or t2 is due to being in the boundary from the previous bounce.
        while (1){            
            
            t1=0; 
            if (!linearConstraints.empty())
                    _getNextLinearHitTime(a,b,t1,cn1); 
            
            t2=0;
            if (!quadraticConstraints.empty() )
                    {   _getNextQuadraticHitTime(a,b,t2,cn2, first_bounce);                        
                        first_bounce=false;}

            double t =t1;     // how much time to move. if t==0, move tt 
            bool linear_hit = true;
            if (t2>0 && (t1==0 || t2<t1 )){
                t=t2;
                linear_hit = false;
            }

            if (t==0 || tt < t){                 // if no wall to be hit (t==0) or not enough time left to hit the wall (tt<t2) 
                break;
            }
            else{            
                if (returnTrace){
                    _updateTrace(a,b,t,tracePoints);
                }

                tt = tt-t;
                VectorXd new_b   = sin(t)*a + cos(t)*b;   // hit location 
                VectorXd hit_vel = cos(t)*a - sin(t)*b;   // hit velocity
                b = new_b;            
           

                // reflect the velocity and verify that it points in the right direction
    
                if (!linear_hit){
                    QuadraticConstraint qc = quadraticConstraints[cn2];
                    VectorXd nabla = 2* ((qc.A)*b) + qc.B;            
                    double alpha = (nabla.dot(hit_vel))/(nabla.dot(nabla));
                    a = hit_vel - 2*alpha*nabla;                  // reflected velocity    
                    velsign = a.dot(nabla);
                }
                else {                
                    LinearConstraint ql = linearConstraints[cn1];
                    double f2 = ((ql.f).dot((ql.f)));
                    double alpha = ((ql.f).dot(hit_vel))/f2;
                    a = hit_vel - 2*alpha*(ql.f);                  // reflected velocity                                         
                    velsign = a.dot((ql.f));
                }
                if (velsign <0 ) break ;    //get out of while(1). resample the velocity and start again. this occurs rarely, due to numerical instabilities
            }

        } //while(1)

        if (velsign<0) {/* cout << "wrong velocity " << endl; */  continue;}    // go to beginning of while(2)
        
        VectorXd bb =  sin(tt)*a + cos(tt)*b;             //make last move of time tt without hitting walls  

        double check = _verifyConstraints( bb);
        if (check >= 0){     //verify that we don't violate the constraints due to a numerical instability
            lastSample = bb;      
            if (returnTrace){
                _updateTrace(a,b,tt,tracePoints);
                return tracePoints.transpose();          

            } else return lastSample.transpose();
        }
       // at this point we have check<0, so we violated constraints: resample. 

    } // while(2)
}



void HmcSampler::addLinearConstraint(const VectorXd & f, const double & g){
    LinearConstraint newConstraint;
    newConstraint.f =f;
    newConstraint.g =g;
    linearConstraints.push_back(newConstraint);
}
void HmcSampler::addQuadraticConstraint(const MatrixXd & A, const VectorXd & B, const double & C){
    QuadraticConstraint newConstraint;
    newConstraint.A =A;
    newConstraint.B =B;
    newConstraint.C =C;
    quadraticConstraints.push_back(newConstraint);
    
}

void HmcSampler::_getNextLinearHitTime(const VectorXd & a, const VectorXd & b, double & hit_time, int & cn ){
    hit_time=0;
    
    for (int i=0; i != linearConstraints.size(); i++ ){
        LinearConstraint lc = linearConstraints[i];
        double fa = (lc.f).dot(a);
        double fb = (lc.f).dot(b);
        double u = sqrt(fa*fa + fb*fb);
        if (u>lc.g && u>-lc.g){
                double phi =atan2(-fa,fb);      //     -pi < phi < pi
                double t1 = acos(-lc.g/u)-phi;  //     -pi < t1 < 2*pi
                
                
                if (t1<0) t1 += 2*M_PI;                //  0 < t1 < 2*pi                  
                if (abs(t1) < min_t ) t1=0;    
                else if (abs(t1-2*M_PI) < min_t ) t1=0;                                            
                
                
                double t2 = -t1-2*phi;             //  -4*pi < t2 < 3*pi
                if (t2<0) t2 += 2*M_PI;                 //-2*pi < t2 < 2*pi
                if (t2<0) t2 += 2*M_PI;                 //0 < t2 < 2*pi
                
                if (abs(t2) < min_t ) t2=0;    
                else if (abs(t2-2*M_PI) < min_t ) t2=0;                            
                
                
                double t=t1;                
                if (t1==0) t = t2;
                else if (t2==0) t = t1;
                else t=(t1<t2?t1:t2);
                
                if  (t> min_t  && (hit_time == 0 || t < hit_time)){
                    hit_time=t;
                    cn =i;                    
                }            
        }       
    }        
}

void HmcSampler::_getNextQuadraticHitTime(const VectorXd & a, const VectorXd & b, double & hit_time, int & cn , const bool first_bounce ){
        hit_time=0;
  
        double mint;
        if (first_bounce) {mint=0;}
        else {mint=min_t;}
    
    for (int i=0; i != quadraticConstraints.size(); i++ ){
        
        QuadraticConstraint qc = quadraticConstraints[i];
        double q1= - ((a.transpose())*(qc.A))*a;
        q1 = q1 + ((b.transpose())*(qc.A))*b;
        double q2= (qc.B).dot(b);
        double q3= qc.C + a.transpose()*(qc.A)*a;
        double q4= 2*b.transpose()*(qc.A)*a;
        double q5= (qc.B).dot(a);

        double r4 = q1*q1 + q4*q4;
        double r3 = 2*q1*q2 + 2*q4*q5;
        double r2 = q2*q2 + 2*q1*q3 + q5*q5 -q4*q4;
        double r1 = 2*q2*q3 - 2*q4*q5;
        double r0=  q3*q3 - q5*q5;

        double roots[]={0,0,0,0};
        double aa = r3/r4;
        double bb = r2/r4;
        double cc = r1/r4;
        double dd = r0/r4;

        //Solve quartics of the form x^4 + aa x^3 + bb x^2 + cc x + dd ==0
        int sols = quarticSolve(aa, bb, cc, dd, roots[0], roots[1],  roots[2],  roots[3]);
        for (int j=0; j<sols; j++){
            double r = roots[j];
            if (abs(r) <=1 ){               
                double l1 = q1*r*r + q2*r + q3;
                double l2 = -sqrt(1-r*r)*(q4*r + q5); 
                if (l1/l2 > 0){
                    double t = acos(r);
                    if (   t> mint      && (hit_time == 0 || t < hit_time)){
                       hit_time=t;
                       cn=i;                                          
                    }                    
                }
            }            
        }                
    }    
    
    
}




double HmcSampler::_verifyConstraints(const VectorXd & b){
    double r =0;
    
    for (int i=0; i != quadraticConstraints.size(); i++ ){       
        QuadraticConstraint qc = quadraticConstraints[i];
        double check = ((b.transpose())*(qc.A))*b + (qc.B).dot(b) + qc.C;
        if (i==0 || check < r) {
            r = check;
        }
    }

    for (int i=0; i != linearConstraints.size(); i++ ){       
    LinearConstraint lc = linearConstraints[i];
    double check = (lc.f).dot(b) + lc.g;
    if (i==0 || check < r) {
        r = check;
    }
    }
    
    
    return r;
}

void HmcSampler::_updateTrace( VectorXd const & a,  VectorXd const & b, double const & t, MatrixXd & tracePoints){
    double const stepsize = .01;
    int steps = t/stepsize;
    
    int c = tracePoints.cols();
    tracePoints.conservativeResize(NoChange, c+steps+1);
    for (int i=0; i<steps; i++){
        VectorXd bb= sin(i*stepsize)*a + cos(i*stepsize)*b;
//      cout << bb.transpose() << endl;
        tracePoints.col(c+i) = bb;                    
    }
        VectorXd bb= sin(t)*a + cos(t)*b;
        tracePoints.col(c+steps) = bb;
}
