#include <math.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <numeric>
#include "pthread.h"
#include <string>
#include <fstream>

constexpr const std::size_t NumThreads = 8;
long long total = 0 ; 

constexpr const double dt = 0.01;
    
constexpr const double gNa = 120, eNa = 115, gK = 36, eK = -12, gL = 0.3, eL = 10.6;

const double Vmin = -80, Vmax = 30, Vstep = 4 , hstep = 0.05, nstep = 0.05, mstep = 0.05 ;

struct state {
  double V, m, h, n;
  void print (){
    std::cout << " V = " << V << " m = " << m << " h = " << h << " n= " << n << std::endl;
  }
};

std::vector<std::vector<double>> Vvals(NumThreads);
std::vector<std::pair<state, double>> data [NumThreads];

struct threadArgs {
  std::vector<double> &Vvals;
  std::vector<std::pair<state, double>> &data;
};

double alphaM (double V) {
  return ((2.5 - 0.1*(V+65)) / (std::exp(2.5 - 0.1*(V+65)) - 1));
}

double alphaN(double V) {
  return ((0.1 - 0.01*(V+65)) / (std::exp(1 - 0.1*(V+65)) - 1));
}

double alphaH(double V) {
  return 0.07 * (std::exp(-(V+65)/20));
}
    
double betaM(double V) {
  return 4.0*(std::exp(-(V+65)/18));
}

double betaN(double V) {
  return 0.125*(std::exp(-(V+65)/80));
}

double betaH(double V) {
  return 1.0/(std::exp(3.0-0.1*(V+65))+1);
}


void step (double I, state &S, double t) {

    // if (t*1000-int(t*1000) != 0):
    //     print("enter values upto 4 decimal places")
  double V = S.V;
  double m = S.m;
  double h = S.h;
  double n = S.n;

  int looptimes = (int)(t / dt);
  // std::cout << "loops " << looptimes << std::endl;
  
  for (int i = 0; i < looptimes; i ++){
    double V1 = V + dt*((gNa * m*m*m * h * (eNa - (V+65))) + (gK * n*n*n*n * (eK - (V+65))) + (gL * (eL - (V + 65))) + I);
    // std::cout << " V " << V << " V1 " << V1 << " I " << I << " gL " << gL << " eL " << eL << std::endl;
    double m1 = m + dt*(alphaM(V)*(1-m) - betaM(V)*m);
    double h1 = h + dt*(alphaH(V)*(1-h) - betaH(V)*h);
    double n1 = n + dt*(alphaN(V)*(1-n) - betaN(V)*n);
    V = V1;
    m = m1;
    h = h1;
    n = n1;
  }
  S.V = V;
  S.m = m;
  S.h = h;
  S.n = n;
}

bool doesOscilate (state initState, double I){
  for (int i = 0; i < 13333; i ++){
    // initState.print();
    step(I, initState, 0.01);
  }
  // throw "error";
  int Vmin = initState.V;
  int Vmax = initState.V;
  for (int i = 0; i < 6666; i ++){
    step(I, initState, 0.01);
    if (initState.V > Vmax){
      Vmax = initState.V;
    }
    if (initState.V < Vmin){
      Vmin = initState.V;
    }
  }
  // std::cout << Vmax - Vmin << std::endl;
  return (Vmax - Vmin >= 90);
}

double findI (state initState, double Imin, double Imax){
  std::vector<double> currs;
  for (double i = Imin; i < Imax; i += 0.01){
    currs.push_back(i);
  }
  int startIdx = 0, endIdx = currs.size();
  while (startIdx < endIdx-1){
    int mid = (startIdx + endIdx)/2;
    if (doesOscilate(initState, currs[mid])){
      endIdx = mid;
    }
    else {
      startIdx = mid;
    }
  }
  return currs[startIdx];
}

void* computePerThread (void* args){
  threadArgs x = *((threadArgs*) args);
  for (double V : x.Vvals){
    for (double h = 0; h <= 1; h += hstep){
      for (double n = 0; n <= 1; n += nstep){
  	for (double m = 0; m <= 1; m += mstep){
  	  state S = {V, m, h, n};
  	  x.data.push_back({S, findI(S, 4, 11)});
      total ++ ; 
      if( total% 1000 == 0 ){
        std::cout << total << "\n" ; 
      } 
  	  // std::cout << "V = " << S.V << " h = " << S.h << " n = " << S.n << " m = " << S.m << " I = " << x.data[x.data.size()-1].second << std::endl;
  	}
      }
    }
  }
  pthread_exit(NULL);
}

int main (){
  std::vector<pthread_t> allThreads(NumThreads);
  std::vector<threadArgs> args;
  for (double V = Vmin; V <= Vmax; V += Vstep){
    int tNum = (int)((NumThreads*(V-Vmin-1))/(Vmax-Vmin));
    Vvals[tNum].push_back(V);
  }
  for (std::size_t threadnum = 0; threadnum < NumThreads; threadnum++){
    args.push_back({Vvals[threadnum], data[threadnum]});
  }
  for (std::size_t threadnum = 0; threadnum < NumThreads; threadnum ++){
    int t = pthread_create(&allThreads[threadnum], NULL, computePerThread, (void*)(&args[threadnum]));
  }
  for (unsigned int thread = 0; thread < NumThreads; thread ++){
    int t = pthread_join(allThreads[thread], NULL);
  }

  std::ofstream fw("data.txt" ,  std::ofstream::out);
  fw << "V m n h I\n" ; 
  for( int i = 0 ; i < NumThreads ; i ++ ){
    for( auto ele : data[i]){
      fw << ele.first.V << " " << ele.first.m << " " << ele.first.n << " " << ele.first.h<< " " << ele.second  << "\n" ;  
    }
  }
}
