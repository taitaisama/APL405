#include <math.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <numeric>

constexpr const double dt = 0.01;
    
constexpr const double gNa = 120, eNa = 115, gK = 36, eK = -12, gL = 0.3, eL = 10.6;

struct state {
  double V, m, h, n;
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
  
  for (int i = 0; i < looptimes; i ++){
    double V1 = V + dt*((gNa * m*m*m * h * (eNa - (V+65))) + (gK * n*n*n*n * (eK - (V+65))) + (gL * (eL - (V + 65))) + I);
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

int main (){
  auto start = std::chrono::system_clock::now();
  std::vector<state> data;
  std::vector<double> times;
  
  for (double V = -80; V < 30; V ++){
    for (double h = 0; h <= 1; h += 0.04){
      for (double n = 0; n <= 1; n += 0.04){
	for (double m = 0; m <= 1; m += 0.04){
	  double Iguess = 
	}
      }
    }
  }
  state S = {0, 0, 0, 0};
  for (int i = 0; i < 20000; i ++){
    data.push_back(S);
    // times.push_back(i);
    step(100, S, 0.01);
  }
  auto end = std::chrono::system_clock::now();
  std::cout << std::fixed << std::setprecision(9) << std::left;
  std::chrono::duration<double> diff = end-start;
  std::cout << std::setw(9) << diff.count() << std::endl;
}
