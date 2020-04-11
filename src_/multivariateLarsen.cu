#include "larsen-inl.h"


double flopCounter(int M, int N, int numModels, int *hNVars) {
  double flop = 0;
  // r = y - mu
  flop += (double) M * (double) numModels;
  // c = X' * r
  flop += 2.0 * (double) M * (double) N * (double) numModels;
  // abs(c)
  flop += (double) N * (double) numModels;
  for (int i = 0; i < numModels; i++) {
    // G = X(:, A)' * X(:, A)
    flop += 2.0 * (double) hNVars[i] * (double) M * (double) hNVars[i];
    // b_OLS = G\(X(:,A)'*y) + rand
    flop += 2.0 * (double) M * (double) hNVars[i] + 10.0 * (double) hNVars[i] * (double) hNVars[i];
    // Inverse ops for G
    flop += (2.0 / 3.0) * (double) hNVars[i] * (double) hNVars[i] * (double) hNVars[i];
    // d = X(: , A) * b_OLS - mu
    flop += 2.0 * (double) M * (double) hNVars[i] + (double) M;
    // gamma_tilde
    flop += 2.0 * (double) hNVars[i];
    // b update + l1 check
    flop += 5.0 * (double) hNVars[i];
    // norm1
    flop += 2.0 * (double) hNVars[i];
  }
  // cd = X'*d
  flop += 2.0 * (double) M * (double) N * (double) numModels;
  // gamma
  flop += 6.0 * (double) N * (double) numModels;
  // mu update
  flop += 2.0 * (double) M * (double) numModels;
  // norm2
  flop += 3.0 * (double) M * (double) numModels;
  // norm2 sqrt, step, sb, err
  flop += 4.0 * (double) numModels;
  // G
  flop += 3.0 * (double) M * (double) numModels;
  return flop;
}

//
int main(int argc, char *argv[]) {
  if (argc < 11) {
    printf("Insufficient parameters, required 10!\n");
    printf("Use: flatMriPath, numModels, numStreams, max l1, min l2, min g, max vars, max steps, proc prec, corr prec\n");
    printf("Input 0 for a parameter to use it's default value!\n");
    return 0;
  }
  if (argv[9][0] == 'f' && argv[10][0] == 'f') {
    larsen<float, float>(argc, argv);
  }
  // else if (argv[9][0] == 'f' && argv[10][0] == 'd') {
  // 	larsen<float, double>(argc, argv);
  // }
  // else if (argv[9][0] == 'd' && argv[10][0] == 'd') {
  // 	larsen<double, double>(argc, argv);
  // }
  else {
    printf("Invalid precisions\n");
  }
  return 0;
}
