#ifndef LARSEN_INL_H
#define LARSEN_INL_H

#include "headers.h"
#include "blas.h"
#include "kernels-inl.h"
#include "utilities-inl.h"

using namespace std;
using namespace thrust::placeholders;

// Perform the larsen iterations.
template<typename ProcPrec, typename CorrPrec>
void larsen(int argc, char *argv[]) {
  // Reading flattened mri image.
  Variable<ProcPrec, CorrPrec> X, Y;
  int M, N;
  std::tie(M, N) = ReadFlatMri<ProcPrec, CorrPrec>(argv[1], X, Y);
  printf("Read FMRI Data X of shape: (%d,%d)\n", M, N);
  printf("Read FMRI Data Y of shape: (%d,%d)\n", M, N);

  // Number of models to solve in ||l.
  int numModels = atoi(argv[2]);
  numModels = (numModels == 0)? 512: numModels;
  int totalModels = N;
  printf("Total number of models: %d\n", totalModels);
  printf("Number of models in ||l: %d\n", numModels);

  int numStreams = atoi(argv[3]);
  numStreams = (numStreams == 0)? 8: numStreams;
  numStreams = pow(2, int(log(numStreams) / log(2)));
  printf("Number of streams: %d\n", numStreams);

  ProcPrec l1 = atof(argv[4]);
  l1 = (l1 == 0)? 1000: l1;
  printf("Max L1: %f\n", l1);

  ProcPrec l2 = atof(argv[5]);
  printf("Min L2: %f\n", l2);

  ProcPrec g = atof(argv[6]);
  g = (g == 0)? 0.1: g;
  printf("Lambda: %f\n", g);

  int maxVariables = atoi(argv[7]);
  maxVariables = (maxVariables == 0)? N - 1: maxVariables;
  maxVariables = min(min(M, N - 1), maxVariables);
  printf("Max Variables: %d\n", maxVariables);

  int maxSteps = atoi(argv[8]);
  maxSteps = (maxSteps == 0)? 8 * maxVariables: maxSteps;
  maxSteps = min(8 * maxVariables, maxSteps);
  printf("Max Steps: %d\n", maxSteps);

  // Declare all lars variables.
  Variable<int, int> nVars(numModels), eVars(numModels), step(numModels), lasso(numModels), done(numModels), cidx(numModels), act(numModels), dropidx(numModels);
  Variable<int, int> info(numModels), lVars(numModels * M);

  Variable<ProcPrec, CorrPrec> cmax(numModels), a1(numModels), a2(numModels), lambda(numModels), gamma(numModels), gamma_tilde(numModels);
  Variable<ProcPrec, CorrPrec> y(numModels * M), mu(numModels * M), r(numModels * M), betaOls(numModels * M), d(numModels * M);
  Variable<ProcPrec, CorrPrec> beta(numModels * N), c(numModels * N), cd(numModels * N), beta_prev(numModels * N);
  Variable<ProcPrec, CorrPrec> rander1(numModels * N), rander2(numModels * N), randnrm(numModels);
  Variable<ProcPrec, CorrPrec> XA[numModels], XA1[numModels], G[numModels], I[numModels];
  Variable<ProcPrec *, CorrPrec *> pXA(numModels), pG(numModels), pI(numModels);
  CorrPrec alp = 1, bet = 0;

  for (int i = 0; i < numModels; i++) {
    XA[i].AllocProc(M * M);
    XA1[i].AllocProc(M * M);
    G[i].AllocProc(M * M);
    I[i].AllocProc(M * M);
    pXA.hVar[i] = XA[i].GetRawDevPtr();
    pG.hVar[i] = G[i].GetRawDevPtr();
    pI.hVar[i] = I[i].GetRawDevPtr();
  }
  pXA.SyncDev();
  pG.SyncDev();
  pI.SyncDev();

  // Random inits.

  // for (int i = 0; i < numModels; i++) {
  // 	init_var<precision>(XA[i], M * M);
  // 	init_var<precision>(XA1[i], M * M);
  // 	init_var<precision>(G[i], M * M);
  // 	init_var<precision>(I[i], M * M);

  // 	init_var<corr_precision>(corr_XA[i], M * M);
  // 	init_var<corr_precision>(corr_G[i], M * M);
  // 	init_var<corr_precision>(corr_I[i], M * M);
  // }

  // cudaMalloc(&dXA, numModels * sizeof(precision *));
  // cudaMemcpy(dXA, XA, numModels * sizeof(precision *), cudaMemcpyHostToDevice);
  // cudaMalloc(&dG, numModels * sizeof(precision *));
  // cudaMemcpy(dG, G, numModels * sizeof(precision *), cudaMemcpyHostToDevice);
  // cudaMalloc(&dI, numModels * sizeof(precision *));
  // cudaMemcpy(dI, I, numModels * sizeof(precision *), cudaMemcpyHostToDevice);

  cublasHandle_t hnd;
  cublasCreate(&hnd);
  cudaStream_t streams[numStreams];
  for (int i = 0; i < numStreams; i++) cudaStreamCreate(&streams[i]);

  // Setting initial values for the first set of models in the buffer to cold start.
  for (int i = 0; i < numModels; i++) {
    int bufModel = i;
    int actModel = i;
    thrust::fill(thrust::cuda::par.on(
      streams[i & (numStreams - 1)]),
      beta.dVar.begin() + i * N,
      beta.dVar.begin() + i * N + N, 0
    );
    set_model_kernel<ProcPrec><<<1, M, 0, streams[i & (numStreams - 1)]>>>(
      Y.GetRawDevPtr(), y.GetRawDevPtr(), mu.GetRawDevPtr(),
      a1.GetRawDevPtr(), a2.GetRawDevPtr(), lambda.GetRawDevPtr(), randnrm.GetRawDevPtr(),
      nVars.GetRawDevPtr(), eVars.GetRawDevPtr(), lasso.GetRawDevPtr(), step.GetRawDevPtr(),
      done.GetRawDevPtr(), act.GetRawDevPtr(), M, N, bufModel, actModel
    );
  }
  cudaDeviceSynchronize();

  GpuTimer timer;
  std::ofstream stepf("step.csv"), nvarsf("nvars.csv"), a1f("l1.csv"), a2f("err.csv"), lambdaf("G.csv"), betaf("beta.csv");

  int top = numModels;
  // double totalFlop = 0, corr_flop = 0;
  std::map<std::string, float> times;
  int completed_count = 0;
  std::map<int, int> completed;

  while (true) {

    // Check which models have completed.
    timer.start();
    auto it_check = thrust::make_zip_iterator(
      thrust::make_tuple(
        nVars.dVar.begin(), step.dVar.begin(),
        a1.dVar.begin(), a2.dVar.begin(),
        lambda.dVar.begin(), done.dVar.begin()
      )
    );
    thrust::transform(
      it_check,
      it_check + numModels,
      done.dVar.begin(),
      checkOp<ProcPrec>(maxVariables, maxSteps, l1, l2, g)
    );
    timer.stop();
    times["1. While (conditions)"] += timer.elapsed();
    printf("\n1. While (conditions)");

    // Check if any model has been completed to start correction and writing.
    timer.start();
    int ctrl = 0;
    done.SyncHost();
    act.SyncHost();
    for (int i = 0; i < numModels; i++) {
      if (done.hVar[i] && !completed[act.hVar[i]]) {
        ctrl = 1;
        break;
      }
    }

    // Perform correction and writing if there are models finished.
    if (ctrl) {
      step.SyncHost();
      nVars.SyncHost();
      act.SyncHost();
      a1.SyncHost();
      a2.SyncHost();
      lambda.SyncHost();

      for (int i = 0; i < numModels; i++) {
        if (done.hVar[i] && !completed[act.hVar[i]]) {
          completed[act.hVar[i]] = 1;
          completed_count++;
          stepf << act.hVar[i] << ", " << step.hVar[i] << "\n";
          nvarsf << act.hVar[i] << ", " << nVars.hVar[i] << "\n";
          a1f << act.hVar[i] << ", " << a1.hVar[i] << "\n";
          a2f << act.hVar[i] << ", " << a2.hVar[i] << "\n";
          lambdaf << act.hVar[i] << ", " << lambda.hVar[i] << "\n";
  // 				int hlVars[hNVars[i]];
  // 				corr_precision hbeta[hNVars[i]];
  // 				cudaMemcpy(hlVars, lVars + i * M, hNVars[i] * sizeof(int), cudaMemcpyDeviceToHost);
  // 				cudaMemcpy(hbeta, corr_beta + i * M, hNVars[i] * sizeof(corr_precision), cudaMemcpyDeviceToHost);
  // 				for (int j = 0; j < hNVars[i]; j++) betaf << hact[i] << ", " << hlVars[j] << ", " << hbeta[j] << "\n";
        }
      }

      // Replace the completed models with fresh models to be run.
      for (int i = 0, s = 0; i < numModels && top < totalModels; i++) {
        if (done.hVar[i]) {
          int bufModel = i;
          int actModel = top++;
          thrust::fill(thrust::cuda::par.on(
            streams[s & (numStreams - 1)]),
            beta.dVar.begin() + i * N,
            beta.dVar.begin() + i * N + N, 0
          );
          set_model_kernel<ProcPrec><<<1, M, 0, streams[s & (numStreams - 1)]>>>(
            Y.GetRawDevPtr(), y.GetRawDevPtr(), mu.GetRawDevPtr(),
            a1.GetRawDevPtr(), a2.GetRawDevPtr(), lambda.GetRawDevPtr(), randnrm.GetRawDevPtr(),
            nVars.GetRawDevPtr(), eVars.GetRawDevPtr(), lasso.GetRawDevPtr(), step.GetRawDevPtr(),
            done.GetRawDevPtr(), act.GetRawDevPtr(), M, N, bufModel, actModel
          );
          s++;
        }
      }
      cudaDeviceSynchronize();
    }
    printf("\rCompleted %d models", completed_count);
    if (completed_count == totalModels) {
      break;
    }
    timer.stop();
    times["2. Write Out"] += timer.elapsed();
    printf("\n2. Write Out");

    // Drop variables if condition met.
    timer.start();
    drop_kernel<<<1, numModels>>>(
      lVars.GetRawDevPtr(), dropidx.GetRawDevPtr(), nVars.GetRawDevPtr(),
      lasso.GetRawDevPtr(), M, numModels, done.GetRawDevPtr()
    );
    cudaDeviceSynchronize();
    timer.stop();
    times["3. A(dropidx) = []"] += timer.elapsed();
    printf("\n3. A(dropidx) = []");

    // Find residue.
    timer.start();
    mat_sub<ProcPrec>(y.GetRawDevPtr(), mu.GetRawDevPtr(), r.GetRawDevPtr(), numModels * M);
    cudaDeviceSynchronize();
    timer.stop();
    times["4. r = y - mu"] += timer.elapsed();
    printf("\n4. r = y - mu");

    // Compute the correlations.
    timer.start();
    cublasSetStream(hnd, NULL);
    gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N, N, numModels, M, &alp, X.GetRawDevPtr(), N, r.GetRawDevPtr(), M, &bet, c.GetRawDevPtr(), N);
    cudaDeviceSynchronize();
    timer.stop();
    times["5. c = X' * r"] += timer.elapsed();
    printf("\n5. c = X' * r");

    // Exclude logic and tricks.
    timer.start();
    exclude_kernel<ProcPrec><<<1, numModels>>>(c.GetRawDevPtr(), lVars.GetRawDevPtr(), nVars.GetRawDevPtr(), eVars.GetRawDevPtr(), act.GetRawDevPtr(), M, N, numModels, 0, done.GetRawDevPtr());
    cudaDeviceSynchronize();
    timer.stop();
    times["6. c(A) = 0"] += timer.elapsed();
    printf("\n6. c(A) = 0");

    // Finding cmax, cidx.
    timer.start();
    thrust::reduce_by_key(
      thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), _1 / N),
      thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), _1 / N) + numModels * N,
      thrust::make_zip_iterator(
        thrust::make_tuple(
          thrust::make_transform_iterator(c.dVar.begin(), absoluteOp<ProcPrec>()),
          thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), _1 % N)
        )
      ),
      thrust::make_discard_iterator(),
      thrust::make_zip_iterator(thrust::make_tuple(cmax.dVar.begin(), cidx.dVar.begin())),
      thrust::equal_to<int>(),
      thrust::maximum<thrust::tuple<ProcPrec, int> >()
    );
    cudaDeviceSynchronize();
    timer.stop();
    times["7. cmax, cidx = max(abs(c))"] += timer.elapsed();
    printf("\n7. cmax, cidx = max(abs(c))");

    // Adding variable to active set.
    timer.start();
    lasso_add_kernel<ProcPrec><<<1, numModels>>>(c.GetRawDevPtr(), lasso.GetRawDevPtr(), lVars.GetRawDevPtr(), nVars.GetRawDevPtr(), cidx.GetRawDevPtr(), M, N, numModels, done.GetRawDevPtr());
    cudaDeviceSynchronize();
    timer.stop();
    times["8. A = [A cidx]"] += timer.elapsed();printf("\n8. A = [A cidx]");

    // Compute XA = X(:, A).
    timer.start();
    nVars.SyncHost();
    lasso.SyncHost();
    dropidx.SyncHost();
    done.SyncHost();
    for (int i = 0, s = 0; i < numModels; i++) {
      if (done.hVar[i]) continue;
      gather<ProcPrec>(XA[i].GetRawDevPtr(), XA1[i].GetRawDevPtr(), X.GetRawDevPtr(), lVars.GetRawDevPtr(), nVars.hVar[i], lasso.hVar[i], dropidx.hVar[i], M, N, i, streams[s & (numStreams - 1)]);
      s++;
    }
    cudaDeviceSynchronize();
    timer.stop();
    times["9. XA = X(:, A)"] += timer.elapsed();
    printf("\n9. XA = X(:, A)");

    // Compute the Gram matrix.
    timer.start();
    for (int i = 0, s = 0; i < numModels; i++) {
      if (done.hVar[i]) continue;
      cublasSetStream(hnd, streams[s & (numStreams - 1)]);
      gemm(hnd, CUBLAS_OP_T, CUBLAS_OP_N, nVars.hVar[i], nVars.hVar[i], M, &alp, XA[i].GetRawDevPtr(), M, XA[i].GetRawDevPtr(), M, &bet, G[i].GetRawDevPtr(), nVars.hVar[i]);
      s++;
    }
    cudaDeviceSynchronize();
    timer.stop();
    times["10. G = XA' * XA"] += timer.elapsed();
    printf("\n10. G = XA' * XA");

    // Computer Inverse of Gram matrix.
    timer.start();
    for (int i = 0, s = 0; i < numModels; i++) {
      if (done.hVar[i]) continue;
      cublasSetStream(hnd, streams[s & (numStreams - 1)]);
      getrfBatched(hnd, nVars.hVar[i], pG.GetRawDevPtr() + i, nVars.hVar[i], NULL, info.GetRawDevPtr() + i, 1);
      s++;
    }
    cudaDeviceSynchronize();
    timer.stop();
    times["11. Inv(G) (1)"] += timer.elapsed();
    printf("\n11. Inv(G) (1)");

    timer.start();
    for (int i = 0, s = 0; i < numModels; i++) {
      if (done.hVar[i]) continue;
      cublasSetStream(hnd, streams[s & (numStreams - 1)]);
      getriBatched(hnd, nVars.hVar[i], pG.GetRawDevPtr() + i, nVars.hVar[i], NULL, pI.GetRawDevPtr() + i, nVars.hVar[i], info.GetRawDevPtr() + i, 1);
      s++;
    }
    cudaDeviceSynchronize();
    timer.stop();
    times["12. Inv(G) (2)"] += timer.elapsed();
    printf("\n12. Inv(G) (2)");

    // Check if inverse matrix is well conditioned.
    timer.start();
    int maxVar = -1;
    for (int i = 0; i < numModels; i++) maxVar = max(maxVar, nVars.hVar[i]);
    IrBatched<ProcPrec>(pI.GetRawDevPtr(), rander1.GetRawDevPtr(), r.GetRawDevPtr(), nVars.GetRawDevPtr(), M, numModels, maxVar);
    cudaDeviceSynchronize();
    IrBatched<ProcPrec>(pI.GetRawDevPtr(), rander2.GetRawDevPtr(), d.GetRawDevPtr(), nVars.GetRawDevPtr(), M, numModels, maxVar);
    cudaDeviceSynchronize();
    checkNan_kernel<ProcPrec><<<1, numModels>>>(
      nVars.GetRawDevPtr(), eVars.GetRawDevPtr(), lVars.GetRawDevPtr(),
      info.GetRawDevPtr(), r.GetRawDevPtr(), d.GetRawDevPtr(),
      randnrm.GetRawDevPtr(), M, numModels, done.GetRawDevPtr()
    );
    cudaDeviceSynchronize();
    info.SyncHost();
    nVars.SyncHost();
    timer.stop();
    times["13. Conditioning(I)"] += timer.elapsed();
    printf("\n13. Conditioning(I)");

    // Recovering the I matrix for bad condition cases.
    timer.start();
    for (int i = 0, s = 0; i < numModels; i++) {
      if (info.hVar[i] != 0 && !done.hVar[i]) {
        cublasSetStream(hnd, streams[s & (numStreams - 1)]);
        gemm(hnd, CUBLAS_OP_T, CUBLAS_OP_N, nVars.hVar[i], nVars.hVar[i], M, &alp, XA[i].GetRawDevPtr(), M, XA[i].GetRawDevPtr(), M, &bet, G[i].GetRawDevPtr(), nVars.hVar[i]);
        s++;
      }
    }
    cudaDeviceSynchronize();

    for (int i = 0, s = 0; i < numModels; i++) {
      if (info.hVar[i] != 0 && !done.hVar[i]) {
        cublasSetStream(hnd, streams[s & (numStreams - 1)]);
        getrfBatched(hnd, nVars.hVar[i], pG.GetRawDevPtr() + i, nVars.hVar[i], NULL, info.GetRawDevPtr() + i, 1);
        s++;
      }
    }
    cudaDeviceSynchronize();

    for (int i = 0, s = 0; i < numModels; i++) {
      if (info.hVar[i] != 0 && !done.hVar[i]) {
        cublasSetStream(hnd, streams[s & (numStreams - 1)]);
        getriBatched(hnd, nVars.hVar[i], pG.GetRawDevPtr() + i, nVars.hVar[i], NULL, pI.GetRawDevPtr() + i, nVars.hVar[i], info.GetRawDevPtr() + i, 1);
        s++;
      }
    }
    cudaDeviceSynchronize();
    timer.stop();
    times["14. Recovering ill I"] += timer.elapsed();
    printf("\n14. Recovering ill I");

    timer.start();
    XAt_yBatched<ProcPrec>(pXA.GetRawDevPtr(), y.GetRawDevPtr(), r.GetRawDevPtr(), nVars.GetRawDevPtr(), M, numModels);
    cudaDeviceSynchronize();
    timer.stop();
    times["15. XA' * y"] += timer.elapsed();
    printf("\n15. XA' * y");

    timer.start();
    maxVar = -1;
    for (int i = 0; i < numModels; i++) maxVar = max(maxVar, nVars.hVar[i]);
    IrBatched<ProcPrec>(pI.GetRawDevPtr(), r.GetRawDevPtr(), betaOls.GetRawDevPtr(), nVars.GetRawDevPtr(), M, numModels, maxVar);
    cudaDeviceSynchronize();
    timer.stop();
    times["16. betaOls = I * (XA' * y)"] += timer.elapsed();
    printf("\n16. betaOls = I * (XA' * y)");

    timer.start();
    XAbetaOlsBatched<ProcPrec>(pXA.GetRawDevPtr(), betaOls.GetRawDevPtr(), d.GetRawDevPtr(), nVars.GetRawDevPtr(), M, numModels, maxVar);
    cudaDeviceSynchronize();
    timer.stop();
    times["17. d = XA * betaOls"] += timer.elapsed();
    printf("\n17. d = XA * betaOls");

    timer.start();
    mat_sub<ProcPrec>(d.GetRawDevPtr(), mu.GetRawDevPtr(), d.GetRawDevPtr(), numModels * M);
    cudaDeviceSynchronize();
    timer.stop();
    times["18. d = d - mu"] += timer.elapsed();
    printf("\n18. d = d - mu");

    timer.start();
    gammat_kernel<ProcPrec><<<1, numModels>>>(gamma_tilde.GetRawDevPtr(), beta.GetRawDevPtr(), betaOls.GetRawDevPtr(), dropidx.GetRawDevPtr(), lVars.GetRawDevPtr(), nVars.GetRawDevPtr(), lasso.GetRawDevPtr(), M, N, numModels, done.GetRawDevPtr());
    cudaDeviceSynchronize();
    timer.stop();
    times["19. gamma_tilde, dropidx"] += timer.elapsed();
    printf("\n19. gamma_tilde, dropidx");

    timer.start();
    cublasSetStream(hnd, NULL);
    gemm(hnd, CUBLAS_OP_N, CUBLAS_OP_N, N, numModels, M, &alp, X.GetRawDevPtr(), N, d.GetRawDevPtr(), M, &bet, cd.GetRawDevPtr(), N);
    cudaDeviceSynchronize();
    timer.stop();
    times["20. cd = X' * d"] += timer.elapsed();
    printf("\n20. cd = X' * d");

    timer.start();
    thrust::reduce_by_key(
      thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), _1 / N),
      thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), _1 / N) + numModels * N,
      thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(
          c.dVar.begin(),
          cd.dVar.begin(),
          thrust::make_permutation_iterator(
              cmax.dVar.begin(),
              thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), _1 / N)
          )
        )),
        cdTransform<ProcPrec>()
      ),
      thrust::make_discard_iterator(),
      gamma.dVar.begin(),
      thrust::equal_to<int>(),
      thrust::minimum<ProcPrec>()
    );
    cudaDeviceSynchronize();
    timer.stop();
    times["21. gamma"] += timer.elapsed();
    printf("\n21. gamma");

    timer.start();
    set_gamma_kernel<ProcPrec><<<1, numModels>>>(
      gamma.GetRawDevPtr(), gamma_tilde.GetRawDevPtr(), lasso.GetRawDevPtr(),
      nVars.GetRawDevPtr(), maxVariables, M, numModels, done.GetRawDevPtr()
    );
    cudaDeviceSynchronize();
    timer.stop();
    times["22. gamma or gamma_tilde"] += timer.elapsed();
    printf("\n22. gamma or gamma_tilde");

    timer.start();
    update_kernel<ProcPrec><<<1, numModels>>>(
      beta.GetRawDevPtr(), beta_prev.GetRawDevPtr(), mu.GetRawDevPtr(), d.GetRawDevPtr(),
      betaOls.GetRawDevPtr(), gamma.GetRawDevPtr(), pXA.GetRawDevPtr(), y.GetRawDevPtr(),
      a1.GetRawDevPtr(), a2.GetRawDevPtr(), lambda.GetRawDevPtr(), lVars.GetRawDevPtr(),
      nVars.GetRawDevPtr(), step.GetRawDevPtr(), M, N, numModels, l1, done.GetRawDevPtr()
    );
    cudaDeviceSynchronize();
    timer.stop();
    times["23. Update mu, beta + L1 correction"] += timer.elapsed();
    printf("\n23. Update mu, beta + L1 correction");

  // 	totalFlop += flopCounter(M, N, numModels, hNVars);
  }

  stepf.close();
  nvarsf.close();
  a1f.close();
  a2f.close();
  lambdaf.close();
  betaf.close();

  // Statistics
  // double transferTime = times[0];
  // double execTime = 0;
  // for (int i = 1; i < 25; i++) execTime += times[i];
  // printf("\n");

  // std::ofstream speedf("speed.csv");
  // for (int i = 0; i < 25; i++) {
  // 	speedf << i << ", " << times[i] << "\n";
  // }
  // speedf << (corr_flop * 1.0e-9) / (transferTime * 1.0e-3) << ", " << (totalFlop * 1.0e-9) / (execTime * 1.0e-3) << "\n";
  // speedf.close();

  // cudaFree(corr_beta);
  // cudaFree(corr_sb);
  // cudaFree(corr_y);
  // cudaFree(corr_tmp);
  // cudaFree(corr_betaols);
  // cudaFree(corr_yh);
  // cudaFree(corr_z);

  // for (int i = 0; i < numModels; i++) {

  // 	cudaFree(corr_XA[i]);
  // 	cudaFree(corr_G[i]);
  // 	cudaFree(corr_I[i]);
  // }

  for (int i = 0; i < numStreams; i++) cudaStreamDestroy(streams[i]);

  // cudaFree(corr_dXA);
  // cudaFree(corr_dG);
  // cudaFree(corr_dI);

  cublasDestroy(hnd);
}


#endif
