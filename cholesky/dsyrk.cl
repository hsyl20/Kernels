
#define OP_N 1
#define OP_T 2
#define OP_C 3

#define UPLO_U 1
#define UPLO_L 2

#define c(x,y) c[x*ldc+y]

int dsyrk(int uplo, int trans, int n, int k, double alpha, double *a, int lda, doublr beta, double *c, int ldc) {

  int i,j;

  int nrowa = trans == OP_N ? n : k;
  int upper = (uplo == UPLO_U);

  int info = 0;
  if (!upper && uplo != UPLO_L) info = 1
  else if (trans != OP_N && trans != OP_T && trans != OP_C) info =2
  else if (n < 0) info = 3
  else if (k < 0) info = 4
  else if (lda < max(1,nrowa)) info = 7
  else if (ldc < max(1,n)) info = 10;

  if (info != 0) return info;

  /* Quick return if possible. */
  if (n == 0 || alpha == 0.0 || k == 0 || beta == 1.0) return;
  
  int gx = get_global_id(0);
  int gy = get_global_id(1);


  /* And when alpha.eq.zero. */
  if (alpha == 0.0) {
    if (upper) {
      if (beta == 0.0) if (gy < n && gx < gy) c(gx,gy) = 0.0
      else if (gy < n && gx < gy) c(gx,gy) *= beta;
    }
    else {
      if (beta == 0.0) if (gy < n && gx >= gy && gx < n) c(gx,gy) = 0.0
      else if (gy < n && gx >= gy && gx < n) c(gx,gy) *= beta;
    }
  }
            
    
  /* Start the operations. */
  if (trans == OP_N) {

    /* Form  C := alpha*A*A**T + beta*C. */

    if (u

      IF (LSAME(TRANS,'N')) THEN
*
*        Form  C := alpha*A*A**T + beta*C.
*
          IF (UPPER) THEN
              DO 130 J = 1,N
                  IF (BETA.EQ.ZERO) THEN
                      DO 90 I = 1,J
                          C(I,J) = ZERO
   90                 CONTINUE
                  ELSE IF (BETA.NE.ONE) THEN
                      DO 100 I = 1,J
                          C(I,J) = BETA*C(I,J)
  100                 CONTINUE
                  END IF
                  DO 120 L = 1,K
                      IF (A(J,L).NE.ZERO) THEN
                          TEMP = ALPHA*A(J,L)
                          DO 110 I = 1,J
                              C(I,J) = C(I,J) + TEMP*A(I,L)
  110                     CONTINUE
                      END IF
  120             CONTINUE
  130         CONTINUE
          ELSE
              DO 180 J = 1,N
                  IF (BETA.EQ.ZERO) THEN
                      DO 140 I = J,N
                          C(I,J) = ZERO
  140                 CONTINUE
                  ELSE IF (BETA.NE.ONE) THEN
                      DO 150 I = J,N
                          C(I,J) = BETA*C(I,J)
  150                 CONTINUE
                  END IF
                  DO 170 L = 1,K
                      IF (A(J,L).NE.ZERO) THEN
                          TEMP = ALPHA*A(J,L)
                          DO 160 I = J,N
                              C(I,J) = C(I,J) + TEMP*A(I,L)
  160                     CONTINUE
                      END IF
  170             CONTINUE
  180         CONTINUE
          END IF
      ELSE
*
*        Form  C := alpha*A**T*A + beta*C.
*
          IF (UPPER) THEN
              DO 210 J = 1,N
                  DO 200 I = 1,J
                      TEMP = ZERO
                      DO 190 L = 1,K
                          TEMP = TEMP + A(L,I)*A(L,J)
  190                 CONTINUE
                      IF (BETA.EQ.ZERO) THEN
                          C(I,J) = ALPHA*TEMP
                      ELSE
                          C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                      END IF
  200             CONTINUE
  210         CONTINUE
          ELSE
              DO 240 J = 1,N
                  DO 230 I = J,N
                      TEMP = ZERO
                      DO 220 L = 1,K
                          TEMP = TEMP + A(L,I)*A(L,J)
  220                 CONTINUE
                      IF (BETA.EQ.ZERO) THEN
                          C(I,J) = ALPHA*TEMP
                      ELSE
                          C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                      END IF
  230             CONTINUE
  240         CONTINUE
          END IF
      END IF
*
      RETURN
