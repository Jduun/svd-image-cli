package cmd

import (
	"gonum.org/v1/gonum/mat"
	"log"
)

func SVD(matrix *mat.Dense) (*mat.Dense, *mat.Dense, *mat.Dense) {
	var svd mat.SVD
	ok := svd.Factorize(matrix, mat.SVDFull)
	if !ok {
		// TODO: handle error
		log.Fatal("SVD failed")
	}
	var u, v, vt, sigma mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	vt.CloneFrom(v.T())
	singularValues := svd.Values(nil)
	sigma.CloneFrom(mat.NewDiagDense(len(singularValues), singularValues))
	return &u, &sigma, &vt
}

func truncatedSVD(matrix *mat.Dense, rank int) (*mat.Dense, *mat.Dense, *mat.Dense) {
	u, sigma, vt := SVD(matrix)

	truncatedU := mat.DenseCopyOf(u.Slice(0, matrix.RawMatrix().Rows, 0, rank))
	truncatedSigma := mat.DenseCopyOf(sigma.Slice(0, rank, 0, rank))
	truncatedVT := mat.DenseCopyOf(vt.Slice(0, rank, 0, matrix.RawMatrix().Cols))

	return truncatedU, truncatedSigma, truncatedVT
}
