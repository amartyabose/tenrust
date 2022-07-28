use std::ops::AddAssign;

use crate::error::*;
use crate::index::*;
use crate::tensor::*;

use ndarray::*;
use ndarray_linalg::Lapack;
use ndarray_linalg::SVD;
use num::One;
use num::{Num, Zero};

/// Struct for holding the output of the singular value decomposition.
/// T = U * S * V^T
pub struct TensorSVD<T: 'static + Clone + Copy + Num + Lapack> {
    pub utensor: Tensor<T>,
    pub stensor: Tensor<T>,
    pub v_trans_tensor: Tensor<T>,
}

impl<T: 'static + Clone + Copy + Num + Lapack> TensorSVD<T> {
    pub fn new(uparam: Tensor<T>, sparam: Tensor<T>, vparam: Tensor<T>) -> TensorSVD<T> {
        TensorSVD {
            utensor: uparam,
            stensor: sparam,
            v_trans_tensor: vparam,
        }
    }
}

pub enum SVDIndices<'a> {
    Rows(&'a [&'a Index]),
    Cols(&'a [&'a Index]),
}

impl<T: 'static + Clone + Copy + Num + Lapack + AddAssign + Zero + One> Tensor<T> {
    fn transform_to_matrix(&self, svd_inds: SVDIndices) -> (Array2<T>, Vec<Index>, Vec<Index>) {
        match svd_inds {
            SVDIndices::Rows(rows) => {
                let (m, cols) = self.data_as_matrix_given_rows(rows);
                (
                    m,
                    rows.iter().map(|x| (*x).clone()).collect::<Vec<_>>(),
                    cols,
                )
            }
            SVDIndices::Cols(cols) => {
                let (m, rows) = self.data_as_matrix_given_cols(cols);
                (
                    m,
                    rows,
                    cols.iter().map(|x| (*x).clone()).collect::<Vec<_>>(),
                )
            }
        }
    }

    pub fn svd(
        &self,
        svd_inds: SVDIndices,
        cutoff: Option<T::Real>,
        maxdim: Option<usize>,
    ) -> Result<TensorSVD<T>> {
        let (mat, mut row_inds, mut col_inds) = self.transform_to_matrix(svd_inds);
        let svddat = mat.svd(true, true)?;
        let (uopt, sdiag, vtopt) = svddat;
        let umat = uopt.ok_or(TenRustError::SVDError)?;
        let vmat = vtopt.ok_or(TenRustError::SVDError)?;
        let sdiag = if let Some(cutoff_val) = cutoff {
            let sum_sdiag_sq = &sdiag.dot(&sdiag);
            let sdiag_sq = &sdiag * &sdiag;
            let mut num_diags = 0usize;
            let mut sum_diags_sq_progressive = T::Real::zero();
            let one_minus_cutoff = T::Real::one() - cutoff_val;
            for (i, svalsq) in sdiag_sq.iter().enumerate() {
                sum_diags_sq_progressive += *svalsq as T::Real;
                if sum_diags_sq_progressive / *sum_sdiag_sq >= one_minus_cutoff {
                    num_diags = i + 1;
                    break;
                }
            }
            sdiag.slice(s![..num_diags]).to_owned()
        } else {
            sdiag
        };
        let sdiag = if let Some(maxdim_val) = maxdim {
            if sdiag.shape()[0] <= maxdim_val {
                sdiag
            } else {
                sdiag.slice(s![..maxdim_val]).to_owned()
            }
        } else {
            sdiag
        };
        // We want minimum storage in case of non-square matrices:
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        // Then using Mathematica:
        // U is a 2x2 matrix
        // S = [[9.5, 0.0, 0.0],
        //      [0.0, 0.77, 0.0]]
        // V is consequently a 3x3 matrix.
        // Notice that the last row of V is inconsequential because the last column of S, which multiplies it is all 0.
        // So the real dimensionality is governed by the number of non-zero singular values.
        let linkdim0 = sdiag.shape()[0] as u64;
        let linkind0 = Index::new(linkdim0).add_tag("S_Link");
        let linkdim1 = linkdim0;
        let linkind1 = Index::new(linkdim1).add_tag("S_Link");
        row_inds.push(linkind0.clone());
        let utensor = Tensor::<T>::new_owned(&row_inds).with_data_from_iter(
            umat.slice_axis(Axis(1), Slice::from(0..linkdim0 as i32))
                .iter()
                .copied(),
        )?;
        let mut stensor = Tensor::<T>::new(&[&linkind0, &linkind1]);
        for (ind, sval) in sdiag.iter().enumerate() {
            let i0val = IndexVal::new(&linkind0, ind as u64)?;
            let i1val = IndexVal::new(&linkind1, ind as u64)?;
            stensor[&[i0val, i1val]] = ndarray_linalg::Scalar::from_real(*sval);
        }
        col_inds.insert(0, linkind1);
        let vtensor = Tensor::<T>::new_owned(&col_inds).with_data_from_iter(
            vmat.slice_axis(Axis(0), Slice::from(0..linkdim0 as i32))
                .iter()
                .copied(),
        )?;
        Ok(TensorSVD::<T>::new(utensor, stensor, vtensor))
    }
}

#[cfg(test)]
mod singular_value_tests {
    use ndarray_linalg::assert_close_l2;

    use super::*;

    #[test]
    fn untruncated_svd_tenrust_normal() -> Result<()> {
        let i = Index::new(4);
        let j = Index::new(4);
        let atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ])?;
        let TensorSVD {
            utensor,
            stensor,
            v_trans_tensor: vtensor,
        } = atensor.svd(SVDIndices::Rows(&[&i]), None, None)?;
        dbg!(&stensor.data);
        let ans = utensor.dot(&stensor.dot(&vtensor)?)?;
        assert_close_l2!(&atensor.data, &ans.data, 1e-12);
        assert!(&utensor.indices.iter().any(|x| x == &i));
        assert!(&vtensor.indices.iter().any(|x| x == &j));
        Ok(())
    }

    #[test]
    fn untruncated_svd_tenrust_reverse() -> Result<()> {
        let i = Index::new(4);
        let j = Index::new(4);
        let atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ])?;
        let TensorSVD {
            utensor,
            stensor,
            v_trans_tensor: vtensor,
        } = atensor.svd(SVDIndices::Cols(&[&i]), None, None)?;
        let ans = utensor.dot(&stensor.dot(&vtensor)?)?;
        assert_close_l2!(&atensor.data.t(), &ans.data, 1e-12);
        assert!(&utensor.indices.iter().any(|x| x == &j));
        assert!(&vtensor.indices.iter().any(|x| x == &i));
        Ok(())
    }

    #[test]
    fn maxdim_truncated_svd() -> Result<()> {
        let i = Index::new(4);
        let j = Index::new(4);
        let atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ])?;
        let TensorSVD {
            utensor,
            stensor,
            v_trans_tensor: vtensor,
        } = atensor.svd(SVDIndices::Rows(&[&i]), None, Some(2))?;
        let ans = utensor.dot(&stensor.dot(&vtensor)?)?;
        dbg!(&stensor.data);
        assert_close_l2!(&atensor.data, &ans.data, 1e-12);
        assert!(&utensor.indices.iter().any(|x| x == &i));
        assert!(&vtensor.indices.iter().any(|x| x == &j));
        assert!(stensor.indices[0].dim <= 3);
        assert!(stensor.indices[1].dim <= 3);
        Ok(())
    }

    #[test]
    fn cutoff_truncated_svd() -> Result<()> {
        let i = Index::new(4);
        let j = Index::new(4);
        let atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ])?;
        let TensorSVD {
            utensor,
            stensor,
            v_trans_tensor: vtensor,
        } = atensor.svd(SVDIndices::Rows(&[&i]), Some(1e-14), None)?;
        let ans = utensor.dot(&stensor.dot(&vtensor)?)?;
        dbg!(&stensor.data);
        assert_close_l2!(&atensor.data, &ans.data, 1e-12);
        assert!(&utensor.indices.iter().any(|x| x == &i));
        assert!(&vtensor.indices.iter().any(|x| x == &j));
        assert!(stensor.indices[0].dim <= 3);
        assert!(stensor.indices[1].dim <= 3);
        Ok(())
    }
}
