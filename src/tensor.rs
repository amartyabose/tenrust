use crate::error::*;
use crate::index::*;

use std::collections::HashSet;

use ndarray::*;
use ndarray_linalg::Lapack;
use ndarray_linalg::SVD;
use num::Num;

#[derive(Debug)]
pub struct Tensor<T: 'static + Clone + Copy + Num> {
    pub indices: Vec<Index>,
    pub data: ArrayD<T>,
}

impl<T: 'static + Clone + Copy + Num> Tensor<T> {
    pub fn new(inds: &[&Index]) -> Tensor<T> {
        let indices = inds.to_vec();
        let shape_vec = (&indices)
            .iter()
            .map(|x| x.dim as usize)
            .collect::<Vec<usize>>();
        let shape = &shape_vec[..shape_vec.len()];
        Tensor {
            indices: indices.into_iter().cloned().collect::<Vec<Index>>(),
            data: ArrayD::<T>::zeros(IxDyn(shape)),
        }
    }

    pub fn new_owned(indices: &[Index]) -> Tensor<T> {
        let shape_vec = indices
            .iter()
            .map(|x| x.dim as usize)
            .collect::<Vec<usize>>();
        let shape = &shape_vec[..shape_vec.len()];
        Tensor {
            indices: indices.to_vec(),
            data: ArrayD::<T>::zeros(IxDyn(shape)),
        }
    }

    pub fn with_data(mut self, dat: &[T]) -> Result<Tensor<T>> {
        if dat.len() as usize == self.data.len() {
            let tmp = Array::<T, Ix1>::from_iter(dat.iter().cloned());
            self.data = tmp.into_shape(self.data.shape())?;
            Ok(self)
        } else {
            Err(TenRustError::DimensionMismatch("Tensor::with_data"))
        }
    }

    pub fn with_data_from_iter(mut self, dat: impl IntoIterator<Item = T>) -> Result<Tensor<T>> {
        let tmp = Array::<T, Ix1>::from_iter(dat);
        self.data = tmp.into_shape(self.data.shape())?;
        Ok(self)
    }

    fn data_as_matrix_given_rows(&self, row_inds: &[&Index]) -> (Array<T, Ix2>, Vec<Index>) {
        let mut my_inds_order = Vec::new();
        let mut my_rows = 1;
        for ind in row_inds {
            my_rows *= ind.dim;
            if let Some(loc) = self.indices.iter().position(|x| x == *ind) {
                my_inds_order.push(loc);
            }
        }
        let mut my_cols: u64 = 1;
        let mut cols = Vec::new();
        for (loc, ind) in (&self.indices).iter().enumerate() {
            if !row_inds.contains(&ind) {
                cols.push(ind);
                my_inds_order.push(loc);
                my_cols *= ind.dim;
            }
        }
        let my_data = self.data.clone().permuted_axes(&my_inds_order[..]);
        (
            Array::from_iter(my_data.iter().cloned())
                .into_shape(Ix2(my_rows as usize, my_cols as usize))
                .unwrap(),
            cols.iter().map(|x| (*x).clone()).collect::<Vec<_>>(),
        )
    }

    fn data_as_matrix_given_cols(&self, col_inds: &[&Index]) -> (Array<T, Ix2>, Vec<Index>) {
        let mut my_inds_order = Vec::new();
        let mut my_rows = 1;
        let mut rows = Vec::new();
        for (loc, ind) in (&self.indices).iter().enumerate() {
            if !col_inds.contains(&ind) {
                rows.push(ind);
                my_inds_order.push(loc);
                my_rows *= ind.dim;
            }
        }
        let mut my_cols: u64 = 1;
        for ind in col_inds {
            my_cols *= ind.dim;
            if let Some(loc) = self.indices.iter().position(|x| x == *ind) {
                my_inds_order.push(loc);
            }
        }
        let my_data = self.data.clone().permuted_axes(&my_inds_order[..]);
        (
            Array::from_iter(my_data.iter().cloned())
                .into_shape(Ix2(my_rows as usize, my_cols as usize))
                .unwrap(),
            rows.iter().map(|x| (*x).clone()).collect::<Vec<_>>(),
        )
    }

    fn data_as_matrix(&self, row_inds: &[Index], col_inds: &[Index]) -> Array<T, Ix2> {
        let mut my_inds_order = Vec::new();
        let mut my_rows = 1;
        for ind in row_inds {
            my_rows *= ind.dim;
            if let Some(loc) = self.indices.iter().position(|x| x == ind) {
                my_inds_order.push(loc);
            }
        }
        let mut my_cols: u64 = 1;
        for ind in col_inds {
            my_cols *= ind.dim;
            if let Some(loc) = self.indices.iter().position(|x| x == ind) {
                my_inds_order.push(loc);
            }
        }
        let my_data = self.data.clone().permuted_axes(&my_inds_order[..]);
        Array::from_iter(my_data.iter().cloned())
            .into_shape(Ix2(my_rows as usize, my_cols as usize))
            .unwrap()
    }

    pub fn dot(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        let self_inds = self.indices.iter().cloned().collect::<HashSet<_>>();
        let other_inds = other.indices.iter().cloned().collect::<HashSet<_>>();
        let common_inds = self_inds
            .intersection(&other_inds)
            .cloned()
            .collect::<Vec<_>>();
        let self_extra = self_inds
            .difference(&other_inds)
            .cloned()
            .collect::<Vec<_>>();
        let other_extra = other_inds
            .difference(&self_inds)
            .cloned()
            .collect::<Vec<_>>();

        let my_data = self.data_as_matrix(&self_extra[..], &common_inds[..]);
        let other_data = other.data_as_matrix(&common_inds[..], &other_extra[..]);
        let result_data = my_data.dot(&other_data);
        let mut indices = Vec::new();
        for inds in &self_extra {
            indices.push(inds);
        }
        for inds in &other_extra {
            indices.push(inds);
        }

        Tensor::new(&indices[..]).with_data_from_iter(result_data.iter().cloned())
    }
}

pub struct TensorSVD<T: 'static + Clone + Copy + Num + Lapack> {
    utensor: Tensor<T>,
    stensor: Tensor<<T as ndarray_linalg::Scalar>::Real>,
    v_trans_tensor: Tensor<T>,
}

impl<T: 'static + Clone + Copy + Num + Lapack> TensorSVD<T> {
    pub fn new(
        uparam: Tensor<T>,
        sparam: Tensor<<T as ndarray_linalg::Scalar>::Real>,
        vparam: Tensor<T>,
    ) -> TensorSVD<T> {
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

impl<T: 'static + Clone + Copy + Num + Lapack> Tensor<T> {
    pub fn untruncated_svd(&self, svd_inds: SVDIndices) -> Result<TensorSVD<T>> {
        let (mat, mut row_inds, mut col_inds) = match svd_inds {
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
        };
        let svddat = mat.svd(true, true)?;
        let (uopt, sdiag, vtopt) = svddat;
        let umat = uopt.ok_or(TenRustError::SVDError)?;
        let vmat = vtopt.ok_or(TenRustError::SVDError)?;
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
        let mut stensor =
            Tensor::<<T as ndarray_linalg::Scalar>::Real>::new(&[&linkind0, &linkind1]);
        for (ind, sval) in sdiag.iter().enumerate() {
            let i0val = IndexVal::new(&linkind0, ind as u64)?;
            let i1val = IndexVal::new(&linkind1, ind as u64)?;
            stensor[&[i0val, i1val]] = *sval;
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

impl<T: 'static + Clone + Copy + Num> std::ops::Index<&[IndexVal]> for Tensor<T> {
    type Output = T;

    fn index(&self, inds: &[IndexVal]) -> &Self::Output {
        if inds.len() != self.indices.len() {
            panic!(
                "The number of indices provided, {}, does not match the rank of the tensor, {}.",
                inds.len(),
                self.indices.len()
            );
        }
        let mut loc_vec = Vec::new();
        for indval in inds {
            loc_vec.push(indval.val as usize);
        }
        &self.data[&loc_vec[..]]
    }
}

impl<T: 'static + Clone + Copy + Num> std::ops::IndexMut<&[IndexVal]> for Tensor<T> {
    fn index_mut(&mut self, inds: &[IndexVal]) -> &mut Self::Output {
        if inds.len() != self.indices.len() {
            panic!(
                "The number of indices provided, {}, does not match the rank of the tensor, {}.",
                inds.len(),
                self.indices.len()
            );
        }
        let mut loc_vec = Vec::new();
        for indval in inds {
            loc_vec.push(indval.val as usize);
        }
        &mut self.data[&loc_vec[..]]
    }
}

#[cfg(test)]
mod tensor_tests;
