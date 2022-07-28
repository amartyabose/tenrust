use std::collections::HashSet;

use crate::error::*;
use crate::index::*;

use ndarray::*;
use num::Num;

#[derive(Debug, Clone)]
pub struct Tensor<T: 'static + Clone + Copy + Num> {
    pub indices: Vec<Index>,
    pub indices_set: HashSet<Index>,
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
            indices_set: inds
                .to_vec()
                .into_iter()
                .cloned()
                .collect::<HashSet<Index>>(),
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
            indices_set: indices.to_vec().into_iter().collect::<HashSet<Index>>(),
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

    pub fn data_as_matrix_given_rows(&self, row_inds: &[&Index]) -> (Array<T, Ix2>, Vec<Index>) {
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

    pub fn data_as_matrix_given_cols(&self, col_inds: &[&Index]) -> (Array<T, Ix2>, Vec<Index>) {
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

    pub fn data_as_matrix(&self, row_inds: &[Index], col_inds: &[Index]) -> Array<T, Ix2> {
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

pub fn common_indices<T: 'static + Copy + Clone + Num>(
    tens1: &Tensor<T>,
    tens2: &Tensor<T>,
) -> Vec<Index> {
    tens1
        .indices_set
        .intersection(&tens2.indices_set)
        .cloned()
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tensor_tests {
    use super::*;

    #[test]
    fn build_zero_tesor() {
        let i = Index::new(2);
        let j = Index::new(3);
        let atensor = Tensor::<f64>::new(&[&i, &j]);

        assert_eq!(atensor.indices.len(), 2);
        assert_eq!(atensor.data.shape(), [i.dim as usize, j.dim as usize]);
    }

    #[test]
    fn build_tensor_with_data() {
        let i = Index::new(2);
        let j = Index::new(3);
        let atensor_res = Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert!(atensor_res.is_ok());
        let atensor = atensor_res.unwrap();
        assert_eq!(atensor.indices.len(), 2);
        assert_eq!(atensor.data.shape(), [i.dim as usize, j.dim as usize]);
    }

    #[test]
    fn build_tensor_with_data_fail() {
        let i = Index::new(2);
        let j = Index::new(3);
        let atensor_res =
            Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        assert!(atensor_res.is_err());
    }

    #[test]
    fn access_tensor_elements() -> Result<()> {
        let i = Index::new(2);
        let j = Index::new(3);
        let atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

        assert_eq!(
            atensor[&[IndexVal::new(&i, 0)?, IndexVal::new(&j, 0)?]],
            1.0
        );
        assert_eq!(
            atensor[&[IndexVal::new(&i, 1)?, IndexVal::new(&j, 0)?]],
            4.0
        );
        assert_eq!(
            atensor[&[IndexVal::new(&i, 0)?, IndexVal::new(&j, 1)?]],
            2.0
        );
        Ok(())
    }

    #[test]
    fn modify_tensor_elements() -> Result<()> {
        let i = Index::new(2);
        let j = Index::new(3);
        let mut atensor =
            Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        atensor[&[IndexVal::new(&i, 0)?, IndexVal::new(&j, 0)?]] = 20.0;

        assert_eq!(
            atensor[&[IndexVal::new(&i, 0)?, IndexVal::new(&j, 0)?]],
            20.0
        );
        assert_eq!(
            atensor[&[IndexVal::new(&i, 1)?, IndexVal::new(&j, 0)?]],
            4.0
        );
        assert_eq!(
            atensor[&[IndexVal::new(&i, 0)?, IndexVal::new(&j, 1)?]],
            2.0
        );
        Ok(())
    }
}
