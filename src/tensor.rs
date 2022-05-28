use crate::error::*;
use crate::index::*;

use std::collections::HashSet;

use ndarray::*;
use ndarray_linalg::*;
use num::Num;

#[derive(Debug)]
pub struct Tensor<T: Num + Clone + std::fmt::Debug> {
    pub indices: Vec<Index>,
    pub data: ArrayD<T>,
}

impl<T: Num + Clone + std::fmt::Debug> Tensor<T> {
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

    pub fn with_data(mut self, dat: &[T]) -> Result<Tensor<T>> {
        if dat.len() as usize == self.data.len() {
            let tmp = Array::from_iter(dat.iter().cloned());
            self.data = tmp.into_shape(self.data.shape())?;
            Ok(self)
        } else {
            Err(TenRustError::DimensionMismatch("Tensor<T>::with_data"))
        }
    }

    pub fn with_data_from_iter(mut self, dat: impl IntoIterator<Item = T>) -> Result<Tensor<T>> {
        let tmp = Array::from_iter(dat);
        self.data = tmp.into_shape(self.data.shape())?;
        Ok(self)
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

        let mut my_inds_order = Vec::new();
        let mut my_rows = 1;
        for ind in &self_extra {
            my_rows *= ind.dim;
            if let Some(loc) = self.indices.iter().position(|x| x == ind) {
                my_inds_order.push(loc);
            }
        }
        let mut other_inds_order = Vec::new();
        let mut my_cols = 1;
        for ind in &common_inds {
            my_cols *= ind.dim;
            if let Some(loc) = self.indices.iter().position(|x| x == ind) {
                my_inds_order.push(loc);
            }
            if let Some(loc) = other.indices.iter().position(|x| x == ind) {
                other_inds_order.push(loc);
            }
        }
        let mut other_cols = 1;
        for ind in &other_extra {
            other_cols *= ind.dim;
            if let Some(loc) = other.indices.iter().position(|x| x == ind) {
                other_inds_order.push(loc);
            }
        }

        let my_data = self.data.clone();
        let my_data = my_data.permuted_axes(&my_inds_order[..]);
        let my_data = Array::from_iter(my_data.iter().cloned()).into_shape([my_rows as usize, my_cols as usize])?;
        let other_data = other.data.clone();
        let other_data = other_data.permuted_axes(&other_inds_order[..]);
        let other_data = Array::from_iter(other_data.iter().cloned()).into_shape([my_cols as usize, other_cols as usize])?;
        let result_data = my_data.dot(&other_data);
        let mut indices = Vec::new();
        for inds in &self_extra {
            indices.push(inds);
        }
        for inds in &other_extra {
            indices.push(inds);
        }

        Ok(Tensor::new(&indices[..]).with_data_from_iter(result_data.iter()))
    }
}

impl<T: Num + Clone + std::fmt::Debug> std::ops::Index<&[IndexVal]> for Tensor<T> {
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

impl<T: Num + Clone + std::fmt::Debug> std::ops::IndexMut<&[IndexVal]> for Tensor<T> {
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
mod tensor_tests {
    use crate::tensor::*;

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
        let atensor_res = Tensor::<u64>::new(&[&i, &j]).with_data(&[1, 2, 3, 4, 5, 6]);

        assert!(atensor_res.is_ok());
        let atensor = atensor_res.unwrap();
        assert_eq!(atensor.indices.len(), 2);
        assert_eq!(atensor.data.shape(), [i.dim as usize, j.dim as usize]);
    }

    #[test]
    fn build_tensor_with_data_fail() {
        let i = Index::new(2);
        let j = Index::new(3);
        let atensor_res = Tensor::<u64>::new(&[&i, &j]).with_data(&[1, 2, 3, 4, 5, 6, 7, 8]);

        assert!(atensor_res.is_err());
    }

    #[test]
    fn access_tensor_elements() -> Result<()> {
        let i = Index::new(2);
        let j = Index::new(3);
        let atensor = Tensor::<u64>::new(&[&i, &j]).with_data(&[1, 2, 3, 4, 5, 6])?;

        assert_eq!(atensor[&[IndexVal::new(&i, 0)?, IndexVal::new(&j, 0)?]], 1);
        assert_eq!(atensor[&[IndexVal::new(&i, 1)?, IndexVal::new(&j, 0)?]], 4);
        assert_eq!(atensor[&[IndexVal::new(&i, 0)?, IndexVal::new(&j, 1)?]], 2);
        Ok(())
    }

    #[test]
    fn multiply_tenrust() -> Result<()> {
        let i = Index::new(2);
        let j = Index::new(3);
        let k = Index::new(2);
        let atensor = Tensor::<u64>::new(&[&i, &j]).with_data(&[1, 2, 3, 4, 5, 6])?;
        let btensor = Tensor::<u64>::new(&[&j, &k]).with_data(&[1, 2, 3, 4, 5, 6])?;
        atensor.dot(&btensor);

        assert_eq!(atensor[&[IndexVal::new(&i, 0)?, IndexVal::new(&j, 0)?]], 1);
        assert_eq!(atensor[&[IndexVal::new(&i, 1)?, IndexVal::new(&j, 0)?]], 4);
        assert_eq!(atensor[&[IndexVal::new(&i, 0)?, IndexVal::new(&j, 1)?]], 2);
        Ok(())
    }
}
