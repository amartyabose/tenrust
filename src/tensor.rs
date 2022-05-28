use crate::error::*;
use crate::index::*;

use std::collections::HashSet;

use ndarray::*;
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

        let my_data = self.data.clone().permuted_axes(&my_inds_order[..]);
        // let my_data = my_data.permuted_axes(&my_inds_order[..]);
        let my_data = Array::from_iter(my_data.iter().cloned())
            .into_shape([my_rows as usize, my_cols as usize])?;
        let other_data = other.data.clone().permuted_axes(&other_inds_order[..]);
        // let other_data = other_data.permuted_axes(&other_inds_order[..]);
        let other_data = Array::from_iter(other_data.iter().cloned())
            .into_shape([my_cols as usize, other_cols as usize])?;
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
mod tensor_tests {
    use ndarray_linalg::assert_close_l2;

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
    fn multiply_tenrust() -> Result<()> {
        let i = Index::new(2);
        let j = Index::new(3);
        let k = Index::new(2);
        let atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let btensor = Tensor::<f64>::new(&[&j, &k]).with_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let ctensor = atensor.dot(&btensor)?;
        let anses = vec![22.0, 28.0, 49.0, 64.0];
        let ans = ndarray::Array::from(anses);
        let ans = ans.into_shape(IxDyn(&[2, 2]))?;
        assert_close_l2!(&ctensor.data, &ans, 1e-12);
        Ok(())
    }
}
