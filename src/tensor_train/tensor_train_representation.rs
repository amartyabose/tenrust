use crate::error::*;

use crate::index::Index;
use crate::linalg::singular_value::*;
use crate::tensor::*;
use std::iter::zip;

use ndarray_linalg::Lapack;
use num::Num;

pub trait TensorTrain<T: 'static + Clone + Copy + Num> {
    fn get_tensors(&self) -> &Vec<Tensor<T>>;

    fn get_tensor(&self) -> Result<Tensor<T>> {
        let tensors = self.get_tensors();
        let mut tens = tensors.first().cloned().unwrap();
        for tensor in tensors.iter().skip(1) {
            tens = tens.dot(tensor)?;
        }
        Ok(tens)
    }

    fn link_indices(&self) -> Vec<Index> {
        let tensors = self.get_tensors();
        let mut indices = Vec::new();
        for (a, b) in zip(tensors.iter(), tensors.iter().skip(1)) {
            let mut inds = common_indices(a, b);
            indices.append(&mut inds);
        }
        indices
    }
}

#[derive(Debug)]
pub struct MatrixProduct<T: 'static + Clone + Copy + Num> {
    pub tensors: Vec<Tensor<T>>,
}

impl<T> TensorTrain<T> for MatrixProduct<T>
where
    T: 'static + Clone + Copy + Num,
{
    fn get_tensors(&self) -> &Vec<Tensor<T>> {
        &self.tensors
    }
}

impl<T: 'static + Clone + Copy + Num + Lapack> MatrixProduct<T> {
    pub fn recompress(
        self,
        cutoff: Option<T::Real>,
        maxdim: Option<usize>,
    ) -> Result<MatrixProduct<T>> {
        let TensorSVD {
            utensor,
            stensor,
            v_trans_tensor: vtensor,
        } = self.tensors[0].svd(
            SVDIndices::Rows(&[&self.tensors[0].indices[0]]),
            cutoff,
            maxdim,
        )?;
        let mut mps = vec![utensor];
        let mut vnew = stensor.dot(&vtensor.dot(&self.tensors[1])?)?;
        for _n in 2..self.tensors.len() {
            let TensorSVD {
                utensor,
                stensor,
                v_trans_tensor: vtensor,
            } = vnew.svd(
                SVDIndices::Rows(&[&vnew.indices[0], &vnew.indices[1]]),
                cutoff,
                maxdim,
            )?;
            mps.push(utensor);
            vnew = stensor.dot(&vtensor.dot(&self.tensors[1])?)?;
        }
        mps.push(vnew);
        Ok(MatrixProduct { tensors: mps })
    }
}

pub fn mp_representation<T>(
    atensor: &Tensor<T>,
    cutoff: Option<T::Real>,
    maxdim: Option<usize>,
) -> Result<MatrixProduct<T>>
where
    T: 'static + Clone + Copy + Num + Lapack,
{
    if atensor.indices.len() == 1 {
        Ok(MatrixProduct {
            tensors: vec![atensor.clone()],
        })
    } else {
        let TensorSVD {
            utensor,
            stensor,
            v_trans_tensor: vtensor,
        } = atensor.svd(SVDIndices::Rows(&[&atensor.indices[0]]), cutoff, maxdim)?;
        let mut vnew = stensor.dot(&vtensor)?;
        let mut mps = vec![utensor];
        for _n in 2..atensor.indices.len() {
            let TensorSVD {
                utensor,
                stensor,
                v_trans_tensor: vtensor,
            } = vnew.svd(
                SVDIndices::Rows(&[&vnew.indices[0], &vnew.indices[1]]),
                cutoff,
                maxdim,
            )?;
            mps.push(utensor);
            vnew = stensor.dot(&vtensor)?;
        }
        mps.push(vnew);
        Ok(MatrixProduct { tensors: mps })
    }
}

#[cfg(test)]
mod mp_representation_tests {
    use super::*;
    use crate::index::*;
    use ndarray_linalg::assert_close_l2;

    #[test]
    fn single_index_mp() -> Result<()> {
        let i = Index::new(2);
        let atensor = Tensor::<f64>::new(&[&i]).with_data(&[1., 2.])?;
        let atensor_mps = mp_representation(&atensor, Some(1e-5), Some(20))?;
        let atensor_mult = atensor_mps.get_tensor()?;
        assert_close_l2!(&atensor.data, &atensor_mult.data, 1e-12);
        Ok(())
    }

    #[test]
    fn two_index_mp() -> Result<()> {
        let i = Index::new(2);
        let j = Index::new(2);
        let atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[1., 2., 3., 4.])?;
        let atensor_mps = mp_representation(&atensor, Some(1e-5), Some(20))?;
        let atensor_mult = atensor_mps.get_tensor()?;
        assert_close_l2!(
            &atensor.data_as_matrix(&[i.clone()], &[j.clone()]),
            &atensor_mult.data_as_matrix(&[i], &[j]),
            1e-12
        );
        Ok(())
    }

    #[test]
    fn three_index_mp() -> Result<()> {
        let i = Index::new(2);
        let j = Index::new(2);
        let k = Index::new(2);
        let atensor =
            Tensor::<f64>::new(&[&i, &j, &k]).with_data(&[1., 2., 3., 4., 5., 6., 7., 8.])?;
        let atensor_mps = mp_representation(&atensor, Some(1e-5), Some(20))?;
        let atensor_mult = atensor_mps.get_tensor()?;
        assert_close_l2!(
            &atensor.data_as_matrix(&[i.clone(), j.clone()], &[k.clone()]),
            &atensor_mult.data_as_matrix(&[i, j], &[k]),
            1e-12
        );
        Ok(())
    }

    #[test]
    fn three_index_mp_get_link() -> Result<()> {
        let i = Index::new(2);
        let j = Index::new(2);
        let k = Index::new(2);
        let atensor =
            Tensor::<f64>::new(&[&i, &j, &k]).with_data(&[1., 2., 3., 4., 5., 6., 7., 8.])?;
        let atensor_mps = mp_representation(&atensor, Some(1e-5), Some(20))?;
        let link_inds = atensor_mps.link_indices();
        link_inds
            .iter()
            .for_each(|li| assert_eq!(li.tags, "S_Link"));
        Ok(())
    }
}
