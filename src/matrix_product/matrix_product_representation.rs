use crate::error::*;
use crate::linalg::singular_value::*;
use crate::tensor::*;

use ndarray_linalg::Lapack;
use num::Num;

#[derive(Debug)]
pub struct MatrixProduct<T: 'static + Clone + Copy + Num> {
    pub tensors: Vec<Tensor<T>>,
}

impl<T: 'static + Clone + Copy + Num> MatrixProduct<T> {
    pub fn get_tensor(&self) -> Result<Tensor<T>> {
        let mut tens = self.tensors.first().cloned().unwrap();
        for tensor in (&self.tensors).into_iter().skip(1) {
            tens = tens.dot(tensor)?;
        }
        Ok(tens)
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
            dbg!(&vnew.indices[0]);
            let TensorSVD {
                utensor,
                stensor,
                v_trans_tensor: vtensor,
            } = vnew.svd(SVDIndices::Rows(&[&vnew.indices[0]]), cutoff, maxdim)?;
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
}
