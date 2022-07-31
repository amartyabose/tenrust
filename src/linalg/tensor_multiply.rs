use crate::error::*;
use crate::tensor::*;

use num::Num;

impl<T: 'static + Clone + Copy + Num> Tensor<T> {
    pub fn dot(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        let common_inds = common_indices(self, other);
        let self_extra = if common_inds.is_empty() {
            self.indices.to_vec()
        } else {
            self.indices_set
                .difference(&other.indices_set)
                .cloned()
                .collect::<Vec<_>>()
        };
        let other_extra = if common_inds.is_empty() {
            other.indices.to_vec()
        } else {
            other
                .indices_set
                .difference(&self.indices_set)
                .cloned()
                .collect::<Vec<_>>()
        };

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

#[cfg(test)]
mod tensor_multiply_tests {
    use super::*;
    use crate::index::*;

    use ndarray::*;
    use ndarray_linalg::assert_close_l2;

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
        assert_eq!(&atensor.indices, &vec![i.clone(), j.clone()]);
        assert_eq!(&btensor.indices, &vec![j, k.clone()]);
        assert_eq!(&ctensor.indices, &vec![i, k]);
        Ok(())
    }

    #[test]
    fn trace() -> Result<()> {
        let i = Index::new(2);
        let j = Index::new(2);
        let atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0])?;
        let delta = Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 0.0, 0.0, 1.0])?;
        let trace = atensor.dot(&delta)?.as_scalar()?;
        approx::assert_abs_diff_eq!(trace, 5.0, epsilon = 1e-10);
        Ok(())
    }

    #[test]
    fn trace_of_product() -> Result<()> {
        let i = Index::new(2);
        let j = Index::new(2);
        let atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0])?;
        let btensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0])?;
        let trace = atensor.dot(&btensor)?.as_scalar()?;
        approx::assert_abs_diff_eq!(trace, 30.0, epsilon = 1e-10);
        Ok(())
    }

    #[test]
    fn noncontracting_product() -> Result<()> {
        let i = Index::new(2).add_tag("i");
        let j = Index::new(2).add_tag("j");
        let k = Index::new(2).add_tag("k");
        let l = Index::new(2).add_tag("l");
        let atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0])?;
        let btensor = Tensor::<f64>::new(&[&k, &l]).with_data(&[1.0, 2.0, 3.0, 4.0])?;
        let ctensor = atensor.dot(&btensor)?;
        assert_eq!(ctensor.indices.len(), 4);
        dbg!(&ctensor);
        approx::assert_abs_diff_eq!(
            ctensor[&[
                IndexVal::new(&i, 1)?,
                IndexVal::new(&j, 0)?,
                IndexVal::new(&k, 0)?,
                IndexVal::new(&l, 1)?
            ]],
            6.0,
            epsilon = 1e-10
        );
        Ok(())
    }
}
