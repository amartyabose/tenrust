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
fn modify_tensor_elements() -> Result<()> {
    let i = Index::new(2);
    let j = Index::new(3);
    let mut atensor = Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
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
fn svd_tenrust() -> Result<()> {
    let i = Index::new(2);
    let j = Index::new(4);
    let atensor =
        Tensor::<f64>::new(&[&i, &j]).with_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    let TensorSVD {
        utensor,
        stensor,
        v_trans_tensor: vtensor,
    } = atensor.svd(&[&i])?;
    let ans = utensor.dot(&stensor.dot(&vtensor)?)?;
    assert_close_l2!(&atensor.data, &ans.data, 1e-12);
    Ok(())
}
