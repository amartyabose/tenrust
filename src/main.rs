use tenrust::*;

fn main() -> tenrust::error::Result<()> {
    let i = index::Index::new(2).add_tag("i");
    let j = index::Index::new(2).add_tag("j");
    // let k = index::Index::new(2).add_tag("k");
    let atensor = tensor::Tensor::new(&[&i, &j]).with_data(&[1., 2., 3., 4.])?;
    println!("{:#?}", &atensor);

    //    let mut atensor = tensor::Tensor::new(vec![&i, &j]);
    //    let mut val = 1f64;
    //    for i_ind in 0..2 {
    //        for j_ind in 0..2 {
    //            atensor.set_elem(
    //                vec![
    //                    index::IndexVal::new(&i, i_ind)?,
    //                    index::IndexVal::new(&j, j_ind)?,
    //                ],
    //                val,
    //            )?;
    //            val += 3f64;
    //        }
    //    }
    //
    //    val = 2f64;
    //    let mut btensor = tensor::Tensor::new(vec![&k, &j]);
    //    for i_ind in 0..2 {
    //        for j_ind in 0..2 {
    //            btensor.set_elem(
    //                vec![
    //                    index::IndexVal::new(&k, i_ind)?,
    //                    index::IndexVal::new(&j, j_ind)?,
    //                ],
    //                val,
    //            )?;
    //            val += 2f64;
    //        }
    //    }
    //    let ctensor = &atensor * &btensor;
    //    println!("A = {:?}", atensor.data);
    //    println!("B = {:?}", btensor.data);
    //    println!("C = {:?}", ctensor.data);
    //
    //    let ind = index::Index::new(3).add_tag("i");
    //    let vector = tensor::Tensor::new(vec![&ind]).with_data(vec![1.0, 2.0, 3.0])?;
    //    println!("vector = {:?}", &vector);
    //    println!("dot product = {:?}", &vector * &vector);

    Ok(())
}
