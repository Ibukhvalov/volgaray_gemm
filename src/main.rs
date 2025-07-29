mod matrix;
mod metal_context;

use clap::{Arg, command, value_parser};
use log::info;
use matrix::Matrix;
use metal_context::GeMMMetalContext;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::{fs::read, path::PathBuf};

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let matches_result = command!()
        .arg(
            Arg::new("input_tensors_path")
                .long("input")
                .short('i')
                .value_parser(value_parser!(PathBuf))
                .help("Input tensor path relative to src directory")
                .required(true),
        )
        .arg(
            Arg::new("output_tensor_path")
                .long("output")
                .short('o')
                .value_parser(value_parser!(PathBuf))
                .help("Ouput path to write resulted tensor")
                .required(true),
        )
        .get_matches();

    let input_path = matches_result
        .get_one::<PathBuf>("input_tensors_path")
        .unwrap();
    let output_path = matches_result
        .get_one::<PathBuf>("output_tensor_path")
        .unwrap();

    info!("Reading tensors from file");
    let data = read(input_path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    info!("Parsing tensors to matrixes");
    let a_tensor = tensors.tensor("A")?;
    let b_tensor = tensors.tensor("B")?;

    let mut a = Matrix::from_bytes(a_tensor.data(), a_tensor.shape());
    let mut b = Matrix::from_bytes(b_tensor.data(), b_tensor.shape());

    info!("Initialization metal context");
    let mut gemm_core = GeMMMetalContext::gemm_init();

    info!("Binding data");
    gemm_core.bind_data(&mut a, &mut b);

    info!("Computing");
    let start = Instant::now();
    let result_data = unsafe { gemm_core.compute() };
    let duration = start.elapsed();
    println!("{} took {:?}", "Computation", duration);

    let result_shape = vec![a.size.0, b.size.1];
    let result_data = unsafe {
        std::slice::from_raw_parts(
            result_data.as_ptr() as *const u8,
            result_shape.iter().product::<usize>() * std::mem::size_of::<f32>(),
        )
    };
    let tensor_view = TensorView::new(Dtype::F32, result_shape, result_data)?;

    safetensors::serialize_to_file(
        [("C".to_string(), tensor_view)].into_iter(),
        None,
        &output_path,
    )?;

    Ok(())
}
