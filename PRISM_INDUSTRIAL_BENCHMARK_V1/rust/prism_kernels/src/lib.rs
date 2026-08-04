use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn block_means_prefix<'py>(
    py: Python<'py>,
    origins: PyReadonlyArray1<'py, i64>,
    dense_min: i64,
    value_prefix: PyReadonlyArray1<'py, f64>,
    count_prefix: PyReadonlyArray1<'py, i64>,
    intervals: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let origins = origins.as_slice()?;
    let value_prefix = value_prefix.as_slice()?;
    let count_prefix = count_prefix.as_slice()?;
    let intervals = intervals.as_array();
    if value_prefix.len() != count_prefix.len() || intervals.ncols() != 2 {
        return Err(PyValueError::new_err("invalid prefix or interval shape"));
    }
    let width = intervals.nrows();
    let pairs: Vec<(i64, i64)> = intervals
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect();
    let mut result = vec![0.0_f64; origins.len() * width];
    result
        .par_chunks_mut(width)
        .zip(origins.par_iter())
        .try_for_each(|(row, origin)| -> Result<(), &'static str> {
            for (index, (near, far)) in pairs.iter().copied().enumerate() {
                let start = origin - far - dense_min;
                let stop = origin - near - dense_min;
                if start < 0 || stop < start || stop as usize >= value_prefix.len() {
                    return Err("block outside entity support");
                }
                let start = start as usize;
                let stop = stop as usize;
                let expected = far - near;
                if count_prefix[stop] - count_prefix[start] != expected {
                    return Err("block crosses missing rows");
                }
                row[index] = (value_prefix[stop] - value_prefix[start]) / expected as f64;
            }
            Ok(())
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyArray2::from_vec2_bound(
        py,
        &result.chunks(width).map(|row| row.to_vec()).collect::<Vec<_>>(),
    )?)
}

#[pymodule]
fn _prism_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(block_means_prefix, module)?)?;
    Ok(())
}
