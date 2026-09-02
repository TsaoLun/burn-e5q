use crate::tensor::{FloatTensor, IntTensor};
use crate::{Backend, DType, TensorMetadata, get_device_settings};
use alloc::vec;
use burn_std::{FloatDType, IntDType, Shape};

/// ONNX [`DynamicQuantizeLinear`] expanded into existing float/int ops.
///
/// Used by the default [`FloatTensorOps::float_dynamic_quantize_linear`] and as
/// a numerical reference for backends that fuse the same formula.
///
/// [`DynamicQuantizeLinear`]: https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html
pub fn float_dynamic_quantize_linear_expanded<B: Backend>(
    tensor: FloatTensor<B>,
) -> (IntTensor<B>, FloatTensor<B>, IntTensor<B>) {
    let tensor = if tensor.dtype() == DType::F32 {
        tensor
    } else {
        B::float_cast(tensor, FloatDType::F32)
    };

    let device = tensor.device();
    let shape = tensor.shape();
    let rank = shape.num_dims().max(1);
    let bool_dtype = get_device_settings::<B>(&device).bool_dtype;

    let x_min = B::float_min(tensor.clone());
    let x_max = B::float_max(tensor.clone());
    let zero = B::float_zeros(x_min.shape(), &device, FloatDType::F32);

    // min(0, min(x))
    let min_mask = B::float_lower(zero.clone(), x_min.clone(), bool_dtype);
    let x_min_adj = B::float_mask_where(x_min, min_mask, zero.clone());
    // max(0, max(x))
    let max_mask = B::float_lower(x_max.clone(), zero.clone(), bool_dtype);
    let x_max_adj = B::float_mask_where(x_max, max_mask, zero.clone());

    let scale = B::float_div_scalar(B::float_sub(x_max_adj, x_min_adj.clone()), 255f32.into());

    let zp_float = B::float_clamp(
        B::float_round(B::float_div(B::float_sub(zero, x_min_adj), scale.clone())),
        0f32.into(),
        255f32.into(),
    );
    let zp = B::int_cast(
        B::float_into_int(zp_float.clone(), IntDType::I32),
        IntDType::U8,
    );

    let unit = Shape::from(vec![1; rank]);
    let scale_b = B::float_expand(B::float_reshape(scale.clone(), unit.clone()), shape.clone());
    let zp_b = B::float_expand(B::float_reshape(zp_float, unit), shape.clone());

    let y_f = B::float_clamp(
        B::float_add(B::float_round(B::float_div(tensor, scale_b)), zp_b),
        0f32.into(),
        255f32.into(),
    );
    let y = B::int_cast(B::float_into_int(y_f, IntDType::I32), IntDType::U8);

    (y, scale, zp)
}
