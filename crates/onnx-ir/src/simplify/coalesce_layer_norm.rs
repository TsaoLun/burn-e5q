use std::collections::HashMap;

use crate::ir::{ArgType, Argument, AttributeValue, NodeType, RawNode, TensorDataExt, TensorType};

/// ONNX LayerNormalization (opset 17) on the last axis:
///
/// ```text
/// mean = ReduceMean(x, axes=[-1], keepdims=1)
/// centered = x - mean
/// var = ReduceMean(centered^2, axes=[-1], keepdims=1)
/// y = (centered / sqrt(var + eps)) * gamma + beta
/// ```
///
/// e5 (opset 11) emits this as ReduceMean/Sub/Pow/Add/Sqrt/Div/Mul/Add.
/// Replace the last Add with LayerNormalization so codegen uses flex's
/// SIMD `layer_norm` instead of nine tensor walks.
///
/// PHASE 4b insert; `LayerNormProcessor::infer_types` (opset 17) is not
/// re-run. `build_node` only calls `extract_config`.
pub(crate) fn coalesce_layer_norm(mut nodes: Vec<RawNode>) -> Vec<RawNode> {
    let producer = build_producer_map(&nodes);
    let consumer = build_consumer_map(&nodes);
    let mut replacements: Vec<(usize, RawNode)> = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        if node.node_type != NodeType::Sqrt {
            continue;
        }
        if let Some((last_idx, ln)) = try_match_ln(i, &nodes, &producer, &consumer) {
            log::info!(
                "Simplification: coalescing expanded LayerNorm into '{}' (eps={:?})",
                ln.name,
                ln.attrs.get("epsilon")
            );
            replacements.push((last_idx, ln));
        }
    }

    for (idx, replacement) in replacements {
        nodes[idx] = replacement;
    }
    nodes
}

fn try_match_ln(
    sqrt_idx: usize,
    nodes: &[RawNode],
    producer: &HashMap<String, usize>,
    consumer: &HashMap<String, Vec<usize>>,
) -> Option<(usize, RawNode)> {
    let sqrt = &nodes[sqrt_idx];
    if !is_single_use(&sqrt.outputs[0].name, consumer) {
        return None;
    }

    // sqrt(var + eps) or sqrt(var)
    let (var_arg, eps) = match producer.get(&sqrt.inputs[0].name) {
        Some(&add_idx) if nodes[add_idx].node_type == NodeType::Add => {
            let add = &nodes[add_idx];
            if !is_single_use(&add.outputs[0].name, consumer) {
                return None;
            }
            let (var, c) = split_tensor_and_const(add, nodes, producer)?;
            if !(1e-12..=1e-3).contains(&c) {
                return None;
            }
            (var, c)
        }
        _ => return None,
    };

    let var_mean_idx = *producer.get(&var_arg.name)?;
    let var_mean = &nodes[var_mean_idx];
    if !is_last_axis_reducemean(var_mean) || !is_single_use(&var_mean.outputs[0].name, consumer) {
        return None;
    }

    let sq_arg = &var_mean.inputs[0];
    let sq_idx = *producer.get(&sq_arg.name)?;
    let sq = &nodes[sq_idx];
    let centered = square_input(sq, nodes, producer, consumer)?;

    let sub_idx = *producer.get(&centered.name)?;
    let sub = &nodes[sub_idx];
    if sub.node_type != NodeType::Sub {
        return None;
    }
    let x = &sub.inputs[0];
    let mean_arg = &sub.inputs[1];
    let mean_idx = *producer.get(&mean_arg.name)?;
    let mean = &nodes[mean_idx];
    if !is_last_axis_reducemean(mean) || !is_single_use(&mean.outputs[0].name, consumer) {
        return None;
    }
    if mean.inputs[0].name != x.name {
        return None;
    }

    // centered / std
    let div_idx = *consumer.get(&sqrt.outputs[0].name)?.first()?;
    let div = &nodes[div_idx];
    if div.node_type != NodeType::Div || !is_single_use(&div.outputs[0].name, consumer) {
        return None;
    }
    if div.inputs[0].name != centered.name || div.inputs[1].name != sqrt.outputs[0].name {
        return None;
    }

    // * gamma
    let mul_idx = *consumer.get(&div.outputs[0].name)?.first()?;
    let mul = &nodes[mul_idx];
    if mul.node_type != NodeType::Mul || !is_single_use(&mul.outputs[0].name, consumer) {
        return None;
    }
    let gamma_raw = other_input(mul, &div.outputs[0].name)?;
    let gamma = peel_initializer(gamma_raw, nodes, producer)?;
    if !is_rank1_param(gamma) {
        return None;
    }

    // + beta
    let add_idx = *consumer.get(&mul.outputs[0].name)?.first()?;
    let add_b = &nodes[add_idx];
    if add_b.node_type != NodeType::Add {
        return None;
    }
    let beta_raw = other_input(add_b, &mul.outputs[0].name)?;
    let beta = peel_initializer(beta_raw, nodes, producer)?;
    if !is_rank1_param(beta) {
        return None;
    }

    Some((add_idx, ln_node(add_b, x, gamma, beta, eps)))
}

fn ln_node(last: &RawNode, x: &Argument, gamma: &Argument, beta: &Argument, eps: f64) -> RawNode {
    let mut attrs = crate::ir::Attributes::new();
    attrs.insert("epsilon".into(), AttributeValue::Float32(eps as f32));
    attrs.insert("axis".into(), AttributeValue::Int64(-1));
    // stash_type=0: skip the extra f32 cast LayerNorm codegen emits for 1.
    attrs.insert("stash_type".into(), AttributeValue::Int64(0));
    RawNode {
        custom_identity: None,
        node_type: NodeType::LayerNormalization,
        name: format!("{}_ln", last.name),
        inputs: vec![
            x.clone(),
            with_static_shape(gamma),
            with_static_shape(beta),
        ],
        outputs: last.outputs.clone(),
        attrs,
    }
}

/// `LayerNorm` codegen reads `static_shape_known()` for `d_model`. Constants
/// inserted by this pass (or peeled Unsqueeze) may only have the shape on
/// `value()`, so copy it onto `ty` before `build_node`.
fn with_static_shape(arg: &Argument) -> Argument {
    if arg.ty.static_shape_known().is_some() {
        return arg.clone();
    }
    let Some(data) = arg.value() else {
        return arg.clone();
    };
    let dims: Vec<usize> = data.shape.to_vec();
    let mut out = arg.clone();
    if let ArgType::Tensor(t) = &mut out.ty {
        *t = TensorType::new_known(t.dtype, dims);
    }
    out
}

fn square_input<'a>(
    sq: &'a RawNode,
    nodes: &'a [RawNode],
    producer: &HashMap<String, usize>,
    consumer: &HashMap<String, Vec<usize>>,
) -> Option<&'a Argument> {
    if !is_single_use(&sq.outputs[0].name, consumer) {
        return None;
    }
    match sq.node_type {
        NodeType::Pow => {
            let exp = const_scalar(&sq.inputs[1], nodes, producer)?;
            if !approx(exp, 2.0) {
                return None;
            }
            Some(&sq.inputs[0])
        }
        NodeType::Mul if sq.inputs[0].name == sq.inputs[1].name => Some(&sq.inputs[0]),
        _ => None,
    }
}

fn is_last_axis_reducemean(node: &RawNode) -> bool {
    if node.node_type != NodeType::ReduceMean {
        return false;
    }
    let rank = node.inputs[0].ty.rank();
    if rank < 1 {
        return false;
    }
    let keepdims = node
        .attrs
        .get("keepdims")
        .map(|v| v.clone().into_i64())
        .unwrap_or(1);
    if keepdims != 1 {
        return false;
    }
    let axes = if let Some(attr) = node.attrs.get("axes") {
        match attr {
            AttributeValue::Int64s(v) => v.clone(),
            AttributeValue::Int64(v) => vec![*v],
            _ => return false,
        }
    } else if let Some(arg) = node.inputs.get(1) {
        match arg.value().and_then(|d| d.try_into_vec::<i64>().ok()) {
            Some(v) => v,
            None => return false,
        }
    } else {
        return false;
    };
    if axes.len() != 1 {
        return false;
    }
    let ax = axes[0];
    ax == -1 || ax == rank as i64 - 1
}

fn is_rank1_param(arg: &Argument) -> bool {
    arg.value().is_some() && arg.ty.rank() == 1
}

fn peel_initializer<'a>(
    arg: &'a Argument,
    nodes: &'a [RawNode],
    producer: &HashMap<String, usize>,
) -> Option<&'a Argument> {
    if arg.value().is_some() {
        return Some(arg);
    }
    let idx = *producer.get(&arg.name)?;
    let node = &nodes[idx];
    match node.node_type {
        NodeType::Unsqueeze | NodeType::Squeeze | NodeType::Identity | NodeType::Reshape => {
            peel_initializer(&node.inputs[0], nodes, producer)
        }
        NodeType::Constant => Some(&node.outputs[0]),
        _ => None,
    }
}

fn split_tensor_and_const<'a>(
    node: &'a RawNode,
    nodes: &'a [RawNode],
    producer: &HashMap<String, usize>,
) -> Option<(&'a Argument, f64)> {
    if node.inputs.len() != 2 {
        return None;
    }
    if let Some(v) = const_scalar(&node.inputs[1], nodes, producer) {
        return Some((&node.inputs[0], v));
    }
    if let Some(v) = const_scalar(&node.inputs[0], nodes, producer) {
        return Some((&node.inputs[1], v));
    }
    None
}

fn other_input<'a>(node: &'a RawNode, known: &str) -> Option<&'a Argument> {
    if node.inputs[0].name == known {
        node.inputs.get(1)
    } else if node.inputs.get(1).is_some_and(|a| a.name == known) {
        Some(&node.inputs[0])
    } else {
        None
    }
}

fn const_scalar(
    arg: &Argument,
    nodes: &[RawNode],
    producer: &HashMap<String, usize>,
) -> Option<f64> {
    if let Some(v) = arg.value().and_then(|d| d.scalar_f64().ok()) {
        return Some(v);
    }
    let idx = *producer.get(&arg.name)?;
    let node = &nodes[idx];
    match node.node_type {
        NodeType::Unsqueeze | NodeType::Squeeze | NodeType::Identity | NodeType::Reshape => {
            const_scalar(&node.inputs[0], nodes, producer)
        }
        NodeType::Constant => node.outputs[0].value().and_then(|d| d.scalar_f64().ok()),
        _ => None,
    }
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-4 + 1e-4 * b.abs()
}

fn is_single_use(output_name: &str, consumer: &HashMap<String, Vec<usize>>) -> bool {
    consumer
        .get(output_name)
        .is_some_and(|consumers| consumers.len() == 1)
}

fn build_producer_map(nodes: &[RawNode]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out in &node.outputs {
            map.insert(out.name.clone(), i);
        }
    }
    map
}

fn build_consumer_map(nodes: &[RawNode]) -> HashMap<String, Vec<usize>> {
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for inp in &node.inputs {
            map.entry(inp.name.clone()).or_default().push(i);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, DType, TensorType, ValueSource};
    use crate::tensor_store::{TensorDataRef, TensorStore, ValueStore};

    fn tensor3(name: &str) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 3,
                static_shape: Some(vec![Some(1), Some(8), Some(4)]),
            }),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    fn const_f32(name: &str, value: f32) -> Argument {
        let bytes = bytes::Bytes::copy_from_slice(&value.to_ne_bytes());
        let data_ref = TensorDataRef::new(bytes, vec![1], DType::F32);
        let mut store = TensorStore::new();
        let id = store.store(data_ref);
        let mut constant_map = std::collections::HashMap::new();
        constant_map.insert(name.to_string(), id);
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 0,
                static_shape: Some(vec![]),
            }),
            value_source: ValueSource::Constant,
            value_store: Some(ValueStore::new(
                std::sync::Arc::new(store),
                std::sync::Arc::new(constant_map),
            )),
        }
    }

    fn const_vec(name: &str, values: &[f32]) -> Argument {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        let data_ref = TensorDataRef::new(bytes::Bytes::from(bytes), vec![values.len()], DType::F32);
        let mut store = TensorStore::new();
        let id = store.store(data_ref);
        let mut constant_map = std::collections::HashMap::new();
        constant_map.insert(name.to_string(), id);
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 1,
                static_shape: Some(vec![Some(values.len())]),
            }),
            value_source: ValueSource::Constant,
            value_store: Some(ValueStore::new(
                std::sync::Arc::new(store),
                std::sync::Arc::new(constant_map),
            )),
        }
    }

    fn reduce_mean(name: &str, input: &str, output: &str) -> RawNode {
        RawNode {
            custom_identity: None,
            node_type: NodeType::ReduceMean,
            name: name.to_string(),
            inputs: vec![tensor3(input)],
            outputs: vec![tensor3(output)],
            attrs: [
                ("axes".into(), AttributeValue::Int64s(vec![-1])),
                ("keepdims".into(), AttributeValue::Int64(1)),
            ]
            .into_iter()
            .collect(),
        }
    }

    fn op(name: &str, ty: NodeType, inputs: Vec<Argument>, output: &str) -> RawNode {
        RawNode {
            custom_identity: None,
            node_type: ty,
            name: name.to_string(),
            inputs,
            outputs: vec![tensor3(output)],
            attrs: Default::default(),
        }
    }

    fn e5_ln() -> Vec<RawNode> {
        vec![
            reduce_mean("mean", "x", "mean"),
            op(
                "sub",
                NodeType::Sub,
                vec![tensor3("x"), tensor3("mean")],
                "centered",
            ),
            op(
                "pow",
                NodeType::Pow,
                vec![tensor3("centered"), const_f32("two", 2.0)],
                "sq",
            ),
            reduce_mean("var", "sq", "var"),
            op(
                "add_eps",
                NodeType::Add,
                vec![tensor3("var"), const_f32("eps", 1e-5)],
                "var_eps",
            ),
            op("sqrt", NodeType::Sqrt, vec![tensor3("var_eps")], "std"),
            op(
                "div",
                NodeType::Div,
                vec![tensor3("centered"), tensor3("std")],
                "norm",
            ),
            op(
                "mul_g",
                NodeType::Mul,
                vec![tensor3("norm"), const_vec("gamma", &[1.0, 1.0, 1.0, 1.0])],
                "scaled",
            ),
            op(
                "add_b",
                NodeType::Add,
                vec![tensor3("scaled"), const_vec("beta", &[0.0, 0.0, 0.0, 0.0])],
                "y",
            ),
        ]
    }

    #[test]
    fn e5_pattern_matches() {
        let result = coalesce_layer_norm(e5_ln());
        let ln = result
            .iter()
            .find(|n| n.node_type == NodeType::LayerNormalization)
            .expect("ln");
        assert_eq!(ln.inputs[0].name, "x");
        assert_eq!(ln.inputs[1].name, "gamma");
        assert_eq!(ln.inputs[2].name, "beta");
        assert_eq!(ln.outputs[0].name, "y");
        match ln.attrs.get("epsilon") {
            Some(AttributeValue::Float32(e)) => assert!(((*e as f64) - 1e-5).abs() < 1e-8),
            other => panic!("bad epsilon {other:?}"),
        }
    }

    #[test]
    fn square_via_mul_matches() {
        let mut nodes = e5_ln();
        nodes[2] = op(
            "sq",
            NodeType::Mul,
            vec![tensor3("centered"), tensor3("centered")],
            "sq",
        );
        assert!(
            coalesce_layer_norm(nodes)
                .iter()
                .any(|n| n.node_type == NodeType::LayerNormalization)
        );
    }

    #[test]
    fn wrong_axis_does_not_match() {
        let mut nodes = e5_ln();
        nodes[0].attrs.insert("axes".into(), AttributeValue::Int64s(vec![0]));
        let result = coalesce_layer_norm(nodes);
        assert!(!result
            .iter()
            .any(|n| n.node_type == NodeType::LayerNormalization));
    }

    #[test]
    fn extra_sqrt_consumer_does_not_match() {
        let mut nodes = e5_ln();
        nodes.push(op(
            "other",
            NodeType::Relu,
            vec![tensor3("std")],
            "relu_std",
        ));
        let result = coalesce_layer_norm(nodes);
        assert!(!result
            .iter()
            .any(|n| n.node_type == NodeType::LayerNormalization));
    }
}
