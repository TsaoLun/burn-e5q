use std::collections::HashMap;

use crate::ir::{Argument, NodeType, RawNode, TensorDataExt};

/// ONNX Gelu (opset 20, `approximate=none`) is
/// `0.5 * x * (1 + erf(x / √2))`.
///
/// Legacy exports (e5 qint8, opset 11) emit that formula as Div/Mul + Erf +
/// Add + Mul + Mul. Replace the last Mul with a Gelu node so codegen calls
/// `burn::tensor::activation::gelu` (one pass) instead of five tensor walks.
///
/// Inserted in PHASE 4b; `GeluProcessor::infer_types` (opset 20) is not
/// re-run. `build_node` does not check opset.
pub(crate) fn coalesce_gelu(mut nodes: Vec<RawNode>) -> Vec<RawNode> {
    let producer = build_producer_map(&nodes);
    let consumer = build_consumer_map(&nodes);
    let mut replacements: Vec<(usize, RawNode)> = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        if node.node_type != NodeType::Erf {
            continue;
        }
        if let Some((last_idx, gelu)) = try_match_gelu(i, &nodes, &producer, &consumer) {
            log::info!(
                "Simplification: coalescing expanded GELU into Gelu node '{}'",
                gelu.name
            );
            replacements.push((last_idx, gelu));
        }
    }

    for (idx, replacement) in replacements {
        nodes[idx] = replacement;
    }
    nodes
}

fn try_match_gelu(
    erf_idx: usize,
    nodes: &[RawNode],
    producer: &HashMap<String, usize>,
    consumer: &HashMap<String, Vec<usize>>,
) -> Option<(usize, RawNode)> {
    let erf = &nodes[erf_idx];
    if !is_single_use(&erf.outputs[0].name, consumer) {
        return None;
    }

    let scale_idx = *producer.get(&erf.inputs[0].name)?;
    let scale_node = &nodes[scale_idx];
    if !is_single_use(&scale_node.outputs[0].name, consumer) {
        return None;
    }
    let (x_arg, scale) = split_tensor_and_const(scale_node, nodes, producer)?;
    match scale_node.node_type {
        NodeType::Div if approx(scale, std::f64::consts::SQRT_2) => {}
        NodeType::Mul if approx(scale, std::f64::consts::FRAC_1_SQRT_2) => {}
        _ => return None,
    }

    let add_idx = *consumer.get(&erf.outputs[0].name)?.first()?;
    let add = &nodes[add_idx];
    if add.node_type != NodeType::Add || !is_single_use(&add.outputs[0].name, consumer) {
        return None;
    }
    let add_one = other_input(add, &erf.outputs[0].name)?;
    if !approx(const_scalar(add_one, nodes, producer)?, 1.0) {
        return None;
    }

    let mul_x_idx = *consumer.get(&add.outputs[0].name)?.first()?;
    let mul_x = &nodes[mul_x_idx];
    if mul_x.node_type != NodeType::Mul {
        return None;
    }
    let x_side = other_input(mul_x, &add.outputs[0].name)?;
    let (x_core, half_on_x) = peel_half(x_side, nodes, producer, consumer);

    if !same_value(x_core, x_arg) {
        return None;
    }

    if half_on_x {
        if !is_single_use(&mul_x.outputs[0].name, consumer)
            && consumer
                .get(&mul_x.outputs[0].name)
                .is_some_and(|c| !c.is_empty())
        {
            // Output may be graph-visible; still a valid fuse point.
        }
        return Some((mul_x_idx, gelu_node(mul_x, x_arg)));
    }

    if !is_single_use(&mul_x.outputs[0].name, consumer) {
        return None;
    }
    let half_idx = *consumer.get(&mul_x.outputs[0].name)?.first()?;
    let half = &nodes[half_idx];
    if half.node_type != NodeType::Mul {
        return None;
    }
    let half_c = other_input(half, &mul_x.outputs[0].name)?;
    if !approx(const_scalar(half_c, nodes, producer)?, 0.5) {
        return None;
    }
    Some((half_idx, gelu_node(half, x_arg)))
}

fn gelu_node(last: &RawNode, x: &Argument) -> RawNode {
    RawNode {
        custom_identity: None,
        node_type: NodeType::Gelu,
        name: format!("{}_gelu", last.name),
        inputs: vec![x.clone()],
        outputs: last.outputs.clone(),
        attrs: Default::default(),
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

/// If `arg` is `Mul(x, 0.5)` (single-use), return `(x, true)`.
fn peel_half<'a>(
    arg: &'a Argument,
    nodes: &'a [RawNode],
    producer: &HashMap<String, usize>,
    consumer: &HashMap<String, Vec<usize>>,
) -> (&'a Argument, bool) {
    let Some(&idx) = producer.get(&arg.name) else {
        return (arg, false);
    };
    let node = &nodes[idx];
    if node.node_type != NodeType::Mul || !is_single_use(&node.outputs[0].name, consumer) {
        return (arg, false);
    }
    if let Some(v) = const_scalar(&node.inputs[1], nodes, producer)
        && approx(v, 0.5)
    {
        return (&node.inputs[0], true);
    }
    if let Some(v) = const_scalar(&node.inputs[0], nodes, producer)
        && approx(v, 0.5)
    {
        return (&node.inputs[1], true);
    }
    (arg, false)
}

fn same_value(a: &Argument, b: &Argument) -> bool {
    a.name == b.name
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
    use crate::ir::{ArgType, AttributeValue, DType, TensorType, ValueSource};
    use crate::tensor_store::{TensorDataRef, TensorStore, ValueStore};

    fn tensor3(name: &str) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 3,
                static_shape: Some(vec![Some(1), Some(8), Some(16)]),
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
        let value_store = ValueStore::new(
            std::sync::Arc::new(store),
            std::sync::Arc::new(constant_map),
        );
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 0,
                static_shape: Some(vec![]),
            }),
            value_source: ValueSource::Constant,
            value_store: Some(value_store),
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

    fn e5_gelu() -> Vec<RawNode> {
        vec![
            op(
                "div",
                NodeType::Div,
                vec![tensor3("x"), const_f32("sqrt2", std::f32::consts::SQRT_2)],
                "x_scaled",
            ),
            op("erf", NodeType::Erf, vec![tensor3("x_scaled")], "erf"),
            op(
                "add1",
                NodeType::Add,
                vec![tensor3("erf"), const_f32("one", 1.0)],
                "one_erf",
            ),
            op(
                "mul_x",
                NodeType::Mul,
                vec![tensor3("x"), tensor3("one_erf")],
                "x_erf",
            ),
            op(
                "mul_half",
                NodeType::Mul,
                vec![tensor3("x_erf"), const_f32("half", 0.5)],
                "y",
            ),
        ]
    }

    #[test]
    fn e5_div_erf_path_matches() {
        let result = coalesce_gelu(e5_gelu());
        let gelu = result
            .iter()
            .find(|n| n.node_type == NodeType::Gelu)
            .expect("gelu");
        assert_eq!(gelu.inputs[0].name, "x");
        assert_eq!(gelu.outputs[0].name, "y");
    }

    #[test]
    fn mul_inv_sqrt2_matches() {
        let mut nodes = e5_gelu();
        nodes[0] = op(
            "mul",
            NodeType::Mul,
            vec![
                tensor3("x"),
                const_f32("isq2", std::f32::consts::FRAC_1_SQRT_2),
            ],
            "x_scaled",
        );
        let result = coalesce_gelu(nodes);
        assert!(result.iter().any(|n| n.node_type == NodeType::Gelu));
    }

    #[test]
    fn half_on_x_matches() {
        let nodes = vec![
            op(
                "div",
                NodeType::Div,
                vec![tensor3("x"), const_f32("sqrt2", std::f32::consts::SQRT_2)],
                "x_scaled",
            ),
            op("erf", NodeType::Erf, vec![tensor3("x_scaled")], "erf"),
            op(
                "add1",
                NodeType::Add,
                vec![tensor3("erf"), const_f32("one", 1.0)],
                "one_erf",
            ),
            op(
                "half_x",
                NodeType::Mul,
                vec![tensor3("x"), const_f32("half", 0.5)],
                "xh",
            ),
            op(
                "mul_x",
                NodeType::Mul,
                vec![tensor3("xh"), tensor3("one_erf")],
                "y",
            ),
        ];
        let result = coalesce_gelu(nodes);
        let gelu = result
            .iter()
            .find(|n| n.node_type == NodeType::Gelu)
            .expect("gelu");
        assert_eq!(gelu.inputs[0].name, "x");
        assert_eq!(gelu.outputs[0].name, "y");
    }

    #[test]
    fn wrong_scale_does_not_match() {
        let mut nodes = e5_gelu();
        nodes[0] = op(
            "div",
            NodeType::Div,
            vec![tensor3("x"), const_f32("three", 3.0)],
            "x_scaled",
        );
        let result = coalesce_gelu(nodes);
        assert!(!result.iter().any(|n| n.node_type == NodeType::Gelu));
    }

    #[test]
    fn erf_with_extra_consumer_does_not_match() {
        let mut nodes = e5_gelu();
        nodes.push(op(
            "other",
            NodeType::Relu,
            vec![tensor3("erf")],
            "relu_erf",
        ));
        let result = coalesce_gelu(nodes);
        assert!(!result.iter().any(|n| n.node_type == NodeType::Gelu));
    }

    #[test]
    fn unsqueeze_const_still_matches() {
        let unsqueeze = RawNode {
            custom_identity: None,
            node_type: NodeType::Unsqueeze,
            name: "unsq".into(),
            inputs: vec![const_f32("sqrt2", std::f32::consts::SQRT_2)],
            outputs: vec![tensor3("sqrt2_u")],
            attrs: [("axes".into(), AttributeValue::Int64s(vec![0, 1]))]
                .into_iter()
                .collect(),
        };
        let mut nodes = e5_gelu();
        nodes[0] = op(
            "div",
            NodeType::Div,
            vec![tensor3("x"), tensor3("sqrt2_u")],
            "x_scaled",
        );
        nodes.insert(0, unsqueeze);
        let result = coalesce_gelu(nodes);
        assert!(result.iter().any(|n| n.node_type == NodeType::Gelu));
    }
}
