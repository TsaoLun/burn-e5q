use std::collections::HashMap;

use crate::ir::{Argument, AttributeValue, NodeType, RawNode};

/// Fuse `Cast(int→float) → Mul(scale) → Add(bias) [→ Gelu]` into
/// [`NodeType::DequantAffine`].
///
/// e5 emits this after every MatMulInteger (72 times): 12 FFN1 paths end in
/// Gelu, the other 60 (FFN2 / QKV / out) stop at Add. Replacing the last
/// node lets codegen call `Tensor::dequant_affine[_gelu]` so flex can do
/// one AVX-512 sweep instead of three or four tensor walks.
///
/// PHASE 4b insert, after `coalesce_gelu` so FFN1 already has a Gelu node.
/// `DequantAffineProcessor::infer_types` is not re-run; output types are
/// copied from the last node (already F32).
pub(crate) fn coalesce_dequant_affine(mut nodes: Vec<RawNode>) -> Vec<RawNode> {
    let consumer = build_consumer_map(&nodes);
    let mut replacements: Vec<(usize, RawNode)> = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        if node.node_type != NodeType::Cast {
            continue;
        }
        if let Some((last_idx, fused)) = try_match(i, &nodes, &consumer) {
            log::info!(
                "Simplification: coalescing Cast→Mul→Add{} into DequantAffine '{}'",
                if fused
                    .attrs
                    .get("apply_gelu")
                    .is_some_and(|v| matches!(v, AttributeValue::Int64(1)))
                {
                    "→Gelu"
                } else {
                    ""
                },
                fused.name
            );
            replacements.push((last_idx, fused));
        }
    }

    for (idx, replacement) in replacements {
        nodes[idx] = replacement;
    }
    nodes
}

fn try_match(
    cast_idx: usize,
    nodes: &[RawNode],
    consumer: &HashMap<String, Vec<usize>>,
) -> Option<(usize, RawNode)> {
    let cast = &nodes[cast_idx];
    if !is_int_to_float_cast(cast) {
        return None;
    }
    if !is_single_use(&cast.outputs[0].name, consumer) {
        return None;
    }

    let mul_idx = *consumer.get(&cast.outputs[0].name)?.first()?;
    let mul = &nodes[mul_idx];
    if mul.node_type != NodeType::Mul || !is_single_use(&mul.outputs[0].name, consumer) {
        return None;
    }
    let scale = other_input(mul, &cast.outputs[0].name)?;
    if !scale.ty.is_on_device() {
        return None;
    }

    let add_idx = *consumer.get(&mul.outputs[0].name)?.first()?;
    let add = &nodes[add_idx];
    if add.node_type != NodeType::Add {
        return None;
    }
    let bias = other_input(add, &mul.outputs[0].name)?;
    if !bias.ty.is_on_device() {
        return None;
    }

    let x = &cast.inputs[0];
    if !x.ty.is_on_device() {
        return None;
    }

    // After coalesce_gelu the last Mul is a Gelu, but the expanded
    // erf/div/mul leftovers still consume Add until DCE. Do not require
    // Add to be single-use: any Gelu consumer is enough. A live second
    // consumer of the pre-GELU value still sees the original Add.
    if let Some(consumers) = consumer.get(&add.outputs[0].name) {
        for &gelu_idx in consumers {
            let gelu = &nodes[gelu_idx];
            if gelu.node_type == NodeType::Gelu
                && gelu
                    .inputs
                    .first()
                    .is_some_and(|a| a.name == add.outputs[0].name)
            {
                return Some((
                    gelu_idx,
                    fused_node(gelu, x, scale, bias, true, &gelu.outputs),
                ));
            }
        }
    }

    Some((
        add_idx,
        fused_node(add, x, scale, bias, false, &add.outputs),
    ))
}

fn fused_node(
    last: &RawNode,
    x: &Argument,
    scale: &Argument,
    bias: &Argument,
    apply_gelu: bool,
    outputs: &[Argument],
) -> RawNode {
    let mut attrs = crate::ir::Attributes::new();
    attrs.insert(
        "apply_gelu".into(),
        AttributeValue::Int64(if apply_gelu { 1 } else { 0 }),
    );
    RawNode {
        custom_identity: None,
        node_type: NodeType::DequantAffine,
        name: format!("{}_dequant_affine", last.name),
        inputs: vec![x.clone(), scale.clone(), bias.clone()],
        outputs: outputs.to_vec(),
        attrs,
    }
}

fn is_int_to_float_cast(node: &RawNode) -> bool {
    if node.node_type != NodeType::Cast || node.inputs.is_empty() || node.outputs.is_empty() {
        return false;
    }
    if !node.inputs[0].ty.is_on_device() || !node.outputs[0].ty.is_on_device() {
        return false;
    }
    let in_dt = node.inputs[0].ty.elem_type();
    let out_dt = node.outputs[0].ty.elem_type();
    (in_dt.is_int() || in_dt.is_uint()) && out_dt.is_float()
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

fn is_single_use(output_name: &str, consumer: &HashMap<String, Vec<usize>>) -> bool {
    consumer
        .get(output_name)
        .is_some_and(|consumers| consumers.len() == 1)
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

    fn tensor(name: &str, dtype: DType, rank: usize) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype,
                rank,
                static_shape: Some(vec![Some(1), Some(8), Some(16)]),
            }),
            value_source: ValueSource::Dynamic,
            value_store: None,
        }
    }

    fn op(name: &str, ty: NodeType, inputs: Vec<Argument>, output: Argument) -> RawNode {
        RawNode {
            custom_identity: None,
            node_type: ty,
            name: name.to_string(),
            inputs,
            outputs: vec![output],
            attrs: Default::default(),
        }
    }

    fn e5_ffn1() -> Vec<RawNode> {
        vec![
            op(
                "cast",
                NodeType::Cast,
                vec![tensor("mmi", DType::I32, 3)],
                tensor("xf", DType::F32, 3),
            ),
            op(
                "mul",
                NodeType::Mul,
                vec![tensor("xf", DType::F32, 3), tensor("scale", DType::F32, 3)],
                tensor("xs", DType::F32, 3),
            ),
            op(
                "add",
                NodeType::Add,
                vec![tensor("xs", DType::F32, 3), tensor("bias", DType::F32, 3)],
                tensor("xb", DType::F32, 3),
            ),
            op(
                "gelu",
                NodeType::Gelu,
                vec![tensor("xb", DType::F32, 3)],
                tensor("y", DType::F32, 3),
            ),
        ]
    }

    fn e5_ffn2() -> Vec<RawNode> {
        vec![
            op(
                "cast",
                NodeType::Cast,
                vec![tensor("mmi", DType::I32, 3)],
                tensor("xf", DType::F32, 3),
            ),
            op(
                "mul",
                NodeType::Mul,
                vec![tensor("scale", DType::F32, 3), tensor("xf", DType::F32, 3)],
                tensor("xs", DType::F32, 3),
            ),
            op(
                "add",
                NodeType::Add,
                vec![tensor("bias", DType::F32, 3), tensor("xs", DType::F32, 3)],
                tensor("y", DType::F32, 3),
            ),
        ]
    }

    #[test]
    fn ffn1_cast_mul_add_gelu_matches() {
        let result = coalesce_dequant_affine(e5_ffn1());
        let fused = result
            .iter()
            .find(|n| n.node_type == NodeType::DequantAffine)
            .expect("fused");
        assert_eq!(fused.inputs[0].name, "mmi");
        assert_eq!(fused.inputs[1].name, "scale");
        assert_eq!(fused.inputs[2].name, "bias");
        assert_eq!(fused.outputs[0].name, "y");
        assert!(matches!(
            fused.attrs.get("apply_gelu"),
            Some(AttributeValue::Int64(1))
        ));
    }

    #[test]
    fn ffn2_cast_mul_add_matches() {
        let result = coalesce_dequant_affine(e5_ffn2());
        let fused = result
            .iter()
            .find(|n| n.node_type == NodeType::DequantAffine)
            .expect("fused");
        assert_eq!(fused.inputs[0].name, "mmi");
        assert_eq!(fused.outputs[0].name, "y");
        assert!(matches!(
            fused.attrs.get("apply_gelu"),
            Some(AttributeValue::Int64(0))
        ));
    }

    #[test]
    fn float_cast_does_not_match() {
        let mut nodes = e5_ffn2();
        nodes[0] = op(
            "cast",
            NodeType::Cast,
            vec![tensor("mmi", DType::F32, 3)],
            tensor("xf", DType::F32, 3),
        );
        let result = coalesce_dequant_affine(nodes);
        assert!(
            !result
                .iter()
                .any(|n| n.node_type == NodeType::DequantAffine)
        );
    }

    #[test]
    fn ffn1_gelu_matches_despite_leftover_erf_consumer() {
        let mut nodes = e5_ffn1();
        nodes.push(op(
            "erf_leftover",
            NodeType::Erf,
            vec![tensor("xb", DType::F32, 3)],
            tensor("dead_erf", DType::F32, 3),
        ));
        let result = coalesce_dequant_affine(nodes);
        let fused = result
            .iter()
            .find(|n| n.node_type == NodeType::DequantAffine)
            .expect("fused");
        assert_eq!(fused.outputs[0].name, "y");
        assert!(matches!(
            fused.attrs.get("apply_gelu"),
            Some(AttributeValue::Int64(1))
        ));
        assert!(result.iter().any(|n| n.name == "erf_leftover"));
    }

    #[test]
    fn extra_cast_consumer_does_not_match() {
        let mut nodes = e5_ffn2();
        nodes.push(op(
            "other",
            NodeType::Relu,
            vec![tensor("xf", DType::F32, 3)],
            tensor("relu", DType::F32, 3),
        ));
        let result = coalesce_dequant_affine(nodes);
        assert!(
            !result
                .iter()
                .any(|n| n.node_type == NodeType::DequantAffine)
        );
    }
}
