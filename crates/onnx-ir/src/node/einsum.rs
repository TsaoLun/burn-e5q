//! # Einsum
//!
//! 2-input einsum support for patterns that lower cleanly to Burn tensor operations.
//!
//! Supported examples:
//! - `ij,jk->ik` (explicit form)
//! - `ij,jk` (implicit form, output inferred alphabetically)
//! - `bij,bjk->bik`
//! - `bhwc,hkc->bhwk`
//! - `i,j->ij`
//! - `,ij->ij`
//! - `ij,->ij`
//! - `ij,kl->il` (one-sided reduction)
//!
//! - `...ij,...jk->...ik` (ellipsis broadcasting)
//!
//! Unsupported:
//! - more than 2 inputs
//! - repeated labels within one term
//!
//! ONNX spec: <https://onnx.ai/onnx/operators/onnx__Einsum.html>

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// ONNX attributes for `Einsum`.
#[derive(Debug, Clone, Default)]
pub struct EinsumConfig {
    /// Equation string such as `bhwc,hkc->bhwk`.
    pub equation: String,
}

/// IR node for `Einsum`.
#[derive(Debug, Clone, NodeBuilder)]
pub struct EinsumNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: EinsumConfig,
}

pub(crate) struct EinsumProcessor;

/// Einsum operands can arrive as regular tensors or rank-0 scalar values.
///
/// ONNX rank-0 values may be represented in the IR as `ScalarNative`,
/// `ScalarTensor`, or a rank-0 `Tensor`, but einsum lowering only cares about
/// the effective rank, dtype, and any known per-axis dimensions.
#[derive(Clone, Copy)]
struct EinsumOperand<'a> {
    dtype: crate::ir::DType,
    rank: usize,
    static_shape: Option<&'a [Option<usize>]>,
}

impl<'a> EinsumOperand<'a> {
    fn from_arg(arg: &'a ArgType) -> Result<Self, ProcessError> {
        match arg {
            ArgType::Tensor(tensor) => Ok(Self {
                dtype: tensor.dtype,
                rank: tensor.rank,
                static_shape: tensor.static_shape.as_deref(),
            }),
            ArgType::ScalarNative(dtype) | ArgType::ScalarTensor(dtype) => Ok(Self {
                dtype: *dtype,
                rank: 0,
                static_shape: Some(&[]),
            }),
            _ => Err(ProcessError::TypeMismatch {
                expected: "Tensor or scalar".to_string(),
                actual: format!("{arg:?}"),
            }),
        }
    }
}

impl NodeProcessor for EinsumProcessor {
    type Config = EinsumConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 12,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let equation = node
            .attrs
            .get("equation")
            .ok_or_else(|| ProcessError::MissingAttribute("equation".to_string()))?
            .clone()
            .into_string();

        let lhs = EinsumOperand::from_arg(&node.inputs[0].ty)?;
        let rhs = EinsumOperand::from_arg(&node.inputs[1].ty)?;

        // Expand ellipsis to concrete labels using input ranks.
        let equation =
            expand_ellipsis(&equation, &[lhs.rank, rhs.rank]).map_err(ProcessError::Custom)?;

        // Store the expanded equation back so extract_config sees it.
        node.attrs.insert(
            "equation".to_string(),
            crate::ir::AttributeValue::String(equation.clone()),
        );

        let parsed = ParsedEinsum::parse(&equation).map_err(ProcessError::Custom)?;

        if lhs.dtype != rhs.dtype {
            return Err(ProcessError::TypeMismatch {
                expected: format!("Both inputs to have dtype {:?}", lhs.dtype),
                actual: format!(
                    "Input A has dtype {:?}, Input B has dtype {:?}",
                    lhs.dtype, rhs.dtype
                ),
            });
        }

        if lhs.rank != parsed.lhs.len() {
            return Err(ProcessError::Custom(format!(
                "Einsum input 0 has rank {} but equation '{}' expects rank {}",
                lhs.rank,
                equation,
                parsed.lhs.len()
            )));
        }
        if rhs.rank != parsed.rhs.len() {
            return Err(ProcessError::Custom(format!(
                "Einsum input 1 has rank {} but equation '{}' expects rank {}",
                rhs.rank,
                equation,
                parsed.rhs.len()
            )));
        }

        let static_shape = infer_output_static_shape(&equation, &parsed, lhs, rhs)?;

        node.outputs[0].ty = if parsed.output.is_empty() {
            if node.inputs.iter().any(|input| input.ty.is_on_device()) {
                ArgType::ScalarTensor(lhs.dtype)
            } else {
                ArgType::ScalarNative(lhs.dtype)
            }
        } else {
            ArgType::Tensor(TensorType {
                dtype: lhs.dtype,
                rank: parsed.output.len(),
                static_shape,
            })
        };

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let equation = node
            .attrs
            .get("equation")
            .ok_or_else(|| ProcessError::MissingAttribute("equation".to_string()))?
            .clone()
            .into_string();

        ParsedEinsum::parse(&equation).map_err(ProcessError::Custom)?;

        Ok(EinsumConfig { equation })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Einsum(EinsumNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

/// Parsed representation of a supported 2-input explicit einsum equation.
#[derive(Debug, Clone)]
pub struct ParsedEinsum {
    /// Labels for the left input.
    pub lhs: Vec<char>,
    /// Labels for the right input.
    pub rhs: Vec<char>,
    /// Labels for the output.
    pub output: Vec<char>,
}

impl ParsedEinsum {
    /// Parse a 2-input einsum equation in explicit (`ij,jk->ik`) or implicit (`ij,jk`) form.
    ///
    /// In implicit form the output indices are the alphabetically sorted set of indices
    /// that appear exactly once across all input terms (per the ONNX spec).
    pub fn parse(equation: &str) -> Result<Self, String> {
        let equation: String = equation.chars().filter(|c| !c.is_whitespace()).collect();

        if equation.contains("...") {
            return Err(format!(
                "Einsum equation '{}' contains ellipsis which is not supported",
                equation
            ));
        }

        let (inputs_str, output) = if let Some((lhs_str, rhs_str)) = equation.split_once("->") {
            (lhs_str, rhs_str.chars().collect::<Vec<char>>())
        } else {
            // Implicit form: output is the alphabetically sorted set of indices
            // that appear exactly once across all input terms.
            let all_labels: Vec<char> = equation.replace(',', "").chars().collect();
            let mut counts = std::collections::BTreeMap::new();
            for &c in &all_labels {
                *counts.entry(c).or_insert(0usize) += 1;
            }
            let output: Vec<char> = counts
                .into_iter()
                .filter(|&(_, count)| count == 1)
                .map(|(c, _)| c)
                .collect();
            (equation.as_str(), output)
        };

        let input_parts: Vec<&str> = inputs_str.split(',').collect();
        if input_parts.len() != 2 {
            return Err(format!(
                "Einsum equation '{}' must have exactly 2 inputs, got {}",
                equation,
                input_parts.len()
            ));
        }

        let lhs: Vec<char> = input_parts[0].chars().collect();
        let rhs: Vec<char> = input_parts[1].chars().collect();

        for &c in lhs.iter().chain(rhs.iter()).chain(output.iter()) {
            if !c.is_ascii_lowercase() {
                return Err(format!(
                    "Einsum equation '{}' contains invalid character '{}'; only lowercase letters allowed",
                    equation, c
                ));
            }
        }

        if has_duplicates(&lhs) {
            return Err(format!(
                "Einsum equation '{}' has repeated indices in left input (trace not supported)",
                equation
            ));
        }
        if has_duplicates(&rhs) {
            return Err(format!(
                "Einsum equation '{}' has repeated indices in right input (trace not supported)",
                equation
            ));
        }
        if has_duplicates(&output) {
            return Err(format!(
                "Einsum equation '{}' has repeated indices in output",
                equation
            ));
        }

        for &c in &output {
            if !lhs.contains(&c) && !rhs.contains(&c) {
                return Err(format!(
                    "Einsum equation '{}': output index '{}' not found in any input",
                    equation, c
                ));
            }
        }

        Ok(ParsedEinsum { lhs, rhs, output })
    }

    /// Indices that appear in both inputs and in the output.
    pub fn batch_axes(&self) -> Vec<char> {
        self.lhs
            .iter()
            .filter(|c| self.rhs.contains(c) && self.output.contains(c))
            .copied()
            .collect()
    }

    /// Indices that appear in both inputs but not in the output.
    pub fn contraction_axes(&self) -> Vec<char> {
        self.lhs
            .iter()
            .filter(|c| self.rhs.contains(c) && !self.output.contains(c))
            .copied()
            .collect()
    }

    /// Indices that stay in the output from the left input only.
    pub fn free_lhs_axes(&self) -> Vec<char> {
        self.lhs
            .iter()
            .filter(|c| !self.rhs.contains(c) && self.output.contains(c))
            .copied()
            .collect()
    }

    /// Indices that stay in the output from the right input only.
    pub fn free_rhs_axes(&self) -> Vec<char> {
        self.rhs
            .iter()
            .filter(|c| !self.lhs.contains(c) && self.output.contains(c))
            .copied()
            .collect()
    }

    /// Indices that appear only in the left input and are absent from the output (summed out).
    pub fn reduced_lhs_axes(&self) -> Vec<char> {
        self.lhs
            .iter()
            .filter(|c| !self.rhs.contains(c) && !self.output.contains(c))
            .copied()
            .collect()
    }

    /// Indices that appear only in the right input and are absent from the output (summed out).
    pub fn reduced_rhs_axes(&self) -> Vec<char> {
        self.rhs
            .iter()
            .filter(|c| !self.lhs.contains(c) && !self.output.contains(c))
            .copied()
            .collect()
    }
}

/// Expand ellipsis (`...`) in an einsum equation to concrete lowercase labels.
///
/// The number of ellipsis dimensions is inferred from the tensor ranks: if a term has `...ij`
/// and the input has rank 5, the `...` represents 3 dimensions.
///
/// Returns the expanded equation with `...` replaced by unused lowercase labels.
fn expand_ellipsis(equation: &str, input_ranks: &[usize]) -> Result<String, String> {
    let equation: String = equation.chars().filter(|c| !c.is_whitespace()).collect();

    if !equation.contains("...") {
        return Ok(equation);
    }

    // Split into input and output parts (explicit or implicit form).
    let (inputs_str, output_str) = if let Some((lhs, rhs)) = equation.split_once("->") {
        (lhs.to_string(), Some(rhs.to_string()))
    } else {
        (equation.clone(), None)
    };

    let input_parts: Vec<&str> = inputs_str.split(',').collect();
    if input_parts.len() != input_ranks.len() {
        return Err(format!(
            "Einsum equation '{}' has {} input terms but {} input tensors",
            equation,
            input_parts.len(),
            input_ranks.len()
        ));
    }

    // Determine how many dims each ellipsis represents.
    let mut ellipsis_ndim: Option<usize> = None;
    for (part, &rank) in input_parts.iter().zip(input_ranks) {
        if part.contains("...") {
            let explicit_count = part.len() - 3; // subtract "..."
            if rank < explicit_count {
                return Err(format!(
                    "Einsum term '{}' has {} explicit labels but input has rank {}",
                    part, explicit_count, rank
                ));
            }
            let ndim = rank - explicit_count;
            if let Some(prev) = ellipsis_ndim
                && prev != ndim
            {
                return Err(format!(
                    "Einsum equation '{}' has inconsistent ellipsis dimensions: {} vs {}",
                    equation, prev, ndim
                ));
            }
            ellipsis_ndim = Some(ndim);
        }
    }

    let ndim = match ellipsis_ndim {
        Some(n) => n,
        None => return Ok(equation), // no ellipsis in input terms
    };

    // Collect all explicit labels used in the equation.
    let used_labels: std::collections::HashSet<char> = equation
        .chars()
        .filter(|c| c.is_ascii_lowercase())
        .collect();

    // Pick unused lowercase letters for the ellipsis dimensions.
    let available: Vec<char> = ('a'..='z').filter(|c| !used_labels.contains(c)).collect();
    if available.len() < ndim {
        return Err(format!(
            "Einsum equation '{}' uses too many labels; not enough free letters for {} ellipsis dims",
            equation, ndim
        ));
    }
    let ellipsis_labels: String = available[..ndim].iter().collect();

    // For implicit form + ellipsis, convert to explicit form per the ONNX spec:
    // "In implicit mode, the ellipsis dimensions are set to the beginning of the output."
    // After expanding `...` to concrete labels, those labels appear in both inputs
    // (counted twice) and would be excluded by the standard implicit-output rule.
    // We fix this by computing the output explicitly here.
    let output_str = match output_str {
        Some(out) => Some(out),
        None => {
            // Compute implicit output: ellipsis labels + alphabetically sorted
            // labels that appear exactly once across all expanded input terms.
            let expanded_inputs: String = input_parts
                .iter()
                .map(|p| p.replace("...", &ellipsis_labels))
                .collect::<Vec<_>>()
                .join(",");
            let all_labels: Vec<char> = expanded_inputs.replace(',', "").chars().collect();
            let mut counts = std::collections::BTreeMap::new();
            for &c in &all_labels {
                *counts.entry(c).or_insert(0usize) += 1;
            }
            let singletons: String = counts
                .into_iter()
                .filter(|&(_, count)| count == 1)
                .map(|(c, _)| c)
                .collect();
            Some(format!("{}{}", ellipsis_labels, singletons))
        }
    };

    // Replace each `...` with the concrete labels.
    let full_str = format!("{}->{}", inputs_str, output_str.as_deref().unwrap_or(""));

    let mut result = String::new();
    let mut chars = full_str.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '.' && chars.peek() == Some(&'.') {
            chars.next(); // consume second '.'
            if chars.peek() == Some(&'.') {
                chars.next(); // consume third '.'
                result.push_str(&ellipsis_labels);
            } else {
                result.push_str("..");
            }
        } else {
            result.push(c);
        }
    }

    Ok(result)
}

fn has_duplicates(chars: &[char]) -> bool {
    let mut seen = std::collections::HashSet::new();
    chars.iter().any(|c| !seen.insert(c))
}

fn infer_output_static_shape(
    equation: &str,
    parsed: &ParsedEinsum,
    lhs: EinsumOperand<'_>,
    rhs: EinsumOperand<'_>,
) -> Result<Option<Vec<Option<usize>>>, ProcessError> {
    validate_shared_static_dims(equation, parsed, lhs, rhs)?;

    if lhs.static_shape.is_none() && rhs.static_shape.is_none() {
        return Ok(None);
    }

    Ok(Some(
        parsed
            .output
            .iter()
            .map(|&label| {
                find_static_dim(&parsed.lhs, lhs.static_shape, label)
                    .flatten()
                    .or_else(|| find_static_dim(&parsed.rhs, rhs.static_shape, label).flatten())
            })
            .collect(),
    ))
}

fn validate_shared_static_dims(
    equation: &str,
    parsed: &ParsedEinsum,
    lhs: EinsumOperand<'_>,
    rhs: EinsumOperand<'_>,
) -> Result<(), ProcessError> {
    for &label in parsed.lhs.iter().filter(|label| parsed.rhs.contains(label)) {
        let lhs_dim = find_static_dim(&parsed.lhs, lhs.static_shape, label).flatten();
        let rhs_dim = find_static_dim(&parsed.rhs, rhs.static_shape, label).flatten();

        if let (Some(lhs_dim), Some(rhs_dim)) = (lhs_dim, rhs_dim)
            && lhs_dim != rhs_dim
        {
            return Err(ProcessError::Custom(format!(
                "Einsum equation '{}' has mismatched static dimensions for index '{}': lhs={}, rhs={}",
                equation, label, lhs_dim, rhs_dim
            )));
        }
    }

    Ok(())
}

fn find_static_dim(
    labels: &[char],
    static_shape: Option<&[Option<usize>]>,
    label: char,
) -> Option<Option<usize>> {
    let static_shape = static_shape?;
    let index = labels.iter().position(|&current| current == label)?;
    Some(static_shape[index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(equation: &str, lhs_rank: usize, rhs_rank: usize) -> RawNode {
        create_test_node_with_shapes(equation, lhs_rank, rhs_rank, None, None)
    }

    fn create_test_node_with_shapes(
        equation: &str,
        lhs_rank: usize,
        rhs_rank: usize,
        lhs_shape: Option<Vec<usize>>,
        rhs_shape: Option<Vec<usize>>,
    ) -> RawNode {
        TestNodeBuilder::new(NodeType::Einsum, "test_einsum")
            .input_tensor_f32("A", lhs_rank, lhs_shape)
            .input_tensor_f32("B", rhs_rank, rhs_shape)
            .output_tensor_f32("C", 0, None)
            .attr_string("equation", equation)
            .build()
    }

    fn create_test_node_with_partial_shapes(
        equation: &str,
        lhs_rank: usize,
        rhs_rank: usize,
        lhs_shape: Option<Vec<Option<usize>>>,
        rhs_shape: Option<Vec<Option<usize>>>,
    ) -> RawNode {
        let mut node = create_test_node(equation, lhs_rank, rhs_rank);
        node.inputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::F32,
            rank: lhs_rank,
            static_shape: lhs_shape,
        });
        node.inputs[1].ty = ArgType::Tensor(TensorType {
            dtype: DType::F32,
            rank: rhs_rank,
            static_shape: rhs_shape,
        });
        node
    }

    fn create_test_node_with_types(equation: &str, lhs_ty: ArgType, rhs_ty: ArgType) -> RawNode {
        TestNodeBuilder::new(NodeType::Einsum, "test_einsum")
            .add_input("A", lhs_ty)
            .add_input("B", rhs_ty)
            .output_tensor_f32("C", 0, None)
            .attr_string("equation", equation)
            .build()
    }

    #[test]
    fn test_parse_sam_pattern() {
        let parsed = ParsedEinsum::parse("bhwc,hkc->bhwk").unwrap();
        assert_eq!(parsed.lhs, vec!['b', 'h', 'w', 'c']);
        assert_eq!(parsed.rhs, vec!['h', 'k', 'c']);
        assert_eq!(parsed.output, vec!['b', 'h', 'w', 'k']);
        assert_eq!(parsed.batch_axes(), vec!['h']);
        assert_eq!(parsed.contraction_axes(), vec!['c']);
        assert_eq!(parsed.free_lhs_axes(), vec!['b', 'w']);
        assert_eq!(parsed.free_rhs_axes(), vec!['k']);
    }

    #[test]
    fn test_parse_batch_matmul() {
        let parsed = ParsedEinsum::parse("bij,bjk->bik").unwrap();
        assert_eq!(parsed.batch_axes(), vec!['b']);
        assert_eq!(parsed.contraction_axes(), vec!['j']);
        assert_eq!(parsed.free_lhs_axes(), vec!['i']);
        assert_eq!(parsed.free_rhs_axes(), vec!['k']);
    }

    #[test]
    fn test_parse_simple_matmul() {
        let parsed = ParsedEinsum::parse("ij,jk->ik").unwrap();
        assert_eq!(parsed.batch_axes(), Vec::<char>::new());
        assert_eq!(parsed.contraction_axes(), vec!['j']);
        assert_eq!(parsed.free_lhs_axes(), vec!['i']);
        assert_eq!(parsed.free_rhs_axes(), vec!['k']);
    }

    #[test]
    fn test_parse_outer_product() {
        let parsed = ParsedEinsum::parse("i,j->ij").unwrap();
        assert_eq!(parsed.batch_axes(), Vec::<char>::new());
        assert_eq!(parsed.contraction_axes(), Vec::<char>::new());
        assert_eq!(parsed.free_lhs_axes(), vec!['i']);
        assert_eq!(parsed.free_rhs_axes(), vec!['j']);
    }

    #[test]
    fn test_parse_implicit_form_matmul() {
        let parsed = ParsedEinsum::parse("ij,jk").unwrap();
        assert_eq!(parsed.lhs, vec!['i', 'j']);
        assert_eq!(parsed.rhs, vec!['j', 'k']);
        // j appears twice (summed out), i and k appear once -> output "ik"
        assert_eq!(parsed.output, vec!['i', 'k']);
        assert_eq!(parsed.contraction_axes(), vec!['j']);
    }

    #[test]
    fn test_parse_implicit_form_no_contraction() {
        let parsed = ParsedEinsum::parse("ij,kl").unwrap();
        // All indices appear once -> output "ijkl" (alphabetical)
        assert_eq!(parsed.output, vec!['i', 'j', 'k', 'l']);
    }

    #[test]
    fn test_parse_implicit_form_all_contracted() {
        let parsed = ParsedEinsum::parse("ij,ij").unwrap();
        // All indices appear twice -> scalar output
        assert_eq!(parsed.output, Vec::<char>::new());
    }

    #[test]
    fn test_parse_rejects_ellipsis() {
        assert!(ParsedEinsum::parse("...ij,...jk->...ik").is_err());
    }

    #[test]
    fn test_parse_rejects_three_inputs() {
        assert!(ParsedEinsum::parse("ij,jk,kl->il").is_err());
    }

    #[test]
    fn test_parse_rejects_trace() {
        assert!(ParsedEinsum::parse("ii,j->j").is_err());
    }

    #[test]
    fn test_parse_one_sided_reduction() {
        let parsed = ParsedEinsum::parse("ij,kl->il").unwrap();
        assert_eq!(parsed.reduced_lhs_axes(), vec!['j']);
        assert_eq!(parsed.reduced_rhs_axes(), vec!['k']);
        assert_eq!(parsed.batch_axes(), Vec::<char>::new());
        assert_eq!(parsed.contraction_axes(), Vec::<char>::new());
        assert_eq!(parsed.free_lhs_axes(), vec!['i']);
        assert_eq!(parsed.free_rhs_axes(), vec!['l']);
    }

    #[test]
    fn test_parse_one_sided_reduction_lhs_only() {
        let parsed = ParsedEinsum::parse("ijk,l->il").unwrap();
        assert_eq!(parsed.reduced_lhs_axes(), vec!['j', 'k']);
        assert_eq!(parsed.reduced_rhs_axes(), Vec::<char>::new());
    }

    #[test]
    fn test_parse_rejects_invalid_chars() {
        assert!(ParsedEinsum::parse("iJ,Jk->ik").is_err());
    }

    #[test]
    fn test_parse_whitespace_tolerance() {
        let parsed = ParsedEinsum::parse("ij, jk -> ik").unwrap();
        assert_eq!(parsed.lhs, vec!['i', 'j']);
        assert_eq!(parsed.rhs, vec!['j', 'k']);
        assert_eq!(parsed.output, vec!['i', 'k']);
    }

    #[test]
    fn test_infer_types_sam_pattern() {
        let mut node = create_test_node_with_partial_shapes(
            "bhwc,hkc->bhwk",
            4,
            3,
            Some(vec![None, Some(2), None, None]),
            Some(vec![Some(2), None, None]),
        );
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_infer_types_simple_matmul() {
        let mut node = create_test_node("ij,jk->ik", 2, 2);
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_infer_types_rank_mismatch() {
        let mut node = create_test_node("bhwc,hkc->bhwk", 3, 3);
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_types_dtype_mismatch() {
        let mut node = create_test_node("ij,jk->ik", 2, 2);
        node.inputs[1].ty = ArgType::Tensor(TensorType {
            dtype: DType::F64,
            rank: 2,
            static_shape: None,
        });
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_infer_types_rejects_mismatched_static_dimensions() {
        let mut node =
            create_test_node_with_shapes("ij,jk->ik", 2, 2, Some(vec![2, 3]), Some(vec![4, 5]));
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();

        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_infer_types_accepts_dynamic_multi_contraction_axes() {
        let mut node = create_test_node("cd,dc->", 2, 2);
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(
            node.outputs[0].ty,
            ArgType::ScalarTensor(DType::F32)
        ));
    }

    #[test]
    fn test_infer_types_accepts_dynamic_multi_batch_axes() {
        let mut node = create_test_node("abij,abjk->abik", 4, 4);
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_infer_types_accepts_dynamic_mixed_batch_and_contraction_axes() {
        let mut node = create_test_node("bhwc,hkc->bhwk", 4, 3);
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_infer_types_allows_one_dynamic_shared_axis_when_others_are_static() {
        let mut node = create_test_node_with_partial_shapes(
            "abij,abjk->abik",
            4,
            4,
            Some(vec![Some(2), None, Some(4), Some(5)]),
            Some(vec![Some(2), None, Some(5), Some(7)]),
        );
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(
                    tensor.static_shape,
                    Some(vec![Some(2), None, Some(4), Some(7)])
                );
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_infer_types_allows_static_multi_batch_axes() {
        let mut node = create_test_node_with_shapes(
            "abij,abjk->abik",
            4,
            4,
            Some(vec![2, 3, 4, 5]),
            Some(vec![2, 3, 5, 7]),
        );
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(
                    tensor.static_shape,
                    Some(vec![Some(2), Some(3), Some(4), Some(7)])
                );
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_infer_types_uses_rhs_known_dim_for_shared_output_axis() {
        let mut node = create_test_node_with_partial_shapes(
            "bij,bjk->bik",
            3,
            3,
            Some(vec![None, Some(3), Some(4)]),
            Some(vec![Some(2), Some(4), Some(5)]),
        );
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, Some(vec![Some(2), Some(3), Some(5)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_infer_types_propagates_static_shape() {
        let mut node = create_test_node_with_shapes(
            "bhwc,hkc->bhwk",
            4,
            3,
            Some(vec![1, 2, 3, 4]),
            Some(vec![2, 5, 4]),
        );
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(
                    tensor.static_shape,
                    Some(vec![Some(1), Some(2), Some(3), Some(5)])
                );
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_infer_types_accepts_scalar_native_lhs_empty_term() {
        let mut node = create_test_node_with_types(
            ",ij->ij",
            ArgType::ScalarNative(DType::F32),
            ArgType::Tensor(TensorType::new_known(DType::F32, vec![3, 4])),
        );
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![Some(3), Some(4)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_infer_types_accepts_scalar_tensor_rhs_empty_term() {
        let mut node = create_test_node_with_types(
            "ij,->ij",
            ArgType::Tensor(TensorType::new_known(DType::F32, vec![3, 4])),
            ArgType::ScalarTensor(DType::F32),
        );
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![Some(3), Some(4)]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_infer_types_scalar_scalar_output_is_scalar_native() {
        let mut node = create_test_node_with_types(
            ",->",
            ArgType::ScalarNative(DType::F32),
            ArgType::ScalarNative(DType::F32),
        );
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(
            node.outputs[0].ty,
            ArgType::ScalarNative(DType::F32)
        ));
    }

    #[test]
    fn test_infer_types_scalar_tensor_output_stays_on_device() {
        let mut node = create_test_node_with_types(
            ",->",
            ArgType::ScalarNative(DType::F32),
            ArgType::ScalarTensor(DType::F32),
        );
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(
            node.outputs[0].ty,
            ArgType::ScalarTensor(DType::F32)
        ));
    }

    #[test]
    fn test_extract_config() {
        let node = create_test_node("bhwc,hkc->bhwk", 4, 3);
        let processor = EinsumProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.equation, "bhwc,hkc->bhwk");
    }

    #[test]
    fn test_output_index_not_in_inputs() {
        assert!(ParsedEinsum::parse("ij,jk->iz").is_err());
    }

    #[test]
    fn test_expand_ellipsis_batch_matmul() {
        let result = expand_ellipsis("...ij,...jk->...ik", &[4, 4]).unwrap();
        // 4 - 2 explicit = 2 ellipsis dims, uses first 2 unused letters
        // used letters: i, j, k; unused starting from 'a': a, b
        assert_eq!(result, "abij,abjk->abik");
    }

    #[test]
    fn test_expand_ellipsis_single_batch() {
        let result = expand_ellipsis("...ij,...jk->...ik", &[3, 3]).unwrap();
        // 3 - 2 = 1 ellipsis dim
        assert_eq!(result, "aij,ajk->aik");
    }

    #[test]
    fn test_expand_ellipsis_zero_dims() {
        let result = expand_ellipsis("...ij,...jk->...ik", &[2, 2]).unwrap();
        // 2 - 2 = 0 ellipsis dims
        assert_eq!(result, "ij,jk->ik");
    }

    #[test]
    fn test_expand_ellipsis_no_ellipsis_passthrough() {
        let result = expand_ellipsis("ij,jk->ik", &[2, 2]).unwrap();
        assert_eq!(result, "ij,jk->ik");
    }

    #[test]
    fn test_expand_ellipsis_inconsistent_dims() {
        let result = expand_ellipsis("...ij,...jk->...ik", &[4, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_expand_ellipsis_implicit_form() {
        // Implicit + ellipsis: per ONNX spec, ellipsis dims go at the beginning
        // of the output, followed by alphabetically sorted singleton labels.
        let result = expand_ellipsis("...ij,...jk", &[4, 4]).unwrap();
        // j appears in both inputs (contracted), i and k are singletons
        // Output = ellipsis_labels + singletons = "ab" + "ik" = "abik"
        assert_eq!(result, "abij,abjk->abik");
    }

    #[test]
    fn test_expand_ellipsis_avoids_used_labels() {
        // a, b, c already used; ellipsis needs 2 dims -> picks d, e
        let result = expand_ellipsis("...ab,...bc->...ac", &[4, 4]).unwrap();
        assert_eq!(result, "deab,debc->deac");
    }

    #[test]
    fn test_infer_types_ellipsis_batch_matmul() {
        let mut node = create_test_node_with_shapes(
            "...ij,...jk->...ik",
            4,
            4,
            Some(vec![2, 3, 4, 5]),
            Some(vec![2, 3, 5, 7]),
        );
        let processor = EinsumProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 4);
                assert_eq!(
                    tensor.static_shape,
                    Some(vec![Some(2), Some(3), Some(4), Some(7)])
                );
            }
            _ => panic!("Expected tensor output"),
        }

        // Verify the equation was expanded in the config
        let config = processor.extract_config(&node, 16).unwrap();
        assert_eq!(config.equation, "abij,abjk->abik");
    }
}
