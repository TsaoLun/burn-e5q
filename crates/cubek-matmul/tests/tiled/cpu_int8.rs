//! Integer CpuGemm: `u8/i8` inputs, `i32` accumulate.
//!
//! The shared harness generates `uniform(-1, 1)` which truncates to 0 for `u8`,
//! so these cases feed explicit integer values through [`launch_ref`].

use cubecl::{Runtime, ir::ElemType, prelude::*};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems},
    routine::BlueprintStrategy,
    tiled::cpu_gemm::{
        CpuGemmBlueprint, CpuGemmStrategy, InstructionShape, PlaneGrid, WithLayout, launch_ref,
    },
};
use cubek_std::InputBinding;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, TestInput, skip_unless_cpu};

type TestRuntime = cubecl::TestRuntime;

fn as_i32(x: f32, ty: ElemType) -> i32 {
    match ty {
        ElemType::UInt(cubecl::ir::UIntKind::U8) => x as u8 as i32,
        ElemType::Int(cubecl::ir::IntKind::I8) => x as i8 as i32,
        ElemType::Int(cubecl::ir::IntKind::I32) => x as i32,
        other => panic!("unsupported test dtype {other:?}"),
    }
}

fn naive_i32(
    m: usize,
    n: usize,
    k: usize,
    lhs: &[f32],
    rhs: &[f32],
    lt: ElemType,
    rt: ElemType,
) -> Vec<i32> {
    let mut out = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for kk in 0..k {
                let a = as_i32(lhs[i * k + kk], lt);
                let b = as_i32(rhs[kk * n + j], rt);
                sum = sum.wrapping_add(a.wrapping_mul(b));
            }
            out[i * n + j] = sum;
        }
    }
    out
}

fn run_int8(
    m: usize,
    n: usize,
    k: usize,
    lhs_ty: ElemType,
    rhs_ty: ElemType,
    lhs: Vec<f32>,
    rhs: Vec<f32>,
    strategy: BlueprintStrategy<(), cubek_matmul::tiled::cpu_gemm::CpuGemmRoutine>,
) {
    let client = TestRuntime::client(&Default::default());
    if skip_unless_cpu(&client) {
        return;
    }
    assert_eq!(lhs.len(), m * k);
    assert_eq!(rhs.len(), n * k);

    let expected = naive_i32(m, n, k, &lhs, &rhs, lhs_ty, rhs_ty);
    let out_ty = i32::elem_type_native();

    let lhs_t = TestInput::builder(client.clone(), vec![m, k])
        .dtype(lhs_ty)
        .custom(lhs)
        .generate_without_host_data();
    let rhs_t = TestInput::builder(client.clone(), vec![k, n])
        .dtype(rhs_ty)
        .custom(rhs)
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), vec![m, n])
        .dtype(out_ty)
        .custom(vec![0.0; m * n])
        .generate_without_host_data();

    launch_ref::<TestRuntime>(
        &client,
        WithLayout::strided_input(InputBinding::Normal(lhs_t.binding(), lhs_ty)).unwrap(),
        WithLayout::strided_input(InputBinding::Normal(rhs_t.binding(), rhs_ty)).unwrap(),
        WithLayout::strided_output(out.clone().binding()).unwrap(),
        &strategy,
        &MatmulElems::from_globals(&MatmulGlobalElems {
            lhs: lhs_ty,
            rhs: rhs_ty,
            out: out_ty,
        }),
    )
    .unwrap();

    let actual = match HostData::from_tensor_handle(&client, out, HostDataType::I32).data {
        HostDataVec::I32(v) => v,
        other => panic!("expected i32 host data, got {other:?}"),
    };
    assert_eq!(actual, expected);
}

fn forced(tile: usize) -> BlueprintStrategy<(), cubek_matmul::tiled::cpu_gemm::CpuGemmRoutine> {
    BlueprintStrategy::Forced(CpuGemmBlueprint {
        instruction: InstructionShape {
            m: tile,
            n: tile,
            k: tile,
        },
        planes: PlaneGrid { m: 1, n: 1 },
    })
}

#[test]
fn u8_times_u8_to_i32() {
    let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    run_int8(
        2,
        2,
        3,
        u8::elem_type_native(),
        u8::elem_type_native(),
        lhs,
        rhs,
        forced(2),
    );
}

#[test]
fn i8_times_i8_to_i32() {
    let lhs = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let rhs = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    run_int8(
        2,
        2,
        3,
        i8::elem_type_native(),
        i8::elem_type_native(),
        lhs,
        rhs,
        forced(2),
    );
}

#[test]
fn u8_times_i8_to_i32() {
    // e5 MatMulInteger: DQL activations are u8, weights are typically i8.
    let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
    run_int8(
        2,
        2,
        3,
        u8::elem_type_native(),
        i8::elem_type_native(),
        lhs,
        rhs,
        forced(2),
    );
}

#[test]
fn u8_i8_ragged_k_not_multiple_of_4() {
    let (m, n, k) = (3, 5, 7);
    let lhs: Vec<f32> = (0..m * k).map(|i| ((i * 3) % 17) as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| ((i as i32 % 11) - 5) as f32).collect();
    run_int8(
        m,
        n,
        k,
        u8::elem_type_native(),
        i8::elem_type_native(),
        lhs,
        rhs,
        forced(4),
    );
}

#[test]
fn u8_i8_e5_like_ffn_panel() {
    // FFN-ish: M is a short packed seq, K=384 (e5 hidden), N=32 (a slice of 1536).
    let (m, n, k) = (8, 32, 64);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i % 251) as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| ((i as i32 % 13) - 6) as f32).collect();
    run_int8(
        m,
        n,
        k,
        u8::elem_type_native(),
        i8::elem_type_native(),
        lhs,
        rhs,
        forced(8),
    );
}

#[test]
fn u8_i8_inferred_heuristic() {
    let (m, n, k) = (16, 32, 48);
    let lhs: Vec<f32> = (0..m * k).map(|i| ((i * 5) % 200) as f32).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| ((i as i32 % 9) - 4) as f32).collect();
    run_int8(
        m,
        n,
        k,
        u8::elem_type_native(),
        i8::elem_type_native(),
        lhs,
        rhs,
        BlueprintStrategy::Inferred(CpuGemmStrategy::default()),
    );
}
