#!/usr/bin/env python3
"""Generate a minimal ONNX model with DynamicQuantizeLinear + MatMulInteger.

Graph:
  x(f32[2,2]) -> DynamicQuantizeLinear -> (y u8, scale f32, zp u8)
  y + zp -> MatMulInteger(const i8[2,2]) -> i32[2,2]
  i32 -> DequantizeLinear -> f32[2,2]

No `onnx` package required: we hand-craft the protobuf.
"""
import struct
import sys

def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)

def key(field, wire):
    return varint((field << 3) | wire)

def len_delim(field, payload):
    return key(field, 2) + varint(len(payload)) + payload

def make_node(op_type, name, inputs, outputs, domain="", attrs=None):
    # NodeProto: input=1, output=2, name=3, op_type=4, domain=7, attribute=5
    buf = bytearray()
    for s in inputs:
        buf += len_delim(1, s.encode())
    for s in outputs:
        buf += len_delim(2, s.encode())
    buf += len_delim(3, name.encode())
    buf += len_delim(4, op_type.encode())
    if attrs:
        for a in attrs:
            buf += len_delim(5, a)
    if domain:
        buf += len_delim(7, domain.encode())
    return bytes(buf)

def make_tensor(name, dims, data_type, raw_data):
    # TensorProto: dims=1, data_type=2, name=8, raw_data=9
    buf = bytearray()
    for d in dims:
        buf += key(1, 0) + varint(d)
    buf += key(2, 0) + varint(data_type)
    buf += len_delim(8, name.encode())
    buf += len_delim(9, raw_data)
    return bytes(buf)

def make_value_info(name, elem_type, dims):
    # ValueInfoProto: name=1, type=2
    # TypeProto: tensor_type=1
    # Tensor: elem_type=1, shape=2
    shape = bytearray()
    for d in dims:
        # TensorShapeProto.Dimension: dim_value=1
        dim = key(1, 0) + varint(d)
        shape += len_delim(1, dim)
    tensor_type = key(1, 0) + varint(elem_type) + len_delim(2, bytes(shape))
    type_proto = len_delim(1, tensor_type)
    return len_delim(1, name.encode()) + len_delim(2, type_proto)

def make_graph(nodes, name, inputs, outputs, initializers=None):
    # GraphProto: node=1, name=2, initializer=5, input=11, output=12
    buf = bytearray()
    for n in nodes:
        buf += len_delim(1, n)
    buf += len_delim(2, name.encode())
    if initializers:
        for t in initializers:
            buf += len_delim(5, t)
    for i in inputs:
        buf += len_delim(11, i)
    for o in outputs:
        buf += len_delim(12, o)
    return bytes(buf)

def make_model(graph, opset=13):
    # ModelProto: ir_version=1, opset_import=8, graph=2
    buf = bytearray()
    buf += key(1, 0) + varint(8)
    buf += len_delim(2, graph)
    # OperatorSetIdProto: domain=1, version=2
    opset_id = len_delim(2, "".encode()) + key(2, 0) + varint(opset)
    buf += len_delim(8, opset_id)
    return bytes(buf)

# ONNX TensorProto.DataType
FLOAT = 1
UINT8 = 2
INT8 = 3
INT32 = 6

# Nodes
dql = make_node(
    "DynamicQuantizeLinear",
    "dql1",
    ["x"],
    ["y_quant", "y_scale", "y_zero_point"],
)

# Constant i8 weight [2,2]: [[1,0],[0,1]]
w_raw = struct.pack("<4b", 1, 0, 0, 1)
w_init = make_tensor("w", [2, 2], INT8, w_raw)

mm = make_node(
    "MatMulInteger",
    "mm1",
    ["y_quant", "w", "y_zero_point", ""],
    ["mm_out"],
)

# DequantizeLinear: scale is scalar 1.0, zp is scalar 0
scale_raw = struct.pack("<f", 1.0)
scale_init = make_tensor("dq_scale", [], FLOAT, scale_raw)
zp_raw = struct.pack("<i", 0)
zp_init = make_tensor("dq_zp", [], INT32, zp_raw)

deq = make_node(
    "DequantizeLinear",
    "deq1",
    ["mm_out", "dq_scale", "dq_zp"],
    ["y"],
)

graph = make_graph(
    [dql, mm, deq],
    "dql_matmul",
    inputs=[make_value_info("x", FLOAT, [2, 2])],
    outputs=[make_value_info("y", FLOAT, [2, 2])],
    initializers=[w_init, scale_init, zp_init],
)

model = make_model(graph, opset=13)

out = sys.argv[1] if len(sys.argv) > 1 else "dql_matmul.onnx"
with open(out, "wb") as f:
    f.write(model)
print(f"wrote {out} ({len(model)} bytes)")
