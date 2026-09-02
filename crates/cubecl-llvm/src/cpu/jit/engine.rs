use cubecl_runtime::kernel::BufferIOAttr;
use std::ffi::c_void;
use std::fmt::Display;
use std::sync::{Arc, Once};

use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron_llvm::llvm_sys::core::{LLVMContext, LLVMMemoryBuffer, LLVMModule};
use pliron_llvm::llvm_sys::lljit::LLVMLLJIT;
use pliron_llvm::llvm_sys::target::initialize_native;
use pliron_llvm::to_llvm_ir;

use super::data::PlironData;
use crate::cpu::shared_memory::SharedMemories;

/// Host ABI of a JIT'd kernel: `(buffer_ptrs, cube_count_x/y/z, unit_pos_x/y/z, sync_cube_state,
/// metadata)`. The variable-count pointers — the buffers and then the shared memories — are
/// hidden behind `buffer_ptrs`, while the builtins and both other pointers are passed directly.
type KernelFn = extern "C" fn(*mut *mut c_void, u32, u32, u32, u32, u32, u32, *mut u32, *mut u64);

/// What the host has to provide to launch a kernel, beyond its arguments.
#[derive(Clone, Debug, Default)]
pub struct KernelRequirements {
    /// Whether the kernel synchronizes its cube, in which case each of its units needs a thread
    /// of its own to run on.
    pub needs_parallelism: bool,
    /// The shared memory to reserve for a launch, and where its pointers go.
    pub shared_memories: SharedMemories,
}

/// A JIT-compiled kernel.
///
/// The `LLVMContext` the module was built in is not held here: `LLVMLLJIT::add_module`
/// takes it by value and transfers it to the JIT's thread-safe context, so the JIT is
/// what keeps it alive for as long as the compiled code exists.
#[repr(C)]
struct JitKernel {
    func: KernelFn,
    requirements: KernelRequirements,
    /// What the kernel does with each buffer binding, by buffer position --
    /// read off the IR before the entry ABI lowering erased the arguments,
    /// for the launch path's taint bookkeeping.
    io: Vec<BufferIOAttr>,
    _lljit: LLVMLLJIT,
}

/// Safety: The kernel is immutable machine code plus the JIT/context that keep it alive.
unsafe impl Send for JitKernel {}
unsafe impl Sync for JitKernel {}

/// A compiled kernel, cloneable across worker threads.
#[derive(Clone)]
pub struct PlironEngine(Arc<JitKernel>);

static INIT_NATIVE: Once = Once::new();

impl PlironEngine {
    /// Lower the LLVM-dialect module to LLVM IR and JIT-compile it with ORC/LLJIT.
    pub fn compile(
        ctx: &Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
    ) -> pliron::result::Result<Self> {
        INIT_NATIVE.call_once(|| {
            initialize_native().expect("failed to initialize native target");
        });

        let llvm_ctx = LLVMContext::default();
        let llvm_module = to_llvm_ir::convert_module(ctx, &llvm_ctx, module)?;
        #[cfg(feature = "pliron-dump")]
        if let Some(dir) = ir_dump_path(kernel_name) {
            let _ = std::fs::write(dir.join("llvm.ll"), llvm_module.to_string());
        }

        let llvm_module = optimize(llvm_module, &llvm_ctx, kernel_name)
            .unwrap_or_else(|err| panic!("LLVM optimization failed for '{kernel_name}': {err}"));
        #[cfg(feature = "pliron-dump")]
        if let Some(dir) = ir_dump_path(kernel_name) {
            let _ = std::fs::write(dir.join("llvm.opt.ll"), llvm_module.to_string());
        }

        let lljit = LLVMLLJIT::new_with_default_builder().expect("failed to create LLJIT");
        // Consumes the context: `optimize` re-parsed the module into `llvm_ctx`, which is
        // what `add_module` asserts, and ownership passes to the JIT from here.
        lljit
            .add_module(llvm_ctx, llvm_module)
            .expect("failed to add module to JIT");
        let addr = lljit
            .lookup_symbol(kernel_name)
            .unwrap_or_else(|err| panic!("kernel symbol '{kernel_name}' not found: {err}"));
        // Safety: the generated function is always of this form
        let func: KernelFn = unsafe { std::mem::transmute::<u64, KernelFn>(addr) };

        Ok(PlironEngine(Arc::new(JitKernel {
            func,
            requirements,
            io,
            _lljit: lljit,
        })))
    }

    /// What the host has to provide to launch this kernel, see [`KernelRequirements`].
    pub fn requirements(&self) -> &KernelRequirements {
        &self.0.requirements
    }

    /// What the kernel does with each buffer binding, by buffer position.
    pub fn buffer_io(&self) -> &[BufferIOAttr] {
        &self.0.io
    }

    pub fn run_kernel(&self, data: &mut PlironData) {
        let b = data.builtins;
        let buffer_ptrs = data.shared.buffer_ptrs.as_ptr() as *mut *mut c_void;
        let metadata = data.shared.metadata.as_ptr() as *mut u64;
        let sync_cube_state = data.shared.sync_cube_state.as_ptr() as *mut u32;
        (self.0.func)(
            buffer_ptrs,
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            b[5],
            sync_cube_state,
            metadata,
        );
    }
}

impl Display for PlironEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pliron JIT engine")
    }
}

#[cfg(feature = "pliron-dump")]
/// The kernel's dump directory when `CUBECL_DEBUG_PLIRON` is set: the LLVM IR
/// stages land beside the pliron pass dumps.
pub(crate) fn ir_dump_path(kernel_name: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::var("CUBECL_DEBUG_PLIRON").ok()?;
    let path = std::path::Path::new(&dir).join(kernel_name);
    std::fs::create_dir_all(&path).ok()?;
    Some(path)
}

/// The pipeline run before the JIT: LLJIT runs no IR-level passes of its own,
/// and the dialect lowering emits O0-shaped IR that runs ~50× under the
/// machine's streaming rate. Paid once per kernel and cached.
const PASS_PIPELINE: &str = "default<O3>";

/// Optimizes a module by round-tripping it through textual IR: [`LLVMModule`]
/// seals its `LLVMModuleRef`, so the IR is re-parsed into a context this
/// function owns, optimized there, and parsed back. Fold into a direct pass
/// run when pliron-llvm exposes one.
fn optimize(
    module: LLVMModule,
    llvm_ctx: &LLVMContext,
    kernel_name: &str,
) -> Result<LLVMModule, String> {
    let optimized = run_pipeline(&module.to_string())?;
    drop(module);
    LLVMModule::from_ir_in_memory_buffer(
        llvm_ctx,
        LLVMMemoryBuffer::from_str(&optimized, kernel_name),
    )
}

/// Parses `ir` into a private LLVM context, runs [`PASS_PIPELINE`] over it,
/// and prints the optimized module back out.
fn run_pipeline(ir: &str) -> Result<String, String> {
    use llvm_sys::core::{
        LLVMContextCreate, LLVMContextDispose, LLVMCreateMemoryBufferWithMemoryRangeCopy,
        LLVMDisposeMessage, LLVMDisposeModule, LLVMPrintModuleToString,
    };
    use llvm_sys::error::{LLVMDisposeErrorMessage, LLVMGetErrorMessage};
    use llvm_sys::ir_reader::LLVMParseIRInContext2;
    use llvm_sys::transforms::pass_builder::{
        LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions, LLVMRunPasses,
    };

    unsafe {
        let ctx = LLVMContextCreate();
        let buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
            ir.as_ptr() as *const _,
            ir.len(),
            c"kernel".as_ptr(),
        );
        let mut module = std::ptr::null_mut();
        let mut parse_err = std::ptr::null_mut();
        // `LLVMParseIRInContext2` consumes the buffer, on failure included.
        if LLVMParseIRInContext2(ctx, buffer, &mut module, &mut parse_err) != 0 {
            let msg = std::ffi::CStr::from_ptr(parse_err)
                .to_string_lossy()
                .into_owned();
            LLVMDisposeMessage(parse_err);
            LLVMContextDispose(ctx);
            return Err(msg);
        }

        let passes = std::ffi::CString::new(PASS_PIPELINE).expect("static pass string");
        let options = LLVMCreatePassBuilderOptions();
        // A null target machine makes `default<O3>` target-independent: LoopVectorize /
        // SLP never see host AVX512/VNNI, so integer GEMM stays a scalar i32 FMA.
        // Build a host TM (same CPU/features as this process) and fall back to null
        // if creation fails so non-native hosts still compile.
        let tm = host_target_machine();
        if !tm.is_null() {
            use llvm_sys::target::{LLVMDisposeTargetData, LLVMSetModuleDataLayout};
            use llvm_sys::target_machine::LLVMCreateTargetDataLayout;
            let layout = LLVMCreateTargetDataLayout(tm);
            LLVMSetModuleDataLayout(module, layout);
            LLVMDisposeTargetData(layout);
        }
        let err = LLVMRunPasses(module, passes.as_ptr(), tm, options);
        LLVMDisposePassBuilderOptions(options);
        if !tm.is_null() {
            use llvm_sys::target_machine::LLVMDisposeTargetMachine;
            LLVMDisposeTargetMachine(tm);
        }

        let result = if err.is_null() {
            let c_ir = LLVMPrintModuleToString(module);
            let optimized = std::ffi::CStr::from_ptr(c_ir)
                .to_string_lossy()
                .into_owned();
            LLVMDisposeMessage(c_ir);
            Ok(optimized)
        } else {
            let c_msg = LLVMGetErrorMessage(err);
            let msg = std::ffi::CStr::from_ptr(c_msg)
                .to_string_lossy()
                .into_owned();
            LLVMDisposeErrorMessage(c_msg);
            Err(msg)
        };
        LLVMDisposeModule(module);
        LLVMContextDispose(ctx);
        result
    }
}

/// Host `TargetMachine` for the process CPU, or null if LLVM cannot describe it.
///
/// `LLVMRunPasses(..., tm=null)` is target-independent: it will not emit AVX512 /
/// VNNI even when the machine has them. The caller must dispose a non-null result.
fn host_target_machine() -> llvm_sys::target_machine::LLVMTargetMachineRef {
    use llvm_sys::core::LLVMDisposeMessage;
    use llvm_sys::target_machine::{
        LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetMachine, LLVMGetHostCPUFeatures,
        LLVMGetHostCPUName, LLVMGetTargetFromTriple, LLVMRelocMode,
    };

    unsafe {
        let triple = llvm_sys::target_machine::LLVMGetDefaultTargetTriple();
        if triple.is_null() {
            return std::ptr::null_mut();
        }
        let cpu = LLVMGetHostCPUName();
        let features = LLVMGetHostCPUFeatures();
        let mut target = std::ptr::null_mut();
        let mut error = std::ptr::null_mut();
        let rc = LLVMGetTargetFromTriple(triple, &mut target, &mut error);
        if rc != 0 || target.is_null() {
            if !error.is_null() {
                LLVMDisposeMessage(error);
            }
            LLVMDisposeMessage(triple);
            if !cpu.is_null() {
                LLVMDisposeMessage(cpu);
            }
            if !features.is_null() {
                LLVMDisposeMessage(features);
            }
            return std::ptr::null_mut();
        }

        let empty = c"";
        let cpu_ptr = if cpu.is_null() { empty.as_ptr() } else { cpu };
        let feat_ptr = if features.is_null() {
            empty.as_ptr()
        } else {
            features
        };

        let tm = LLVMCreateTargetMachine(
            target,
            triple,
            cpu_ptr,
            feat_ptr,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
            LLVMRelocMode::LLVMRelocDefault,
            LLVMCodeModel::LLVMCodeModelJITDefault,
        );

        LLVMDisposeMessage(triple);
        if !cpu.is_null() {
            LLVMDisposeMessage(cpu);
        }
        if !features.is_null() {
            LLVMDisposeMessage(features);
        }
        tm
    }
}
