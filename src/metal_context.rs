use std::ptr::NonNull;
use std::ffi::c_void;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLSize};
use objc2_foundation::ns_string;
use objc2::{rc::Retained, runtime::ProtocolObject};
use super::matrix::Matrix;

pub struct GeMMMetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    _queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    compute_encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,

    a_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    b_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    c_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl GeMMMetalContext {
    pub fn gemm_init() -> Self {
        let device = MTLCreateSystemDefaultDevice().unwrap();
        let lib = device.newLibraryWithSource_options_error(
            ns_string!(include_str!("../shaders/gemm.metal")), None).unwrap();
        let func = lib.newFunctionWithName(ns_string!("gemm_tensor_kernel")).unwrap();
        let state = device.newComputePipelineStateWithFunction_error(&func).unwrap();

        let queue = device.newCommandQueue().unwrap();
        let command_buffer = queue.commandBuffer().unwrap();
        let compute_encoder = command_buffer.computeCommandEncoder().unwrap();
        
        compute_encoder.setComputePipelineState(&state);
        Self {
            device,
            _queue: queue,
            compute_encoder,
            command_buffer,

            a_buffer: None,
            b_buffer: None,
            c_buffer: None,
        }
    }

    pub fn bind_data(&mut self, a: &mut Matrix<f32>, b: &mut Matrix<f32>) {
        assert_eq!(a.size.1, b.size.0);
        let m = a.size.0 as u32;
        let k = a.size.1 as u32;
        let n = b.size.1 as u32;

        self.a_buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(a.data.as_mut_ptr() as *mut c_void).unwrap(), 
                a.data.len() * std::mem::size_of::<f32>(), 
                objc2_metal::MTLResourceOptions::StorageModeShared)
            };

        self.b_buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(b.data.as_mut_ptr() as *mut c_void).unwrap(), 
                b.data.len() * std::mem::size_of::<f32>(), 
                objc2_metal::MTLResourceOptions::StorageModeShared)
            };
        
        self.c_buffer = {
            self.device.newBufferWithLength_options(
                (m as usize *n as usize) as usize * std::mem::size_of::<f32>(),
                objc2_metal::MTLResourceOptions::StorageModeShared)
        };

        #[repr(C)]
        struct MatrixDims {
            m: u32,
            k: u32,
            n: u32,
        }

        let sizes = MatrixDims { m,n,k };

        let sizes_uniform_buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(&sizes as *const MatrixDims as *mut c_void).unwrap(), 
                std::mem::size_of::<MatrixDims>(), 
                objc2_metal::MTLResourceOptions::StorageModeShared)
            };
        
        
        let thread_group_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let thread_group_nb = MTLSize {
            width: n as usize,
            height: m as usize,
            depth: 1,
        };

        

        unsafe {
            self.compute_encoder.setBuffer_offset_atIndex(self.a_buffer.as_deref(), 0, 0);
            self.compute_encoder.setBuffer_offset_atIndex(self.b_buffer.as_deref(), 0, 1);
            self.compute_encoder.setBuffer_offset_atIndex(self.c_buffer.as_deref(), 0, 2);
            self.compute_encoder.setBuffer_offset_atIndex(sizes_uniform_buffer.as_deref(), 0, 3);
            self.compute_encoder.dispatchThreadgroups_threadsPerThreadgroup(thread_group_nb, thread_group_size);
        }
    }

    pub unsafe fn compute(&mut self) -> &[f32]{
        unsafe {
            self.compute_encoder.endEncoding();
            self.command_buffer.commit();
            self.command_buffer.waitUntilCompleted();
        }
        
        let c_buffer= self.c_buffer.as_mut().unwrap();
        let result = c_buffer.contents().cast::<f32>().as_ptr();
        unsafe { std::slice::from_raw_parts(result, c_buffer.length() / std::mem::size_of::<f32>()) }
    }
}