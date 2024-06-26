; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

; Range checks: for all the instruction tested in this file, the
; immediate must be within the range [-8, 7] (4-bit immediate). Out of
; range values are tested only in one case (following). Valid values
; are tested all through the rest of the file.

define void @imm_out_of_range(ptr %base, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: imm_out_of_range:
; CHECK:       // %bb.0:
; CHECK-NEXT:    rdvl x8, #8
; CHECK-NEXT:    add x8, x0, x8
; CHECK-NEXT:    ldnt1d { z0.d }, p0/z, [x8]
; CHECK-NEXT:    rdvl x8, #-9
; CHECK-NEXT:    add x8, x0, x8
; CHECK-NEXT:    stnt1d { z0.d }, p0, [x8]
; CHECK-NEXT:    ret
  %base_load = getelementptr <vscale x 2 x i64>, ptr %base, i64 8
  %base_load_bc = bitcast ptr %base_load to ptr
  %data = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1> %mask,
                                                                  ptr %base_load_bc)
  %base_store = getelementptr <vscale x 2 x i64>, ptr %base, i64 -9
  %base_store_bc = bitcast ptr %base_store to ptr
  call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data,
                                            <vscale x 2 x i1> %mask,
                                            ptr %base_store_bc)
  ret void
}

; 2-lane non-temporal load/stores


define void @test_masked_ldst_sv2i64(ptr %base, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldnt1d { z0.d }, p0/z, [x0, #-8, mul vl]
; CHECK-NEXT:    stnt1d { z0.d }, p0, [x0, #-7, mul vl]
; CHECK-NEXT:    ret
  %base_load = getelementptr <vscale x 2 x i64>, ptr %base, i64 -8
  %base_load_bc = bitcast ptr %base_load to ptr
  %data = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1> %mask,
                                                                  ptr %base_load_bc)
  %base_store = getelementptr <vscale x 2 x i64>, ptr %base, i64 -7
  %base_store_bc = bitcast ptr %base_store to ptr
  call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data,
                                            <vscale x 2 x i1> %mask,
                                            ptr %base_store_bc)
  ret void
}

define void @test_masked_ldst_sv2f64(ptr %base, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldnt1d { z0.d }, p0/z, [x0, #-6, mul vl]
; CHECK-NEXT:    stnt1d { z0.d }, p0, [x0, #-5, mul vl]
; CHECK-NEXT:    ret
  %base_load = getelementptr <vscale x 2 x double>, ptr %base, i64 -6
  %base_load_bc = bitcast ptr %base_load to ptr
  %data = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1> %mask,
                                                                    ptr %base_load_bc)
  %base_store = getelementptr <vscale x 2 x double>, ptr %base, i64 -5
  %base_store_bc = bitcast ptr %base_store to ptr
  call void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double> %data,
                                            <vscale x 2 x i1> %mask,
                                            ptr %base_store_bc)
  ret void
}

; 4-lane non-temporal load/stores.

define void @test_masked_ldst_sv4i32(ptr %base, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldnt1w { z0.s }, p0/z, [x0, #6, mul vl]
; CHECK-NEXT:    stnt1w { z0.s }, p0, [x0, #7, mul vl]
; CHECK-NEXT:    ret
  %base_load = getelementptr <vscale x 4 x i32>, ptr %base, i64 6
  %base_load_bc = bitcast ptr %base_load to ptr
  %data = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1> %mask,
                                                                  ptr %base_load_bc)
  %base_store = getelementptr <vscale x 4 x i32>, ptr %base, i64 7
  %base_store_bc = bitcast ptr %base_store to ptr
  call void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32> %data,
                                            <vscale x 4 x i1> %mask,
                                            ptr %base_store_bc)
  ret void
}

define void @test_masked_ldst_sv4f32(ptr %base, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldnt1w { z0.s }, p0/z, [x0, #-1, mul vl]
; CHECK-NEXT:    stnt1w { z0.s }, p0, [x0, #2, mul vl]
; CHECK-NEXT:    ret
  %base_load = getelementptr <vscale x 4 x float>, ptr %base, i64 -1
  %base_load_bc = bitcast ptr %base_load to ptr
  %data = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1> %mask,
                                                                    ptr %base_load_bc)
  %base_store = getelementptr <vscale x 4 x float>, ptr %base, i64 2
  %base_store_bc = bitcast ptr %base_store to ptr
  call void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float> %data,
                                            <vscale x 4 x i1> %mask,
                                            ptr %base_store_bc)
  ret void
}


; 8-lane non-temporal load/stores.

define void @test_masked_ldst_sv8i16(ptr %base, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldnt1h { z0.h }, p0/z, [x0, #6, mul vl]
; CHECK-NEXT:    stnt1h { z0.h }, p0, [x0, #7, mul vl]
; CHECK-NEXT:    ret
  %base_load = getelementptr <vscale x 8 x i16>, ptr %base, i64 6
  %base_load_bc = bitcast ptr %base_load to ptr
  %data = call <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1> %mask,
                                                                  ptr %base_load_bc)
  %base_store = getelementptr <vscale x 8 x i16>, ptr %base, i64 7
  %base_store_bc = bitcast ptr %base_store to ptr
  call void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16> %data,
                                            <vscale x 8 x i1> %mask,
                                            ptr %base_store_bc)
  ret void
}

define void @test_masked_ldst_sv8f16(ptr %base, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldnt1h { z0.h }, p0/z, [x0, #-1, mul vl]
; CHECK-NEXT:    stnt1h { z0.h }, p0, [x0, #2, mul vl]
; CHECK-NEXT:    ret
  %base_load = getelementptr <vscale x 8 x half>, ptr %base, i64 -1
  %base_load_bc = bitcast ptr %base_load to ptr
  %data = call <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1> %mask,
                                                                   ptr %base_load_bc)
  %base_store = getelementptr <vscale x 8 x half>, ptr %base, i64 2
  %base_store_bc = bitcast ptr %base_store to ptr
  call void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half> %data,
                                            <vscale x 8 x i1> %mask,
                                            ptr %base_store_bc)
  ret void
}

define void @test_masked_ldst_sv8bf16(ptr %base, <vscale x 8 x i1> %mask) nounwind #0 {
; CHECK-LABEL: test_masked_ldst_sv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldnt1h { z0.h }, p0/z, [x0, #-1, mul vl]
; CHECK-NEXT:    stnt1h { z0.h }, p0, [x0, #2, mul vl]
; CHECK-NEXT:    ret
  %base_load = getelementptr <vscale x 8 x bfloat>, ptr %base, i64 -1
  %base_load_bc = bitcast ptr %base_load to ptr
  %data = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ldnt1.nxv8bf16(<vscale x 8 x i1> %mask,
                                                                      ptr %base_load_bc)
  %base_store = getelementptr <vscale x 8 x bfloat>, ptr %base, i64 2
  %base_store_bc = bitcast ptr %base_store to ptr
  call void @llvm.aarch64.sve.stnt1.nxv8bf16(<vscale x 8 x bfloat> %data,
                                             <vscale x 8 x i1> %mask,
                                             ptr %base_store_bc)
  ret void
}

; 16-lane non-temporal load/stores.

define void @test_masked_ldst_sv16i8(ptr %base, <vscale x 16 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv16i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldnt1b { z0.b }, p0/z, [x0, #6, mul vl]
; CHECK-NEXT:    stnt1b { z0.b }, p0, [x0, #7, mul vl]
; CHECK-NEXT:    ret
  %base_load = getelementptr <vscale x 16 x i8>, ptr %base, i64 6
  %base_load_bc = bitcast ptr %base_load to ptr
  %data = call <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1> %mask,
                                                                  ptr %base_load_bc)
  %base_store = getelementptr <vscale x 16 x i8>, ptr %base, i64 7
  %base_store_bc = bitcast ptr %base_store to ptr
  call void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8> %data,
                                            <vscale x 16 x i1> %mask,
                                            ptr %base_store_bc)
  ret void
}

; 2-element non-temporal loads.
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1>, ptr)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1>, ptr)

; 4-element non-temporal loads.
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1>, ptr)
declare <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1>, ptr)

; 8-element non-temporal loads.
declare <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1>, ptr)
declare <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1>, ptr)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.ldnt1.nxv8bf16(<vscale x 8 x i1>, ptr)

; 16-element non-temporal loads.
declare <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1>, ptr)

; 2-element non-temporal stores.
declare void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, ptr)
declare void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, ptr)

; 4-element non-temporal stores.
declare void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, ptr)
declare void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, ptr)

; 8-element non-temporal stores.
declare void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, ptr)
declare void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, ptr)
declare void @llvm.aarch64.sve.stnt1.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x i1>, ptr)

; 16-element non-temporal stores.
declare void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, ptr)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }
