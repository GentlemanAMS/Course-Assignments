;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sparkle256s_v7m.asm: Size-optimized ARMv7-M implementation of SPARKLE256. ;;
;; This file is part of the SPARKLE submission to NIST's LW Crypto Project.  ;;
;; Version 1.1.2 (2020-10-30), see <http://www.cryptolux.org/> for updates.  ;;
;; Authors: The SPARKLE Group (C. Beierle, A. Biryukov, L. Cardoso dos       ;;
;; Santos, J. Groszschaedl, L. Perrin, A. Udovenko, V. Velichkov, Q. Wang).  ;;
;; License: GPLv3 (see LICENSE file), other licenses available upon request. ;;
;; Copyright (C) 2019-2020 University of Luxembourg <http://www.uni.lu/>.    ;;
;; ------------------------------------------------------------------------- ;;
;; This program is free software: you can redistribute it and/or modify it   ;;
;; under the terms of the GNU General Public License as published by the     ;;
;; Free Software Foundation, either version 3 of the License, or (at your    ;;
;; option) any later version. This program is distributed in the hope that   ;;
;; it will be useful, but WITHOUT ANY WARRANTY; without even the implied     ;;
;; warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  ;;
;; GNU General Public License for more details. You should have received a   ;;
;; copy of the GNU General Public License along with this program. If not,   ;;
;; see <http://www.gnu.org/licenses/>.                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    
    
    THUMB
    PRESERVE8
    
    
    AREA sparkle_code, CODE, READONLY, ALIGN=2
    
    
    EXPORT sparkle256_arm [CODE]
    
    
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;; REGISTER NAMES AND CONSTANTS ;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    
;; register sptr holds the start address of array 'state'
sptr RN r0
;; register cptr holds the start address of array 'rcon'
cptr RN lr
;; register cnt holds the step counter (for loop termination)
cnt RN r12
;; register step holds the number of steps (parameter 'steps')
step RN r1
;; registers c0w to c3w hold round constants from array 'rcon'
c0w RN r2
c1w RN r3
c2w RN r2
c3w RN r3
;; registers tmpx, tmpy hold temporary values
tmpx RN r2
tmpy RN r3
;; registers x0w to y3w hold 8 words from array 'state'
x0w RN r4
y0w RN r5
x1w RN r6
y1w RN r7
x2w RN r8
y2w RN r9
x3w RN r10
y3w RN r11
    
    
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;; MACROS FOR SPARKLE256 ;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    
    MACRO
    PROLOGUE_256
    push    {r4-r12,lr}
    ldr.w   cptr, =RCON
    ldm     sptr, {x0w-y3w}
    MEND
    
    MACRO
    EPILOGUE_256
    stm     sptr, {x0w-y3w}
    pop     {r4-r12,pc}
    MEND
    
    MACRO
    ADD_STEP_CNT_256
    eor     y1w, y1w, cnt
    and     c0w, cnt, #7
    ldr     c0w, [cptr, c0w, lsl #2]
    eor     y0w, y0w, c0w
    MEND
    
    MACRO
    ARX_BOX $xi, $yi, $ci
    add     $xi, $xi, $yi, ror #31
    eor     $yi, $yi, $xi, ror #24
    eor     $xi, $xi, $ci
    add     $xi, $xi, $yi, ror #17
    eor     $yi, $yi, $xi, ror #17
    eors    $xi, $xi, $ci
    adds    $xi, $xi, $yi
    eor     $yi, $yi, $xi, ror #31
    eor     $xi, $xi, $ci
    add     $xi, $xi, $yi, ror #24
    eor     $yi, $yi, $xi, ror #16
    eor     $xi, $xi, $ci
    MEND
    
    MACRO
    LL_TMPX $xi, $xj
    eor     tmpx, $xi, $xj
    eor     tmpx, tmpx, tmpx, lsl #16
    MEND
    
    MACRO
    LL_TMPY $yi, $yj
    eor     tmpy, $yi, $yj
    eor     tmpy, tmpy, tmpy, lsl #16
    MEND
    
    MACRO
    ARXBOX_LAYER_256
    ;; ARX-box computations for the two left-side branches (i.e. x[0]-y[1]).
    ldmia   cptr!, {c0w,c1w}
    ARX_BOX x0w, y0w, c0w
    ARX_BOX x1w, y1w, c1w
    ;; ARX-box computations for the two right-side branches (i.e. x[2]-y[3]).
    ldmia   cptr!, {c2w,c3w}
    ARX_BOX x2w, y2w, c2w
    ARX_BOX x3w, y3w, c3w
    sub     cptr, cptr, #16
    MEND
    
    MACRO
    LINEAR_LAYER_256
    ;; First part of Feistel round: tmpx and tmpy are computed and XORED to the
    ;; y-words and x-words of the right-side branches (i.e. to y[2], y[3] and
    ;; to x[2], x[3]). Note that y[3] and x[3] are stored in register tmpx and
    ;; tmpy (and not in register y3w and x3w) to reduce the execution time of
    ;; the subsequent branch permutation.
    LL_TMPX x0w, x1w
    eor     y2w, y2w, tmpx, ror #16
    eor     tmpx, y3w, tmpx, ror #16
    LL_TMPY y0w, y1w
    eor     x2w, x2w, tmpy, ror #16
    eor     tmpy, x3w, tmpy, ror #16
    ;; Branch permutation: 1-branch left-rotation of the right-side branches
    ;; along with a swap of the left and right branches (via register writes).
    ;; Also combined with the branch permutation is the second Feistel part,
    ;; in which the left-side branches are XORed with the result of the first
    ;; Feistel part.
    mov.w   y3w, y1w
    eor     y1w, y2w, y0w
    mov.w   y2w, y0w
    eor     y0w, tmpx, y3w
    mov.w   x3w, x1w
    eor     x1w, x2w, x0w
    mov.w   x2w, x0w
    eor     x0w, tmpy, x3w
    MEND
    
    
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;; SPARKLE256 PERMUTATION (BRANCH-UNROLLED) ;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    
;; Function prototype:
;; -------------------
;; void sparkle256_arm(uint32_t *state, int steps)
;;
;; Parameters:
;; -----------
;; state: pointer to an uint32_t-array containing the 8 state words
;; steps: number of steps
;;
;; Return value:
;; -------------
;; None
    
sparkle256_arm PROC
    PROLOGUE_256            ;; push callee-saved registers and load state
    mov cnt, #0             ;; initialize step-counter
loop_256                    ;; start of loop
    ADD_STEP_CNT_256        ;; macro to add step-counter to state
    ARXBOX_LAYER_256        ;; macro for the ARXBOX layer
    LINEAR_LAYER_256        ;; macro for the linear layer
    add cnt, #1             ;; increment step-counter
    teq cnt, step           ;; test whether step-counter equals 'steps'
    bne loop_256            ;; if not then branch to start of loop
    EPILOGUE_256            ;; store state and pop callee-saved registers
    ENDP
    
    
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;; SPARKLE ROUND CONSTANTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    
;; This implementation places the round constants in the .data segment, which
;; means they are loaded from RAM during the computation of the ARX-boxes. It
;; would also be possible to place them in the .rodata segment (by replacing
;; the "READWRITE" attribute in the AREA directive below by "READONLY") so that
;; they are loaded from flash, which reduces the RAM consumption by 32 bytes,
;; but may increase the execution time on devices with a high number of flash
;; wait states.
    
    
    AREA sparkle_rcon, DATA, READWRITE, ALIGN=2
    
    
;; round constants
RCON DCD 0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738, \
         0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D
    
    
    END
