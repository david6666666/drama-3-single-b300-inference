# M128 native FC2 arithmetic gate

One expert205 at layer0, changing valid0..8, eager andgraph. All 39 exact checks pass. Original checkpoint FP8 weights/scales, K32 instruction reductions and four K128 dequant blocks remain unchanged. Only MMA-M/epilogue shape and the corresponding four-slice TMEM-to-shared store coordinates change. Output BF16 conversion and native TMA commit stay intact.

CPU/GPU CUBINs match; actual384threads, 74664bytes shared/CTA, zero stack/local/spill. Correction: the 160/48 source-role hints do not describe actual register allocation. This CUBIN has REG31 and no SASS register-reconfiguration instruction. The earlier two-CTA capacity inference is withdrawn; see ../register-lowering-audit/report.md. No timing or full model/service quality claim follows.
