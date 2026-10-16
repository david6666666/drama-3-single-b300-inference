# M128 native FC2 arithmetic gate

All256expert weights at layers0/19/39,15originalmetadata scenarios,31,457,280BF16 comparisons across eager+graph plus padding/input/weights/RNG, samegraph dynamic metadata. All 306 exact checks pass. Original checkpoint FP8 weights/scales, K32 instruction reductions and four K128 dequant blocks remain unchanged. Only MMA-M/epilogue shape and the corresponding four-slice TMEM-to-shared store coordinates change. Output BF16 conversion and native TMA commit stay intact.

CPU/GPU CUBINs match; actual384threads, 74664bytes shared/CTA, zero stack/local/spill. Correction: the 160/48 source-role hints do not describe actual register allocation. This CUBIN has REG31 and no SASS register-reconfiguration instruction. The earlier two-CTA capacity inference is withdrawn; see ../register-lowering-audit/report.md. No timing or full model/service quality claim follows.
