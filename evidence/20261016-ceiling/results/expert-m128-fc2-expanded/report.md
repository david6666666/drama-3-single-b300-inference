# M128 native FC2 arithmetic gate

All256expert weights at layers0/19/39,15originalmetadata scenarios,31,457,280BF16 comparisons across eager+graph plus padding/input/weights/RNG, samegraph dynamic metadata. All 306 exact checks pass. Original checkpoint FP8 weights/scales, K32 instruction reductions and four K128 dequant blocks remain unchanged. Only MMA-M/epilogue shape and the corresponding four-slice TMEM-to-shared store coordinates change. Output BF16 conversion and native TMA commit stay intact.

CPU/GPU CUBINs match; actual384threads, 74664bytes shared/CTA, zero stack/local/spill. Four160-register consumer warps plus eight48-register roles budget32768registers perCTA; this is a capacity argument, not achieved residency or performance. No timing or full model/service quality claim follows.
