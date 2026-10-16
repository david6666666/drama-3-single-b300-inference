# M128 native FC2 arithmetic gate

One expert205 at layer0, changing valid0..8, eager andgraph. All 39 exact checks pass. Original checkpoint FP8 weights/scales, K32 instruction reductions and four K128 dequant blocks remain unchanged. Only MMA-M/epilogue shape and the corresponding four-slice TMEM-to-shared store coordinates change. Output BF16 conversion and native TMA commit stay intact.

CPU/GPU CUBINs match; actual384threads, 74664bytes shared/CTA, zero stack/local/spill. Four160-register consumer warps plus eight48-register roles budget32768registers perCTA; this is a capacity argument, not achieved residency or performance. No timing or full model/service quality claim follows.
