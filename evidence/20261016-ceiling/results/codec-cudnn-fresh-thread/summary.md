# Fresh native cuDNN plan selection: exact gate failed

The same-thread predecessor is cache-confounded and is retained with an explicit correction. Installed pinned PyTorch source places thread-local cache lookup before the benchmark branch; the benchmark flag is absent from its key. This separately frozen probe constructs each arm on a newly created native thread, serially, without clearing shared caches. Native thread IDs differ. Source and original codec parameter SHA/dtype/storage are unchanged; both arms cache the same native BF16 Linear values.

Fresh selection changes the first B1/F8 PCM, maximum absolute difference0.00390625; the original exact gate fails and no timing follows. Both arms remain BF16 with original model weights. This is not a WER/SIM failure or a proof that every alternative native convolution algorithm is unacceptable. A separately declared numerical/quality study would be required to proceed. No runtime promotion; priority returns to the larger model execution budget. Preparation (control18.91s/candidate3.22s) includes ordered compilation/cache effects and is not a cold-start A/B claim. Raw protocol, hashes, checks and log are retained.

Primary cache source: https://raw.githubusercontent.com/pytorch/pytorch/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cudnn/Conv_v8.cpp . No successful service gain is asserted.
