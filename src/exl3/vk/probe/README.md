# Why `precise` is on the Hadamard accumulators

On 2026-10-10 the decode oracle passed bit-exact on lavapipe but failed on the
Honeykrisp ICD: `rowH128` 67659/294912 exact (maxRel 1.525e25), `colH128`
45814/294912 (maxRel 4.184e22), final f16 290464/294912, bf16 287280/294912 —
while the integer decode (`decodeMcg` 65536/65536, tile f16 294912/294912) was
exact.

`test/sim.c` replays the exact stage-B/C float chains from the same `oracle.bin`
under each hypothesis. The `fma` mode (contract `acc + he*x` into one
correctly-rounded fma) reproduces all six on-device numbers exactly:

rowH128 67659/294912 maxRel 1.525e25; colH128 45814/294912 maxRel 4.184e22;
final f16 290464/294912; final bf16 287280/294912. The `ftz`/`ftzout` modes are
bit-identical to strict (the whole path contains zero denormals), so flush is
ruled out by data.

The Honeykrisp compiler fuses the accumulation into one rounding; lavapipe keeps
`OpFMul`+`OpFAdd` separately rounded. Both are spec-legal without `precise`, so
the qualifier (on both `float acc = 0.0;` in `exl3_decode.comp`) is the fix, not
a workaround. `probe.comp`/`probe.c`/`run.sh` are the on-lavapipe decision tools
(case 8 proves fusion, case 1 closes the denormal question); `precise_decode.comp`
is the earlier staged artifact of the fix, superseded by the real commit.

With `precise`, the on-device rerun on the M1 Max G13C (2026-10-10) is bit-exact
everywhere (maxRel 0.0): `RESULT: PASS`, dmesg GPU-timeout count unchanged.
