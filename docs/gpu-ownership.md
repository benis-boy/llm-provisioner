# Linux GPU ownership proof

`LinuxGPUProof.capture()` is the bounded runtime dependency for the next
SmolLM integration stage. Capture is synchronous by design and must be called
with `asyncio.to_thread`; `identity()`, `residency()`, and `cleanup()` already
offload all NVML and procfs work.

Real capture (`nvml=None`) requires the keyword `host_pid_namespace=True`.
This is an explicit bootstrap/operator attestation, not an automatic claim
that Python can prove from inside an arbitrary container. The operator must
truthfully provide a host-PID procfs mapping and the supervisor PID in that
same namespace (for example, a dedicated hardened host-PID deployment). A
false or missing attestation fails before importing pynvml or reading procfs.
Injected NVML backends are a test-only seam and do not establish production
host-PID evidence.

The proof requires exactly one exposed physical NVML device, the requested
`GPU-*` UUID, MIG disabled (including both current and pending MIG mode), and
one readable compute plus one readable graphics process-list API. API aliases
are tried from newest to oldest only when NVML explicitly reports
`FunctionNotFound` or `NotSupported`; permission and driver failures fail
closed. MIG `NotSupported` is accepted because it precisely means this
hardware cannot expose MIG, while a missing or otherwise failing MIG API does
not.
The capture baseline must have both lists empty. Later GPU PIDs must be strict
descendants (maximum ancestry depth 64) of the captured supervisor PID and
start time. PID reuse, cycles, missing/permission-denied proc entries,
unsupported NVML APIs, and driver errors fail closed. Residency requires a
non-empty process set, snapshots every ancestry record (including leaf start
time and parent) and rereads the complete chain before accepting it. It also
rereads the GPU PID set, so runner additions or removals during proof are not
accepted. Cleanup only reports true when the supervisor is still the same
process and both lists are empty; it never kills a process.

`proc_root` must be an authoritative procfs in the **same host PID namespace**
used by NVML. In a container this requires the appropriate host PID mapping
and proc mount, and the caller must provide the supervisor PID in that
namespace. The implementation never silently translates container-local PIDs.
The captured `supervisor_identity` is immutable and is the identity used by
the callback-style `identity()`, `residency()`, and `cleanup()` checks.

This is ownership evidence only. The candidate image is unchanged, the
production image is not present here, and this work includes no benchmarking,
profiling, or runtime-version approval.
