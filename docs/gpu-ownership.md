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
Capture installs and fences the exact supervisor identity before observing a
non-empty compute or graphics baseline. It retains only a bounded process count
for diagnostics; it never exposes NVML PIDs. Every PID that can be read from
procfs is checked for strict descent (maximum ancestry depth 64) from the captured
supervisor PID and start time. A positively proved supervisor descendant already
using the GPU blocks capture.

The proof is intentionally asymmetric: claiming service ownership is strict,
but identifying every foreign GPU client is not a goal. A non-supervisor NVML
PID whose procfs record or ancestry is missing, unreadable, malformed, dead, or
otherwise unconnected is ignored as unknown. It is never reported as owned and
is never signaled. Readable foreign ancestry is also excluded. The captured
supervisor itself remains strict: disappearance, unreadability, malformed state,
or PID reuse invalidates public proof operations and makes cleanup return false.

Residency compares two positively proved owned identity sets and requires a
nonempty identical set; unrelated and unknown NVML changes do not matter. An
owned-set change fails immediately. Cleanup is conservative and bounded: with a
stable supervisor it succeeds only when two observations contain no positively
owned PID. Seeing a positively owned runner in either observation blocks cleanup.
Unsupported NVML APIs, invalid device identity, and driver errors still fail.
Residency requires at least one owned runner and reclassifies a second NVML
sample, including when its numeric PID set is unchanged, before accepting it.
Cleanup returns a boolean: it reports true only when the exact supervisor
identity fences both sides of a clean NVML sample and no current PID is an owned
descendant. Foreign or unknown processes may enter or leave. A missing, unreadable,
malformed, or reused supervisor returns false; this class cannot independently
prove process-group cleanup. The harness separately fences its owned process
group. The proof never signals a process.

`proc_root` must be an authoritative procfs in the **same host PID namespace**
used by NVML. In a container this requires the appropriate host PID mapping
and proc mount, and the caller must provide the supervisor PID in that
namespace. The implementation never silently translates container-local PIDs.
The captured `supervisor_identity` is immutable and is the identity used by
the callback-style `identity()`, `residency()`, and `cleanup()` checks.

This is shared-GPU service-ownership evidence only. It does not prove physical
GPU monopoly, uncontended throughput, VRAM availability, measured capacity, or
production E2E. The candidate image is unchanged, the
production image is not present here, and this work includes no benchmarking,
profiling, or runtime-version approval.
