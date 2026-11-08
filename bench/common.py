"""Shared helpers for the voice-model benchmarks."""
import os
import subprocess
import time

SUMMARIES = {
    "s080": "Done. The example-agent tests pass, and the branch is ready for you to review.",
    "s120": "I created a worktree for example-frontend from production 4.6.2. The login redirect fix is in, and the build is green.",
    "s200": "The deploy of example-orders-api to uat finished cleanly. Two migrations ran, the health check passed, and error rates "
            "look normal. I can open a pull request against master next, if you want me to.",
    "s300": "Here is where things stand. The example-agent branch now rebases cleanly on production 4.6.2, and all unit tests pass. "
            "The example-bridge service had one flaky test that I retried twice; it passed both times. The voice-gateway changes "
            "are staged but not committed. Should I push them to dev now, or wait?",
}

REQUESTS = {
    "r1": "new worktree in example-frontend from production-4.6.2 for the login redirect bug",
    "r2": "run the tests in example-agent on the dev branch",
    "r3": "open a pull request from uat to master in example-orders-api",
    "r4": "check the build status of example-billing-api on main",
    "r5": "start an agent in example-engine-mcp to fix the timeout in the order handler",
    "r6": "pull the latest master in example-platform and summarise the changes",
    "r7": "new worktree in example-auth from dev for the token refresh issue",
    "r8": "what changed in example-rag and example-bridge since yesterday",
}

REPOS = [
    "example-agent", "example-frontend", "example-api", "example-auth",
    "example-orders-api", "example-app", "example-billing-api", "example-web",
    "example-platform", "example-engine-mcp", "example-bridge", "example-rag",
    "voice-gateway", "example-dashboard",
]
BRANCHES = ["production-4.6.2", "dev", "uat", "master", "main"]

VOCAB_PROMPT = (
    "Repositories: " + ", ".join(REPOS) + ". Branches: " + ", ".join(BRANCHES) + "."
)


def gpu_used_mib():
    """Total used GPU memory (MiB) as nvidia-smi reports it, all processes."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.strip().splitlines()[0])


def proc_gpu_mib(pid=None):
    """GPU memory (MiB) used by one process, from nvidia-smi compute-apps."""
    pid = pid or os.getpid()
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True
    )
    for line in out.strip().splitlines():
        p, mem = [x.strip() for x in line.split(",")]
        if int(p) == pid:
            return int(mem)
    return 0


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
