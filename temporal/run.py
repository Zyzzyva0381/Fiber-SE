import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KLEIDIAI = "6787251d9cc2f38a3a6024b11fd7ace10cde4cd9"
PLATFORMS = {
    "xeon": ("xeon", "22", "x86", "sapphirerapids", 50000, 8),
    "ampereone": ("root@84.32.149.103", "0", "i8mm", "native", 10000, 8),
    "ryzen": ("wsl", "4", "x86", "native", 50000, 8),
    "rdk-x5": ("rdk-x5", "0", "dotprod", "native", 5000, 8),
}


def call(command: list[str], capture: bool = False) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=capture).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("platform", choices=PLATFORMS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    host, cpu, kind, march, iterations, repeats = PLATFORMS[args.platform]
    remote = f"/var/tmp/fiber-se-temporal-{args.platform}"
    call(["ssh", host, "mkdir", "-p", remote])
    if kind == "x86":
        call(["scp", ROOT / "x86.cc", ROOT / "x86_core.cc", f"{host}:{remote}/"])
        call(
            [
                "ssh",
                host,
                f"cd {remote} && g++ -O3 -std=c++17 -march={march} x86.cc -o bench",
            ]
        )
        result = call(
            ["ssh", host, f"taskset -c {cpu} {remote}/bench {iterations} {repeats}"],
            True,
        )
    else:
        call(["scp", ROOT / "arm.cc", f"{host}:{remote}/"])
        call(
            [
                "ssh",
                host,
                f"test -d {remote}/kleidiai/.git || git clone -q https://github.com/ARM-software/kleidiai.git {remote}/kleidiai; git -C {remote}/kleidiai checkout -q {KLEIDIAI}",
            ]
        )
        common = [
            "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c",
            "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c",
        ]
        if kind == "i8mm":
            common += [
                "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c",
                "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_asm.S",
            ]
        else:
            common += [
                "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c"
            ]
        sources = " ".join(f"kleidiai/{item}" for item in common)
        macro = "1" if kind == "dotprod" else "0"
        call(
            [
                "ssh",
                host,
                f"cd {remote} && gcc -O3 -mcpu=native -Ikleidiai -c {sources} && g++ -O3 -std=c++17 -mcpu=native -DB553_ARM_DOTPROD={macro} -Ikleidiai arm.cc *.o -o bench",
            ]
        )
        warmup = 500 if kind == "dotprod" else 1000
        result = call(
            [
                "ssh",
                host,
                f"taskset -c {cpu} {remote}/bench {warmup} {iterations} {repeats}",
            ],
            True,
        )
    output = args.output or ROOT.parent / f"results/temporal/{args.platform}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result)


if __name__ == "__main__":
    main()
