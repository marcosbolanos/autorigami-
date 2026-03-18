# autorigami-

Automatic generation of DNA origamis leveraging a C++ optimizer.

## Python development environment (uv)

This repository uses a `src/` Python package layout and can be installed in editable mode.

1. Create/sync the environment:

```bash
uv sync
```

2. Install the package in editable mode:

```bash
uv pip install -e .
```

3. Install and run dev tooling:

```bash
uv add --dev ruff pyright
uv run ruff check .
uv run pyright
```

Runtime 3D packages are included in the project dependencies (`trimesh`, `numpy`, `scipy`).

## Build C++ binaries

From the repository root:

```bash
git submodule update --init --recursive
cmake -S surface-filling-curve-flows -B surface-filling-curve-flows/build
cmake --build surface-filling-curve-flows/build -j6
```

Expected binaries:

- `surface-filling-curve-flows/build/bin/curve_on_surface`
- `surface-filling-curve-flows/build/bin/mesh_deformation`

## Python wrapper around C++ executables

Module: `autorigami_cpp`

- `build_surface_filling_curve_flows(...)` configures/builds C++ code.
- `run_surface_filling_binary(...)` runs either binary from Python.
- `run_capsule_scene(...)` handles input/output orchestration entirely in Python.

### End-to-end run managed by Python

The Python wrapper now handles both inputs and outputs:

1. Uses a model you provide via CLI/Python (`--model`), or defaults to the bundled capsule asset at `src/autorigami_cpp/assets/simple_capsule_basic_watertight.obj`.
2. Creates a new timestamped directory under `outputs/` named `outputs_DATETIME`.
3. Copies the selected mesh into that run folder.
4. Generates a scene file in that run folder.
5. Runs the selected C++ binary from that run folder so generated files stay isolated.

CLI usage:

```bash
uv run autorigami-cpp --build --binary curve_on_surface
uv run autorigami-cpp --build --binary curve_on_surface --model /path/to/mesh.obj
```

Use `--arg` repeatedly to pass through flags to the C++ executable unchanged.

Python usage:

```python
from autorigami_cpp import build_surface_filling_curve_flows, run_capsule_scene

build_surface_filling_curve_flows(build_dir="build", jobs=6)
run_paths, _ = run_capsule_scene(
    binary_name="curve_on_surface",
    model_path="/path/to/mesh.obj",  # optional
    build_dir="build",
    extra_args=["--headless"],
)
print(run_paths.output_dir)
```

`extra_args` is forwarded directly to the C++ executable so flags can be passed through unchanged.
