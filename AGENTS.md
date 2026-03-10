# Repository Guidelines

## Project Structure & Module Organization
This repository is centered on [`LowLight-Portrait-Enhancement/`](/D:/yu_projects/img_pro/LowLight-Portrait-Enhancement), with external checkouts and local tooling kept at the root. Main Python inference code lives in [`LowLight-Portrait-Enhancement/models/`](/D:/yu_projects/img_pro/LowLight-Portrait-Enhancement/models), ONNX export and deployment helpers in [`LowLight-Portrait-Enhancement/deploy/`](/D:/yu_projects/img_pro/LowLight-Portrait-Enhancement/deploy), C++ runtime code in [`LowLight-Portrait-Enhancement/deploy/cpp/`](/D:/yu_projects/img_pro/LowLight-Portrait-Enhancement/deploy/cpp), and smoke tests in [`LowLight-Portrait-Enhancement/tests/`](/D:/yu_projects/img_pro/LowLight-Portrait-Enhancement/tests). Treat [`references/`](/D:/yu_projects/img_pro/references) as vendored dependencies, not a place for new project code.

## Build, Test, and Development Commands
Work from [`LowLight-Portrait-Enhancement/`](/D:/yu_projects/img_pro/LowLight-Portrait-Enhancement).

```bash
python scripts/setup.py --skip-weights
python tests/test_retinexformer.py --image data/sample.png --weights deploy/models/LOL_v2_synthetic.pth
python deploy/export_onnx.py --weights deploy/models/LOL_v2_synthetic.pth --output deploy/models/retinexformer.onnx --simplify --verify
python deploy/verify_onnx.py --onnx deploy/models/retinexformer.onnx --weights deploy/models/LOL_v2_synthetic.pth --benchmark
cmake -S deploy/cpp -B deploy/cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build deploy/cpp/build --config Release
```

Pass `--weights` explicitly: helper scripts currently disagree on whether weights live in `models/` or `deploy/models/`.

## Coding Style & Naming Conventions
Use 4-space indentation in Python and follow existing PEP 8-style spacing, import grouping, and docstring usage. Keep Python modules and scripts in `snake_case`, classes in `PascalCase`, and CLI entry points `argparse`-driven. Match the existing C++14 style in `deploy/cpp`: same-line braces, `snake_case` filenames, headers in `include/` paired with sources in `src/`.

## Testing Guidelines
There is no enforced coverage gate yet; add focused regression tests when changing inference behavior. New Python tests should go under `tests/test_*.py`. For model or export changes, run the Python smoke test and ONNX verification command above. For C++ changes, build `deploy/cpp/build` in `Release` and run the generated `test_engine` binary with a real `.onnx` model and sample image.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects, sometimes with prefixes such as `feat:`, `chore:`, and `refactor:`. Keep commits scoped to one change and mention the affected layer, for example `feat: improve ONNX verification output`. PRs should include a concise summary, validation commands run, any required local paths or weights, and before/after images or benchmark numbers when output quality or performance changes. Do not commit weights, datasets, ONNX artifacts, build outputs, or personal machine paths from `deploy/cpp/CMakeLists.txt`.
