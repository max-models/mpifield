---
title: Quickstart
description: Run the command line entry point of mpifield.
---

First, ensure that `mpifield` is [installed](/mpifield/getting-started/installation/).

## Basic usage

After installation, you can run the application with:

```bash
mpifield
```

which calls `mpifield.main.main` and prints a greeting.

## Use the package from Python

```python
from mpifield.main import main

main()
```

See the [API reference](/mpifield/api/mpifield/) for every module and function.

## Profiling

mpifield's coarse operations are [scope-profiler](https://github.com/max-models/scope-profiler)
regions (`mpifield.main` here), tagged with the opt-in tag `mpifield`: they are
off by default and cost a fraction of a microsecond per call. A code that profiles itself with
scope-profiler turns them on with the tag:

```python
import scope_profiler as sp

sp.setup(enable_tags=["mpifield"])
```

or without editing the code, with `SCOPE_PROFILER_ENABLE_TAGS=mpifield`,
`scope-profiler run --enable-tag mpifield` or `enable_tags = ["mpifield"]` in the
TOML configuration. New regions use `@sp.profile("mpifield.<name>", tags=PROFILE_TAGS)`
with `PROFILE_TAGS` from the package's `_profiling` module.
