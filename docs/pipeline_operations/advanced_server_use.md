# Advanced server use

This page covers trusted, programmatic runtime-service patterns that go beyond
starting the standard `modacor serve` command. In particular, it explains how
an application or notebook can make project-local `ProcessStep` classes
available for the lifetime of one server process.

## Ephemeral process-step registration

Pipeline YAML contains process-step names, not Python implementations. The
runtime service resolves those names through its in-memory
`ProcessStepRegistry`. A trusted application may import project-local step
classes and register them before the server starts accepting sessions:

```python
# serve_project.py
from modacor.server.api import create_app
from modacor.server.runtime_policy import RuntimePolicy

from my_project.steps import CenterAxis1D, FitPeak1D


app = create_app(runtime_policy=RuntimePolicy.trusted())
registry = app.state.runtime_service.process_step_registry
registry.register(FitPeak1D)
registry.register(CenterAxis1D)
```

Start that application with Uvicorn in the environment containing both
MoDaCor and the project package:

```bash
uvicorn serve_project:app --host 127.0.0.1 --port 8000
```

A submitted pipeline can now refer to the registered class names:

```yaml
steps:
  fit_peak:
    module: FitPeak1D
    configuration:
      with_processing_keys: [sample]
```

The registration is ephemeral in the following precise sense:

- the registry entry exists only in that Python server process;
- restarting the process rebuilds the registry and removes the entry unless
  the launcher registers it again;
- session requests send YAML and data/source registrations, not Python source,
  bytecode, pickles, or class definitions;
- registering a new class does not modify the installed MoDaCor package.

This is useful while developing an instrument pipeline because a local module
can be exercised without first adding it to MoDaCor's curated module exports.
The server still needs to run in the interpreter where the class is importable.

## Notebook-defined steps

A notebook may define a `ProcessStep`, build an app with `create_app()`, register
the class as above, and run that app in a background server thread. Because the
app and registry live in the notebook interpreter, the definition remains
available until that interpreter or server is stopped.

This is different from `LocalRuntimeServer`. `LocalRuntimeServer` launches a
new Python subprocess, so it cannot see classes that exist only in notebook
memory. For that helper, put custom steps in an importable project package and
use a custom server launcher, or install/export the steps normally.

Keep a handle to the Uvicorn `Server` and its thread so the notebook can stop
them explicitly. Use a unique port per active development server and delete or
recreate sessions after changing a step definition; existing processing state
may have been produced by the older implementation.

## Trust and deployment boundary

Only trusted local applications should register arbitrary project code. Do not
add an HTTP endpoint that accepts Python source, pickles, or serialized classes.
Those mechanisms are remote-code-execution interfaces.

The restricted runtime policy intentionally disables filesystem discovery of
unregistered process steps. For shared, containerized, or network-facing
deployments:

1. package the custom steps as reviewed Python code;
2. install that package in the server image;
3. import and explicitly register the approved classes in the server launcher,
   or expose them through MoDaCor's curated module package; and
4. run behind the facility authentication and TLS boundary.

The same pipeline YAML can be used in development and production as long as
the registered class names remain stable.

## Registry lifetime and sessions

The process-step registry belongs to the runtime service, not to an individual
session. All sessions in that server process can resolve a registered class.
Registration should therefore happen before creating sessions, and class names
should be unique. Calling `register(SomeStep, name="FacilityFitPeak")` provides
an explicit registry name when a project-local class name might collide with a
curated MoDaCor step.

Deleting a session removes its pipeline state but does not remove registry
entries. Stopping the server process removes both sessions and ephemeral
registrations.
