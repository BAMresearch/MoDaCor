# Software guidelines

## Data and validation

- Use `attrs` fields, converters, and validators for classes that need input
  validation.
- Use Pint for unit conversion and algebra.
- In processing modules, use `BaseData` arithmetic where it supports the
  scientific operation; do not strip arrays and recreate unit or uncertainty
  propagation.
- Low-level geometry/model kernels may accept numerical arrays, but their module
  adapter must restore explicit `BaseData` semantics.

## Public behavior

- Keep configuration in `ProcessStepDescriber.arguments` and validate unknown
  keys and top-level types during pipeline loading.
- Declare exact source, processing-read, and processing-write dependencies.
- Raise explicit unsupported-capability errors rather than silently using a
  partial fallback.
- Maintain backwards compatibility deliberately and document deprecations.

## Style and scope

- Prefer focused changes and tests.
- Keep format handling out of numerical kernels and instrument policy out of
  the package core.
- Avoid broad output in tests and tools; report step IDs, counts, and focused
  errors.
- Follow the repository lint and import-order configuration.
