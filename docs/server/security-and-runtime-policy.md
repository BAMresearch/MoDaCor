# Security and runtime policy

The runtime service does not provide authentication or TLS termination. Put any
network-facing deployment behind the facility's authenticated reverse proxy,
ingress, or service mesh.

`trusted` policy supports local, single-user development and allows project
paths and reviewed in-process customization. It is not appropriate for an
untrusted network boundary.

`restricted` policy constrains:

- pipeline path loading;
- filesystem module discovery;
- arbitrary custom I/O imports;
- source read roots and sink write roots;
- session count;
- submitted pipeline size; and
- expanded pipeline step count; and
- buffer upload size.

The authored YAML byte limit and expanded-step limit address different risks:
a compact `step_blocks` document can be small while expanding into many
ordinary execution nodes. Configure `max_expanded_pipeline_steps` on
`RuntimePolicy`, or `--max-expanded-pipeline-steps` on the server command, to
bound that expansion before modules are instantiated.

Restricted mode controls what the service will resolve; it does not turn
arbitrary Python code into safe input. Never expose endpoints that accept
Python source, pickles, bytecode, or serialized classes.

Package reviewed custom steps into the deployed environment, import and
register them in the server launcher, and rebuild the deployment through the
normal review process. Treat input paths, output paths, optional backends, and
resource limits as deployment configuration rather than pipeline science.
