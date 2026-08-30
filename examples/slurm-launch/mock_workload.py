#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proxy for the training workload.

Stands in for Megatron so the workload side of the launch path can be tested
without one. Its own service, measuring itself: it takes no timestamps from the
launch layer and joins to it by identity, through OTEL_RESOURCE_ATTRIBUTES.
"""

import os
import sys
import time
from importlib import import_module

_IMPORTS_STARTED = time.time()


def _import_runtime_dependencies():
    trace_module = import_module("opentelemetry.trace")
    context_module = import_module("opentelemetry.context")
    resources_module = import_module("opentelemetry.sdk.resources")
    trace_sdk_module = import_module("opentelemetry.sdk.trace")
    trace_export_module = import_module("opentelemetry.sdk.trace.export")
    span_utilities_module = import_module("nemo.lens.span_utilities")
    return (
        trace_module,
        context_module.Context,
        resources_module.Resource,
        trace_sdk_module.TracerProvider,
        trace_export_module.BatchSpanProcessor,
        trace_export_module.ConsoleSpanExporter,
        span_utilities_module.emit_span,
        span_utilities_module.linux_process_create_time,
    )


(
    trace,
    Context,
    Resource,
    TracerProvider,
    BatchSpanProcessor,
    ConsoleSpanExporter,
    emit_span,
    linux_process_create_time,
) = _import_runtime_dependencies()

_IMPORTS_FINISHED = time.time()

SERVICE = "nv.dl.training"
STARTUP_SPAN = "nv.dl.training.python_startup"
IMPORTS_SPAN = "nv.dl.training.python_imports"


def exporter():
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter()
    return ConsoleSpanExporter()


def main():
    created = linux_process_create_time()

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE}))
    provider.add_span_processor(BatchSpanProcessor(exporter()))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(SERVICE)

    for name, a, b in (
        (STARTUP_SPAN, created, _IMPORTS_STARTED),
        (IMPORTS_SPAN, _IMPORTS_STARTED, _IMPORTS_FINISHED),
    ):
        emit_span(tracer, name, a, b, context=Context())

    provider.force_flush()
    provider.shutdown()

    print(f"  {STARTUP_SPAN}  {_IMPORTS_STARTED - created:.3f}s", file=sys.stderr)
    print(
        f"  {IMPORTS_SPAN}  {_IMPORTS_FINISHED - _IMPORTS_STARTED:.3f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
