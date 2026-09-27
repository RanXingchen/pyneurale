# Optional experiment presentation API

`neurale.experiments.presentation` is the optional concrete presentation and
input layer for the four experiment paradigms. Importing the module remains
lightweight; the native extension and its GLFW, OpenGL, FreeType, and HarfBuzz
dependencies are loaded only when a native symbol is used.

Python wrappers and configurations below are generated from docstrings. Native
pybind11-only types remain listed explicitly so the headless documentation
build does not load the optional extension to enumerate them.

```{eval-rst}
.. py:module:: neurale.experiments.presentation

.. py:function:: dependency_versions()

   Return the versions of GLFW, the OpenGL compatibility baseline, FreeType,
   and HarfBuzz compiled into the native extension when presentation support is enabled.

.. py:function:: renderer_monotonic_now_ns()

   Return the renderer monotonic clock in integer nanoseconds. Accessing this
   function loads the native extension.
```

## Runtime status and evidence

`PresentationRuntimeStatus` reports window, configuration, thread, OpenGL,
text, capacity, cancellation, and rendering-runtime outcomes. It is distinct
from `neurale.experiments.PresentationStatus`, which describes the experiment
semantic outcome of one presentation request.

```{eval-rst}
.. py:class:: PresentationRuntimeStatus
.. py:class:: SoftwarePresentationTimes
.. py:class:: PresentationResourceStats
.. py:class:: PresentationEnvironment
```

Software submit and swap-return timestamps are presentation evidence only; they
are not physical pixel or photon onset.

## Center-Out

```{eval-rst}
.. autoclass:: neurale.experiments.presentation.CenterOutPresenter
   :members:
   :undoc-members:

.. autoclass:: neurale.experiments.presentation.CenterOutSessionPresentationBridge
   :members:
   :undoc-members:

.. autoclass:: neurale.experiments.presentation.CenterOutPresentationConfig
   :members:
   :undoc-members:

.. autoclass:: neurale.experiments.presentation.CenterOutPresentationTheme
   :members:
   :undoc-members:

.. py:class:: CenterOutPresentationControlEvent
.. py:class:: CenterOutPresentationControlKind
```

The presenter consumes experiment snapshots and never performs containment,
target selection, state transitions, assistance, or guidance.

## WebGrid

```{eval-rst}
.. py:class:: WebGridPresenter

   .. py:method:: open(task, config, renderer_origin_ns, experiment_origin_ns)
   .. py:method:: update(snapshot, pointer, last_selection=None)
   .. py:method:: pump_events()
   .. py:method:: poll_input()
   .. py:method:: render(requested_ns, intended_ns)
   .. py:method:: cancel()
   .. py:method:: close()
   .. py:property:: resource_stats
   .. py:property:: environment

.. py:class:: WebGridPresentationConfig

   Requires a `WebGridPresentationStyle`. Window, monitor, swap interval,
   input capacity, selection button, and fullscreen settings are optional.

.. py:class:: WebGridPresentationStyle

   Requires `pointer_radius`; colors and circle-segment count are optional.

.. py:class:: WebGridPresentationInput
.. py:class:: WebGridPresentationInputKind
```

`last_selection` is the typed experiment `WebGridSelectionRecord`; presentation
does not derive correctness or create a second source of task truth.

## Speech

```{eval-rst}
.. py:class:: SpeechPresenter

   .. py:method:: open(catalog, config, renderer_origin_ns, experiment_origin_ns)
   .. py:method:: present(request, attempt_ns)
   .. py:method:: pump_events()
   .. py:method:: poll_control()
   .. py:method:: poll_outcome()
   .. py:method:: reset()
   .. py:method:: cancel()
   .. py:method:: close()
   .. py:property:: resource_stats
   .. py:property:: environment

.. py:class:: SpeechPresentationConfig

   Requires `SpeechTextConfig`. Style, window geometry, monitor, swap interval,
   input/outcome capacities, and fullscreen settings are optional.

.. py:class:: SpeechPresentationStyle
.. py:class:: SpeechTextConfig

   Requires `font_path`; face index, glyph size, atlas dimensions, and text/glyph
   capacities are optional.

.. py:class:: SpeechPresentationResult
.. py:class:: SpeechPresentationEvidence
.. py:class:: SpeechPresentationControlEvent
.. py:class:: SpeechPresentationControlKind
```

Speech text and glyph coverage are prepared before timed presentation. The
presenter preserves experiment request, trial, and stimulus identity and keeps
intended time separate from software-presented time.

## SSVEP

```{eval-rst}
.. autoclass:: neurale.experiments.presentation.SSVEPDisplay
   :members:
   :undoc-members:

.. autoclass:: neurale.experiments.presentation.SSVEPDisplayConfig
   :members:
   :undoc-members:
```

The display uses the task's target frequencies and the configured window
geometry. The {doc}`SSVEP guide <../user_guide/ssvep>` describes the complete
experiment and the limits of software presentation timestamps.

## Shared immutable presentation values

```{eval-rst}
.. py:class:: AspectPolicy
.. py:class:: Color
.. py:class:: LogicalRect
.. py:class:: Point2D
.. py:class:: WindowSize
```

The private native surface, coordinate mapper, input queue, text atlas, and
OpenGL resources are not public Python APIs.
