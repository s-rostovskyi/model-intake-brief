"""Live view of the LangGraph pipeline: topology, run state, and HTML rendering.

The view has three parts, kept apart so each can be tested without Streamlit:

* ``topology()`` reads the nodes and edges from the compiled graph, so the picture
  cannot drift from the code.
* ``PipelineRun`` folds the events emitted by the graph (see ``intake.graph``) into
  a per-node state. Replaying a recorded run is applying the same events again.
* ``render_html()`` turns a ``PipelineRun`` into markup styled by ``assets/pipeline.css``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import Any

START, END = '__start__', '__end__'

NODE_META = {
    START: ('Input', 'Model ID + mode'),
    'validate_and_normalize_target': ('Validate target', 'Check ID, pick route'),
    'load_checklist': ('Load checklist', 'Fixed evidence items'),
    'collect_sources': ('Collect sources', 'Tavily · Jina · HF Hub'),
    'retrieve_regulatory_contexts': ('Regulatory context', 'Embeddings · Pinecone'),
    'assess_evidence': ('Assess evidence', 'LLM + verification'),
    'render_report': ('Render report', 'Markdown brief'),
    END: ('Report', 'Result'),
}

STATE_LABEL = {'pending': 'Waiting', 'running': 'Running', 'done': 'Done', 'error': 'Failed', 'skipped': 'Skipped'}
STEP_MARK = {'running': '●', 'done': '✓', 'warn': '!', 'error': '×', 'skip': '–'}


@dataclass(frozen=True)
class Topology:
    order: tuple[str, ...]                    # START, nodes in execution order, END
    edges: tuple[tuple[str, str, bool], ...]  # (source, target, conditional)

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.order[1:-1]

    @property
    def bypasses(self) -> tuple[tuple[str, str], ...]:
        """Conditional edges that jump over at least one node, widest first."""
        index = {name: i for i, name in enumerate(self.order)}
        jumps = [(s, t) for s, t, cond in self.edges if cond and index[t] - index[s] > 1]
        return tuple(sorted(jumps, key=lambda pair: index[pair[0]] - index[pair[1]]))


def topology(app: Any = None) -> Topology:
    if app is None:
        from .graph import APP as app
    drawable = app.get_graph()
    names = list(drawable.nodes)
    edges = tuple((e.source, e.target, bool(e.conditional)) for e in drawable.edges)
    indegree = {name: 0 for name in names}
    for _, target, _ in edges:
        indegree[target] += 1
    order: list[str] = []
    ready = [name for name in names if indegree[name] == 0]
    while ready:
        name = ready.pop(0)
        order.append(name)
        for source, target, _ in edges:
            if source == name:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
    return Topology(tuple(order), edges)


@dataclass
class Step:
    label: str
    status: str = 'running'     # running | done | warn | error | skip
    detail: str = ''
    t0: float = 0.0
    t1: float | None = None


@dataclass
class NodeRun:
    status: str = 'pending'     # pending | running | done | error | skipped
    t0: float | None = None
    t1: float | None = None
    steps: list[Step] = field(default_factory=list)
    data: list[tuple[str, str]] = field(default_factory=list)

    @property
    def seconds(self) -> float | None:
        if self.t0 is None or self.t1 is None:
            return None
        return max(self.t1 - self.t0, 0.0)


class PipelineRun:
    """State of one run, built by applying events in order."""

    def __init__(self, topo: Topology, note: str = ''):
        self.topo = topo
        self.nodes = {name: NodeRun() for name in topo.nodes}
        self.started = False
        self.finished = False
        self.elapsed: float | None = None
        self.taken: set[tuple[str, str]] = set()   # bypass edges actually used
        self.note = note                           # e.g. "Demo pace"
        self._last_t = 0.0

    # -- reducing events ---------------------------------------------------
    def apply(self, event: dict[str, Any], t: float) -> None:
        self.started = True
        self._last_t = t
        node = self.nodes.get(event.get('node', ''))
        kind = event.get('type')
        if node is None:
            return
        if kind == 'node_started':
            self._mark_skipped_before(event['node'])
            node.status, node.t0 = 'running', t
        elif kind == 'step':
            self._apply_step(node, event, t)
        elif kind == 'node_finished':
            node.status = 'error' if event.get('status') == 'error' else 'done'
            node.t1 = t
            node.data = [(str(k), str(v)) for k, v in event.get('data') or []]
            for step in node.steps:
                if step.status == 'running':
                    step.status, step.t1 = ('error' if node.status == 'error' else 'done'), t

    def _apply_step(self, node: NodeRun, event: dict[str, Any], t: float) -> None:
        label, status = str(event.get('label', '')), event.get('status', 'done')
        if status == 'start':
            node.steps.append(Step(label=label, t0=t))
            return
        for step in reversed(node.steps):
            if step.label == label and step.status == 'running':
                break
        else:
            step = Step(label=label, t0=t)
            node.steps.append(step)
        step.status, step.detail, step.t1 = status, str(event.get('detail') or ''), t

    def _mark_skipped_before(self, name: str) -> None:
        order = self.topo.order
        position = order.index(name)
        previous = None
        for earlier in order[1:position]:
            run = self.nodes[earlier]
            if run.status == 'pending':
                run.status = 'skipped'
            elif run.status in ('done', 'error'):
                previous = earlier
        if previous and order.index(previous) < position - 1:
            self.taken.add((previous, name))

    def finish(self, t: float | None = None) -> None:
        t = self._last_t if t is None else t
        for run in self.nodes.values():
            if run.status == 'running':
                run.status, run.t1 = 'error', t
            elif run.status == 'pending' and self.started:
                run.status = 'skipped'
        self.finished, self.elapsed = True, t

    # -- derived -----------------------------------------------------------
    @property
    def status(self) -> str:
        if not self.started:
            return 'idle'
        if any(run.status == 'error' for run in self.nodes.values()):
            return 'error'
        return 'done' if self.finished else 'running'

    def current(self) -> str | None:
        return next((n for n, run in self.nodes.items() if run.status == 'running'), None)

    def element_state(self, name: str) -> str:
        if name == START:
            return 'done' if self.started else 'pending'
        if name == END:
            report = self.nodes.get('render_report')
            return {'done': 'done', 'error': 'error'}.get(report.status if report else '', 'pending')
        return self.nodes[name].status

    def edge_state(self, source: str, target: str) -> str:
        src, dst = self.element_state(source), self.element_state(target)
        if dst == 'skipped':
            return 'skip'
        if src != 'done':
            return 'idle'
        if dst == 'running':
            return 'flow'
        return 'done' if dst in ('done', 'error') else 'idle'


def replay(topo: Topology, events: list[tuple[float, dict[str, Any]]], note: str = '') -> PipelineRun:
    """Build the final state of a recorded run without any waiting."""
    run = PipelineRun(topo, note)
    for t, event in events:
        run.apply(event, t)
    if events:
        run.finish(events[-1][0])
    return run


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_W_TERM, _W_EDGE, _W_NODE = 0.45, 0.30, 1.6


def _columns(order: tuple[str, ...]) -> tuple[list[float], list[float]]:
    """Grid weights and the x centre (0..1000) of every element column."""
    weights: list[float] = []
    centres: list[float] = []
    for i, name in enumerate(order):
        weights.append(_W_TERM if name in (START, END) else _W_NODE)
        if i < len(order) - 1:
            weights.append(_W_EDGE)
    total, cursor = sum(weights), 0.0
    for w in weights:
        centres.append((cursor + w / 2) / total * 1000)
        cursor += w
    return weights, centres


def _seconds(value: float | None) -> str:
    if value is None or value < 0.005:
        return ''
    return f'{value * 1000:.0f} ms' if value < 1 else f'{value:.1f} s'


def _meta(name: str) -> tuple[str, str]:
    return NODE_META.get(name) or (name.replace('_', ' ').capitalize(), '')


def _card(name: str, run: PipelineRun, index: int) -> str:
    title, subtitle = _meta(name)
    state = run.element_state(name)
    node = run.nodes.get(name)
    timing = _seconds(node.seconds) if node else ''
    badge = f'<span class="pv-badge">{index}</span>' if node else ''
    status = escape(STATE_LABEL.get(state, ''))
    if state == 'running' and node and node.t0 is not None:
        timing = 'running…'
    steps = data = ''
    if node and node.steps:
        rows = ''.join(
            f'<li data-state="{escape(s.status)}"><b>{STEP_MARK.get(s.status, "·")}</b>'
            f'<span>{escape(s.label)}<em>{escape(s.detail)}</em></span>'
            f'<time>{escape(_seconds(None if s.t1 is None else s.t1 - s.t0))}</time></li>'
            for s in node.steps)
        steps = f'<ul class="pv-steps">{rows}</ul>'
    if node and node.data:
        rows = ''.join(f'<li><span>{escape(k)}</span><b>{escape(v)}</b></li>' for k, v in node.data)
        heading = 'Failure' if node.status == 'error' else 'Passed on'
        data = f'<div class="pv-out"><p>{heading}</p><ul>{rows}</ul></div>'
    elif state == 'skipped':
        data = '<p class="pv-skip">Not run: the pipeline went straight to the renderer.</p>'
    return (f'<div class="pv-col"><div class="pv-card" data-state="{state}">'
            f'<div class="pv-top">{badge}<span class="pv-state">{status}</span></div>'
            f'<h4>{escape(title)}</h4><p>{escape(subtitle)}</p>'
            f'<div class="pv-time">{escape(timing)}</div></div>{steps}{data}</div>')


def _terminal(name: str, run: PipelineRun) -> str:
    title, subtitle = _meta(name)
    state = run.element_state(name)
    return (f'<div class="pv-col pv-term"><div class="pv-pill" data-state="{state}">'
            f'<b>{escape(title)}</b><span>{escape(subtitle)}</span></div></div>')


def _arcs(run: PipelineRun, centres: list[float]) -> str:
    """Conditional routes that skip nodes, drawn as bordered boxes (st.html strips inline SVG)."""
    order = run.topo.order
    position = {name: 2 * i for i, name in enumerate(order)}   # element column in the grid
    bypasses = run.topo.bypasses
    if not bypasses:
        return ''
    arcs = []
    for rank, (source, target) in enumerate(bypasses):
        x0, x1 = centres[position[source]] / 10, centres[position[target]] / 10
        top = 8 + rank * 17
        taken = 'taken' if (source, target) in run.taken else 'idle'
        arcs.append(f'<div class="pv-arc" data-state="{taken}" style="left:{x0:.2f}%;width:{x1 - x0:.2f}%;top:{top}px">'
                    f'<span>on error</span></div>')
    tips = ''.join(f'<i class="pv-tip" style="left:{centres[position[t]] / 10:.2f}%"></i>'
                   for t in sorted({t for _, t in bypasses}))
    return f'<div class="pv-arcs">{"".join(arcs)}{tips}</div>'


def _header(run: PipelineRun) -> str:
    status = run.status
    if status == 'idle':
        text = 'Ready. Start an analysis to watch the data move through the graph.'
    elif status == 'running':
        current = run.current()
        text = f'Running: {_meta(current)[0]}' if current else 'Running'
    elif status == 'done':
        text = f'Completed in {_seconds(run.elapsed)}'
    else:
        text = f'Stopped with an error after {_seconds(run.elapsed)}'
    note = f'<span class="pv-note">{escape(run.note)}</span>' if run.note else ''
    return f'<div class="pv-head" data-state="{status}"><span class="pv-dot"></span><b>{escape(text)}</b>{note}</div>'


def render_html(run: PipelineRun) -> str:
    order = run.topo.order
    weights, centres = _columns(order)
    template = ' '.join(f'{w}fr' for w in weights)
    cells, index = [], 0
    for i, name in enumerate(order):
        if name in (START, END):
            cells.append(_terminal(name, run))
        else:
            index += 1
            cells.append(_card(name, run, index))
        if i < len(order) - 1:
            cells.append(f'<div class="pv-edge" data-state="{run.edge_state(name, order[i + 1])}"><i></i></div>')
    legend = ('<div class="pv-legend"><span data-state="done">done</span><span data-state="running">running</span>'
              '<span data-state="error">failed</span><span data-state="skipped">skipped</span>'
              '<span class="pv-dash">dashed: conditional route</span></div>')
    return (f'<div class="pv" data-run="{run.status}">{_header(run)}{_arcs(run, centres)}'
            f'<div class="pv-grid" style="grid-template-columns:{template}">{"".join(cells)}</div>{legend}</div>')
