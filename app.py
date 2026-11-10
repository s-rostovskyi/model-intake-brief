from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from intake.graph import APP

ROOT = Path(__file__).resolve().parent
SAMPLE_REPORTS = {
    'Qwen/Qwen3-8B': ROOT / 'samples' / 'qwen3-8b.md',
    'gpt-5.2-2025-12-11': ROOT / 'samples' / 'gpt-5.2.md',
}
SAMPLE_TITLES = {
    'Qwen/Qwen3-8B': 'Saved report: Qwen/Qwen3-8B',
    'gpt-5.2-2025-12-11': 'Saved report: gpt-5.2-2025-12-11',
}


def sample_keys() -> list[str]:
    return list(SAMPLE_REPORTS)


def sample_path(sample_key: str) -> Path:
    return SAMPLE_REPORTS[sample_key]


def load_saved_report(sample_key: str) -> str:
    return sample_path(sample_key).read_text(encoding='utf-8')


def build_run_payload(live_mode: bool, provider: str | None, model: str) -> dict[str, Any]:
    return {
        'live_mode': bool(live_mode),
        'provider': provider.strip() if isinstance(provider, str) and provider.strip() else None,
        'model': model.strip(),
    }


def run_graph(live_mode: bool, provider: str | None, model: str) -> dict[str, Any]:
    return APP.invoke(build_run_payload(live_mode, provider, model))


def _download_name(model: str, live_mode: bool) -> str:
    suffix = 'live' if live_mode else 'offline'
    slug = ''.join(ch.lower() if ch.isalnum() else '-' for ch in model).strip('-') or 'report'
    return f'{slug}-{suffix}.md'


def _render_sample_preview(sample_key: str) -> None:
    sample_text = load_saved_report(sample_key)
    st.subheader('Saved report preview')
    st.caption('Opening the page or switching samples does not run the graph.')
    st.markdown(f"**{SAMPLE_TITLES[sample_key]}**")
    st.download_button(
        'Download saved report',
        data=sample_text,
        file_name=sample_path(sample_key).name,
        mime='text/markdown',
        key=f'download-{sample_key}',
    )
    st.markdown(sample_text)


def _render_run_result(result: dict[str, Any], live_mode: bool, provider: str | None, model: str) -> None:
    report_markdown = result.get('report_markdown') or ''
    error = result.get('error')

    st.subheader('Run result')
    st.markdown('**Live run**' if live_mode else '**Offline run**')

    if report_markdown:
        st.markdown(report_markdown)
        st.download_button(
            'Download run report',
            data=report_markdown,
            file_name=_download_name(model, live_mode),
            mime='text/markdown',
            key=f'run-download-{live_mode}-{provider or "offline"}-{model}',
        )
    if error:
        st.error(error)
    elif not report_markdown:
        st.warning('Graph returned no report.')


def main() -> None:
    st.set_page_config(page_title='Model Intake Brief', layout='wide')
    st.title('Model Intake Brief')
    st.caption('Saved reports are safe to open locally. Live runs only happen after an explicit button click.')

    sample_key = st.sidebar.radio(
        'Saved reports',
        options=sample_keys(),
        format_func=lambda key: SAMPLE_TITLES[key],
    )

    with st.sidebar.expander('Selected report details', expanded=True):
        st.write(f"Path: `{sample_path(sample_key).as_posix()}`")
        st.write('Mode: saved report preview')

    left, right = st.columns([1.1, 1])
    with left:
        _render_sample_preview(sample_key)

    with right:
        st.subheader('Run graph')
        st.caption('The same LangGraph workflow runs here after you click Run graph.')
        with st.form('run-form', clear_on_submit=False):
            target_mode = st.radio(
                'Model input format',
                options=['Hugging Face ORG/MODEL', 'Hosted provider + exact model ID'],
            )
            if target_mode == 'Hugging Face ORG/MODEL':
                provider = None
                model = st.text_input('Model', value='Qwen/Qwen3-8B')
            else:
                provider = st.text_input('Provider', value='openai')
                model = st.text_input('Model', value='gpt-5.2-2025-12-11')
            live_mode = st.checkbox('Live run', value=False, help='Reads `.env` only after you click Run graph.')
            submitted = st.form_submit_button('Run graph')

        if submitted:
            progress = st.progress(10, text='Validating inputs...')
            progress.progress(35, text='Running the graph...')
            result = run_graph(live_mode=live_mode, provider=provider, model=model)
            progress.progress(100, text='Report ready.')
            _render_run_result(result, live_mode, provider, model)


if __name__ == '__main__':
    main()
