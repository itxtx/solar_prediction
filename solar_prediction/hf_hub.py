"""Hugging Face Hub helpers for downloading datasets and uploading run artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve a Hugging Face token from explicit arg, environment, or Colab secrets."""
    if token:
        return token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        colab_token = userdata.get("HF_TOKEN")
        if colab_token:
            return colab_token
    except Exception:
        pass
    return None


def download_dataset_file(
    repo_id: str,
    filename: str = "solar_weather.csv",
    *,
    local_dir: str | Path = "data",
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a dataset file from a Hugging Face dataset repository."""
    from huggingface_hub import hf_hub_download

    resolved_token = resolve_hf_token(token)
    target_dir = Path(local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(target_dir),
        revision=revision,
        token=resolved_token,
    )
    return Path(downloaded)


def download_artifacts_folder(
    repo_id: str,
    path_in_repo: str,
    *,
    local_dir: str | Path = "artifacts",
    token: str | None = None,
) -> Path | None:
    """Restore ``path_in_repo`` from a model repo into ``local_dir/path_in_repo``.

    Used to resume sweeps after a lost runtime. Returns None when the repo or
    folder does not exist yet.
    """
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{path_in_repo}/**"],
            local_dir=str(local_dir),
            token=resolve_hf_token(token),
        )
    except Exception as exc:  # Missing repo/permissions: start the sweep fresh.
        print(f"Nothing restored from {repo_id}/{path_in_repo}: {exc}")
        return None
    restored = Path(local_dir) / path_in_repo
    return restored if restored.exists() else None


def upload_artifacts_folder(
    folder_path: str | Path,
    repo_id: str,
    *,
    path_in_repo: str | None = None,
    commit_message: str | None = None,
    private: bool = False,
    token: str | None = None,
) -> str:
    """Upload a local artifacts directory to a Hugging Face model repository."""
    from huggingface_hub import HfApi

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Artifacts folder not found: {folder}")

    resolved_token = resolve_hf_token(token)
    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        path_in_repo=path_in_repo or folder.name,
        commit_message=commit_message or f"Upload {folder.name} artifacts",
    )
    return str(info)
